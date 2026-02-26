# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only Qwen3Next MTP Speculative Decoding.

Qwen3-Next / Qwen3-Kimi checkpoints store MTP (a.k.a. NextN) weights under
`mtp.layers.*`. Those layers are full-attention transformer layers, even when
the target model is hybrid (linear + full). Therefore the MTP draft model must
be instantiated as full-attention-only to be compatible with EAGLE draft
attention backends (FlashInfer/Triton/FA3 multi-step).
"""
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_next import Qwen3HybridAttentionDecoderLayer, Qwen3NextForCausalLM
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class Qwen3NextMTPLayer(nn.Module):
    """One MTP layer.

    Matches checkpoint key structure:
      mtp.layers.{i}.eh_proj / enorm / hnorm / final_layernorm
      mtp.layers.{i}.transformer_layer.*
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.enorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.transformer_layer = Qwen3HybridAttentionDecoderLayer(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("transformer_layer", prefix),
            alt_stream=alt_stream,
        )
        self.final_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            raise ValueError("Qwen3NextMTPLayer expects token embeddings to be provided.")

        hidden_states = input_embeds
        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual = None
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual = self.transformer_layer(
                positions, hidden_states, residual, forward_batch=forward_batch
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.final_layernorm(hidden_states, residual)
            else:
                hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class Qwen3NextMTPModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        num_layers = getattr(config, "num_hidden_layers", 1)

        def _get_layer(idx: int, prefix: str) -> nn.Module:
            return Qwen3NextMTPLayer(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(num_layers, _get_layer, prefix=f"{prefix}.layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(
                input_ids,
                positions,
                forward_batch,
                input_embeds=hidden_states,
            )
        return hidden_states


class Qwen3NextForCausalLMMTP(Qwen3NextForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # if not set, model load will be broken in Qwen3NextForCausalLM load_weights()
        self.pp_group = get_pp_group()

        # Force the MTP draft model to be full-attention-only.
        # Qwen3-Kimi configs often provide a full `layer_types` list where the
        # first layer is linear_attention; leaving it unchanged would make the
        # draft model incompatible with EAGLE's multi-step draft attention
        # backends.
        num_mtp_layers = getattr(config, "num_nextn_predict_layers", None)
        if num_mtp_layers is None:
            num_mtp_layers = getattr(config, "mtp_num_layers", 1)
            setattr(config, "num_nextn_predict_layers", num_mtp_layers)

        config.num_hidden_layers = int(num_mtp_layers)
        config.full_attention_interval = 1
        config.layer_types = ["full_attention"] * config.num_hidden_layers

        self.model = Qwen3NextMTPModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        super().load_weights(weights, is_mtp=True)


EntryClass = [Qwen3NextForCausalLMMTP]
