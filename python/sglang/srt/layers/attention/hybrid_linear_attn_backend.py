from typing import Optional, Union

import torch
from einops import rearrange

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.fla.kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_cuda, is_npu

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu


class MambaAttnBackendBase(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self._debug_counter = 0  # 控制打印次数

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None

        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )

                if forward_batch.spec_info.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrive_next_token
                    retrieve_next_sibling = forward_batch.spec_info.retrive_next_sibling
                    retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")
        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._forward_metadata(forward_batch)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self.forward_metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrive_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrive_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].copy_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            bs_without_pad = spec_info.retrive_next_token.shape[0]
            # print(spec_info.retrive_next_token, spec_info.retrive_next_sibling)
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache


class KimiLinearAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]

        # For Qwen3-Kimi all-linear checkpoints we must match the HF remote-code
        # implementation which uses fla-core ops (ShortConvolution + kda kernels).
        import fla.modules.convolution as fla_conv
        from fla.ops.kda import fused_recurrent_kda as fla_fused_recurrent_kda
        from fla.ops.kda.gate import fused_kda_gate as fla_fused_kda_gate

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = layer_cache.conv

        # KDA cache stores q/k/v conv states as a list of 3 tensors:
        #   [pool_size, D, W] (W == kernel_size)
        assert isinstance(conv_states, list) and len(conv_states) == 3, (
            f"Unexpected KDA conv state container: {type(conv_states)=}, {getattr(conv_states, '__len__', lambda: None)()=}"
        )
        q_conv_state, k_conv_state, v_conv_state = conv_states

        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        # q_conv_state = q_conv_state.transpose(-1, -2)
        # k_conv_state = k_conv_state.transpose(-1, -2)
        # v_conv_state = v_conv_state.transpose(-1, -2)

        # fla causal_conv1d_update doesn't support an indices indirection, so we
        # gather per-request caches, update them, then scatter back.
        cache_indices_i64 = cache_indices.to(torch.int64)
        valid_mask = cache_indices_i64 >= 0
        # MambaPool allocates cache lines in [0, size), and leaves an extra
        # all-zero line at the end (size). Use it as the padding slot.
        pad_idx = q_conv_state.shape[0] - 1
        safe_idx = torch.where(
            valid_mask, cache_indices_i64, torch.full_like(cache_indices_i64, pad_idx)
        )

        def _decode_conv(x_2d: torch.Tensor, cache_full: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
            # NOTE: Avoid fla's causal_conv1d_update here. We observed correctness
            # regressions under CUDA graph replay for Qwen3-Kimi all-linear (KDA),
            # likely due to the update kernel's internal state/workspace behavior.
            # Using causal_conv1d(backend="triton") with q_len=1 keeps it graph-safe
            # while remaining functionally equivalent.
            #
            # x_2d: [N, D] -> [1, N, D] (varlen via cu_seqlens=query_start_loc)
            cache_sel = cache_full.index_select(0, safe_idx).contiguous()
            y, cache_sel = fla_conv.causal_conv1d(
                x=x_2d.unsqueeze(0),
                weight=w,
                bias=b,
                residual=None,
                initial_state=cache_sel,
                output_final_state=True,
                activation="silu",
                backend="triton",
                cu_seqlens=query_start_loc,
            )
            # Scatter updated caches back. This is CUDA-graph friendly:
            # no data-dependent control flow or dynamic-shape indexing.
            cache_full.index_copy_(0, safe_idx, cache_sel)

            y = y.squeeze(0)
            # Zero-out padded rows without boolean indexing.
            y = y * valid_mask.to(dtype=y.dtype).unsqueeze(-1)
            return y

        q = _decode_conv(q_proj_states, q_conv_state, q_conv_weights, q_conv_bias)
        k = _decode_conv(k_proj_states, k_conv_state, k_conv_weights, k_conv_bias)
        v = _decode_conv(v_proj_states, v_conv_state, v_conv_weights, v_conv_bias)

        # SGLang uses packed (varlen) decoding: B=1, T=sum(seq_lens)=bs, with cu_seqlens.
        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()  # [N, H]

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fla_fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)  # [N, H, D]

        beta = beta.unsqueeze(0)  # [1, N, H]
        g = g.unsqueeze(0)  # [1, N, H, D]

        pad_state_idx = ssm_states.shape[0] - 1
        safe_state_idx = torch.where(
            valid_mask,
            cache_indices_i64,
            torch.full_like(cache_indices_i64, pad_state_idx),
        )
        initial_state = ssm_states.index_select(0, safe_state_idx).contiguous()
        (
            core_attn_out,
            last_recurrent_state,
        ) = fla_fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )
        ssm_states.index_copy_(
            0, safe_state_idx, last_recurrent_state.to(ssm_states.dtype, copy=False)
        )
        
        # def debug_check(name: str, x: torch.Tensor):
        #     if self._debug_counter >= 5:
        #         return
        #     with torch.no_grad():
        #         if not torch.isfinite(x).all():
        #             print(f"[DEBUG] {name} has NaN/Inf at layer {layer_id}")
        #             print("  any_nan:", torch.isnan(x).any().item())
        #             print("  any_inf:", torch.isinf(x).any().item())
        #             # 只在第一次直接炸，方便看到 call stack
        #             raise RuntimeError(f"{name} contains NaN/Inf")
        #         else:
        #             print(
        #                 f"[DEBUG] {name} OK at layer {layer_id}, "
        #                 f"mean={x.mean().item():.4f}, std={x.std().item():.4f}"
        #             )

        # # 只重点看出问题的第 4 层，不然日志太多
        # if layer_id == 4:
        #     debug_check("hidden_states", hidden_states)
        #     debug_check("q_proj_states", q_proj_states)
        #     debug_check("k_proj_states", k_proj_states)
        #     debug_check("v_proj_states", v_proj_states)
        
        # # === decode 数值监控 ===
        # with torch.no_grad():
        #     if not torch.isfinite(core_attn_out).all():
        #         print(f"[DEBUG] core_attn_out (decode) has NaN/Inf! layer_id={layer_id}")
        #         print("  any_nan:", torch.isnan(core_attn_out).any().item())
        #         print("  any_inf:", torch.isinf(core_attn_out).any().item())
        #         # 这里直接 raise 方便看到第一层 / 第一次炸的地方
        #         raise RuntimeError("core_attn_out decode contains NaN/Inf")
        #     else:
        #         # 如果太吵可以先注释掉
        #         print(f"[DEBUG] core_attn_out (decode) OK layer_id={layer_id}, "
        #               f"mean={core_attn_out.mean().item():.4f}, "
        #               f"std={core_attn_out.std().item():.4f}")

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        # Match HF remote-code implementation (fla-core).
        import fla.modules.convolution as fla_conv
        from fla.ops.kda import chunk_kda as fla_chunk_kda
        from fla.ops.kda import fused_recurrent_kda as fla_fused_recurrent_kda
        from fla.ops.kda.gate import fused_kda_gate as fla_fused_kda_gate

        q_proj_states = kwargs["q_proj_states"]
        k_proj_states = kwargs["k_proj_states"]
        v_proj_states = kwargs["v_proj_states"]
        q_conv_weights = kwargs["q_conv_weights"]
        k_conv_weights = kwargs["k_conv_weights"]
        v_conv_weights = kwargs["v_conv_weights"]

        q_conv_bias = kwargs["q_conv_bias"]
        k_conv_bias = kwargs["k_conv_bias"]
        v_conv_bias = kwargs["v_conv_bias"]

        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        b_proj = kwargs["b_proj"]
        f_a_proj = kwargs["f_a_proj"]
        f_b_proj = kwargs["f_b_proj"]
        hidden_states = kwargs["hidden_states"]
        head_dim = kwargs["head_dim"]
        layer_id = kwargs["layer_id"]

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states_all = mamba_cache_params.conv
        assert isinstance(conv_states_all, list) and len(conv_states_all) == 3, (
            f"Unexpected KDA conv state container: {type(conv_states_all)=}, {getattr(conv_states_all, '__len__', lambda: None)()=}"
        )
        conv_state_q, conv_state_k, conv_state_v = conv_states_all
        # # deal with strides
        # conv_state_q = conv_state_q.transpose(-1, -2)
        # conv_state_k = conv_state_k.transpose(-1, -2)
        # conv_state_v = conv_state_v.transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        cache_indices_i64 = cache_indices.to(torch.int64)
        valid_mask = cache_indices_i64 >= 0
        pad_idx = conv_state_q.shape[0] - 1
        safe_idx = torch.where(
            valid_mask, cache_indices_i64, torch.full_like(cache_indices_i64, pad_idx)
        )

        def _extend_conv(x_2d: torch.Tensor, cache_full: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
            # x_2d: [T, D] -> [1, T, D] for varlen mode (cu_seqlens)
            x = x_2d.unsqueeze(0)
            init = cache_full.index_select(0, safe_idx).contiguous()
            y, final_state = fla_conv.causal_conv1d(
                x=x,
                weight=w,
                bias=b,
                residual=None,
                initial_state=init,
                output_final_state=True,
                activation="silu",
                backend="triton",
                cu_seqlens=query_start_loc,
            )
            # Scatter back (CUDA-graph friendly).
            cache_full.index_copy_(0, safe_idx, final_state)
            # y: [1, T, D] -> [T, D]
            return y.squeeze(0)

        q = _extend_conv(q_proj_states, conv_state_q, q_conv_weights, q_conv_bias)
        k = _extend_conv(k_proj_states, conv_state_k, k_conv_weights, k_conv_bias)
        v = _extend_conv(v_proj_states, conv_state_v, v_conv_weights, v_conv_bias)

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fla_fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)

        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)

        pad_state_idx = ssm_states.shape[0] - 1
        safe_state_idx = torch.where(
            valid_mask,
            cache_indices_i64,
            torch.full_like(cache_indices_i64, pad_state_idx),
        )
        initial_state = ssm_states.index_select(0, safe_state_idx).contiguous()
        # HF reference uses fused_recurrent_kda for short prefill (q_len <= 64),
        # and chunk_kda for longer sequences.
        seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        if isinstance(seq_lens_cpu, list):
            max_q_len = max(seq_lens_cpu) if seq_lens_cpu else 0
        else:
            max_q_len = int(seq_lens_cpu.max().item())
        if max_q_len <= 64:
            core_attn_out, last_recurrent_state = fla_fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
        else:
            core_attn_out, last_recurrent_state = fla_chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
        ssm_states.index_copy_(
            0, safe_state_idx, last_recurrent_state.to(ssm_states.dtype, copy=False)
        )
        
        # # === extend 数值监控 ===
        # with torch.no_grad():
        #     if not torch.isfinite(core_attn_out).all():
        #         print(f"[DEBUG] core_attn_out (extend) has NaN/Inf! layer_id={layer_id}")
        #         print("  any_nan:", torch.isnan(core_attn_out).any().item())
        #         print("  any_inf:", torch.isinf(core_attn_out).any().item())
        #         raise RuntimeError("core_attn_out extend contains NaN/Inf")
        #     else:
        #         print(f"[DEBUG] core_attn_out (extend) OK layer_id={layer_id}, "
        #               f"mean={core_attn_out.mean().item():.4f}, "
        #               f"std={core_attn_out.std().item():.4f}")

        return core_attn_out


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend using Mamba kernel."""

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = layer_cache.conv
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        is_target_verify = forward_batch.forward_mode.is_target_verify()

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        retrieve_next_token = self.forward_metadata.retrieve_next_token
        retrieve_next_sibling = self.forward_metadata.retrieve_next_sibling
        retrieve_parent_token = self.forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = mamba_cache_params.conv
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            conv_states_to_use = conv_states.clone()
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0
            conv_states_to_use = conv_states

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = (
                mixed_qkv.view(batch_size, draft_token_num, -1)
                .transpose(1, 2)
                .contiguous()
            )
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states_to_use,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = (
                mixed_qkv_processed.transpose(1, 2).contiguous().view(seq_len, -1)
            )
        else:
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)

        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)

        if is_target_verify:
            core_attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_state_cache,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            recurrent_state = ssm_states[cache_indices]
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            ssm_states[cache_indices] = last_recurrent_state

        return core_attn_out


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata.query_start_loc,
            metadata.mamba_cache_indices,
            self.mamba_chunk_size,
            forward_batch,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        metadata = self._capture_metadata(bs, req_pool_indices, forward_mode, spec_info)
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata.query_start_loc, metadata.mamba_cache_indices, seq_lens
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        metadata = self._replay_metadata(
            bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu
        )
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata.query_start_loc, metadata.mamba_cache_indices, seq_lens
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_id: int,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        return mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.linear_attn_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(self, accepted_indices, model):
        request_number = accepted_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        intermediate_conv_window_cache = mamba_caches.intermediate_conv_window

        # SSM state updates (chunked to reduce peak memory)
        valid_mask = accepted_indices >= 0

        # Compute common indices once to avoid duplication
        valid_state_indices = state_indices_tensor[valid_mask].to(torch.int64)  # [N]
        last_steps = accepted_indices[valid_mask].to(torch.int64)  # [N]

        # scatter into ssm_states at the chosen cache lines
        ssm_states[:, valid_state_indices, :] = intermediate_state_cache[
            :, valid_state_indices, last_steps
        ].to(ssm_states.dtype, copy=False)

        # Scatter into conv_states at the chosen cache lines
        conv_states[:, valid_state_indices, :, :] = intermediate_conv_window_cache[
            :, valid_state_indices, last_steps
        ].to(conv_states.dtype, copy=False)
