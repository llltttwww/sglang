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


def _in_cuda_graph_capture() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        # 某些版本没有这个 API，保守一点：当成不在 capture
        return False

def _dbg_print(*args, **kwargs):
    if _in_cuda_graph_capture():
        return
    print(*args, **kwargs)

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
            # _dbg_print(spec_info.retrive_next_token, spec_info.retrive_next_sibling)
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
        
        import os

        def maybe_attach_debugpy(tag: str = ""):
            """在需要的地方调用这个函数，就可以让当前进程等待 VSCode attach。"""
            # if os.getenv("SGLANG_DEBUGPY", "0") != "1":
            #     return

            try:
                import debugpy
            except ImportError:
                _dbg_print("[DEBUGPY] debugpy not installed, skip attach")
                return

            if not debugpy.is_client_connected():
                # 这里可以改端口，但 5678 是默认习惯
                debugpy.listen(("0.0.0.0", 5678))
                _dbg_print(f"[DEBUGPY] Waiting for debugger attach on 5678... ({tag})")
                debugpy.wait_for_client()
                _dbg_print("[DEBUGPY] Debugger attached.")

            # 在这一行相当于打了一个断点
            debugpy.breakpoint()

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states = layer_cache.conv  # [pool_size, 3 * proj_size, state_len]

        # 本地 projection_size= q_proj_states 的最后一维
        proj_size = q_proj_states.shape[-1]

        # 安全起见加个断言（炸了至少知道是哪儿不对）
        assert conv_states.shape[1] == 3 * proj_size, (
            f"Unexpected conv dim: {conv_states.shape} vs proj_size={proj_size}"
        )

        # 按 dim=1 切成 q/k/v 三块，每块 [pool_size, proj_size, state_len]
        q_conv_state, k_conv_state, v_conv_state = conv_states.split(proj_size, dim=1)

        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        # q_conv_state = q_conv_state.transpose(-1, -2)
        # k_conv_state = k_conv_state.transpose(-1, -2)
        # v_conv_state = v_conv_state.transpose(-1, -2)

        q = causal_conv1d_update(
            q_proj_states,
            q_conv_state,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        k = causal_conv1d_update(
            k_proj_states,
            k_conv_state,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        v = causal_conv1d_update(
            v_proj_states,
            v_conv_state,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim), (q, k, v)
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()

        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)

        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)

        initial_state = ssm_states[cache_indices].contiguous()
        (
            core_attn_out,
            last_recurrent_state,
        ) = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )
        ssm_states[cache_indices] = last_recurrent_state
        
        # def debug_check(name: str, x: torch.Tensor):
        #     if self._debug_counter >= 5:
        #         return
        #     with torch.no_grad():
        #         if not torch.isfinite(x).all():
        #             _dbg_print(f"[DEBUG] {name} has NaN/Inf at layer {layer_id}")
        #             _dbg_print("  any_nan:", torch.isnan(x).any().item())
        #             _dbg_print("  any_inf:", torch.isinf(x).any().item())
        #             # 只在第一次直接炸，方便看到 call stack
        #             raise RuntimeError(f"{name} contains NaN/Inf")
        #         else:
        #             _dbg_print(
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
        #         _dbg_print(f"[DEBUG] core_attn_out (decode) has NaN/Inf! layer_id={layer_id}")
        #         _dbg_print("  any_nan:", torch.isnan(core_attn_out).any().item())
        #         _dbg_print("  any_inf:", torch.isinf(core_attn_out).any().item())
        #         # 这里直接 raise 方便看到第一层 / 第一次炸的地方
        #         raise RuntimeError("core_attn_out decode contains NaN/Inf")
        #     else:
        #         # 如果太吵可以先注释掉
        #         _dbg_print(f"[DEBUG] core_attn_out (decode) OK layer_id={layer_id}, "
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
        """
        普通 extend/prefill：沿用 causal_conv1d_fn + chunk_kda（快速）
        EAGLE verify(target_verify)：逐 step 跑 causal_conv1d_update + fused_recurrent_kda，
        并把每一步的 SSM/conv 状态写入 intermediate buffers，供 update_mamba_state_after_mtp_verify scatter。
        """
        from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_fn

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

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        
        # _dbg_print(f'@@@is_target_verify: {is_target_verify}')

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices_all = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        conv_states_all = mamba_cache_params.conv  # [pool_size, 3*proj, state_len]
        ssm_states = mamba_cache_params.temporal

        proj_size = q_proj_states.shape[-1]
        if conv_states_all.shape[1] != 3 * proj_size:
            raise RuntimeError(
                f"[KimiLinearAttnBackend] Unexpected conv dim: conv_states_all.shape={tuple(conv_states_all.shape)} "
                f"vs proj_size={proj_size} (expected conv_states_all.shape[1]==3*proj_size)"
            )

        # -----------------------------
        # helpers: write intermediate buffers with flexible shapes
        # -----------------------------
        def _store_intermediate_ssm(
            inter_ssm: torch.Tensor,
            layer_id_: int,
            pool_indices: torch.Tensor,  # [bs]
            step_t: int,
            state_tensor: torch.Tensor,  # [bs, ...] (structured or flat)
        ):
            """
            inter_ssm supported shapes:
            - [pool, steps, state_dim]                         (3D)
            - [layers, pool, steps, state_dim]                 (4D)
            - [pool, steps, ...state_shape...]                 (5D+)
            - [layers, pool, steps, ...state_shape...]         (6D+)
            """
            dim = inter_ssm.dim()
            st = state_tensor

            if dim == 3:
                # [pool, steps, state_dim]
                target = inter_ssm[pool_indices, step_t, :]  # [bs, state_dim]
                if st.dim() != 2:
                    st = st.reshape(st.shape[0], -1)
                if target.shape != st.shape:
                    raise RuntimeError(
                        f"[KimiLinearAttnBackend] intermediate_ssm (3D) shape mismatch: "
                        f"target={tuple(target.shape)} vs state={tuple(st.shape)}"
                    )
                target.copy_(st)

            elif dim == 4:
                # [layers, pool, steps, state_dim]
                target = inter_ssm[layer_id_, pool_indices, step_t, :]  # [bs, state_dim]
                if st.dim() != 2:
                    st = st.reshape(st.shape[0], -1)
                if target.shape != st.shape:
                    raise RuntimeError(
                        f"[KimiLinearAttnBackend] intermediate_ssm (4D) shape mismatch: "
                        f"target={tuple(target.shape)} vs state={tuple(st.shape)}"
                    )
                target.copy_(st)

            elif dim == 5:
                # [pool, steps, ...state_shape...]
                target = inter_ssm[pool_indices, step_t]  # [bs, ...]
                if target.shape != st.shape:
                    # allow flatten->reshape if inter expects flat and state is structured (unlikely in 5D)
                    if target.dim() == 2 and st.dim() > 2:
                        st2 = st.reshape(st.shape[0], -1)
                        if target.shape != st2.shape:
                            raise RuntimeError(
                                f"[KimiLinearAttnBackend] intermediate_ssm (5D) shape mismatch: "
                                f"target={tuple(target.shape)} vs state={tuple(st.shape)}"
                            )
                        target.copy_(st2)
                    else:
                        raise RuntimeError(
                            f"[KimiLinearAttnBackend] intermediate_ssm (5D) shape mismatch: "
                            f"target={tuple(target.shape)} vs state={tuple(st.shape)}"
                        )
                else:
                    target.copy_(st)

            elif dim == 6:
                # [layers, pool, steps, ...state_shape...]
                target = inter_ssm[layer_id_, pool_indices, step_t]  # [bs, ...]
                if target.shape != st.shape:
                    if target.dim() == 2 and st.dim() > 2:
                        st2 = st.reshape(st.shape[0], -1)
                        if target.shape != st2.shape:
                            raise RuntimeError(
                                f"[KimiLinearAttnBackend] intermediate_ssm (6D) shape mismatch: "
                                f"target={tuple(target.shape)} vs state={tuple(st.shape)}"
                            )
                        target.copy_(st2)
                    else:
                        raise RuntimeError(
                            f"[KimiLinearAttnBackend] intermediate_ssm (6D) shape mismatch: "
                            f"target={tuple(target.shape)} vs state={tuple(st.shape)}"
                        )
                else:
                    target.copy_(st)

            else:
                raise RuntimeError(
                    f"[KimiLinearAttnBackend] intermediate_ssm has unsupported dim={dim}, shape={tuple(inter_ssm.shape)}. "
                    "Expected 3/4/5/6D."
                )

        def _store_intermediate_conv(
            inter_conv: torch.Tensor,
            layer_id_: int,
            pool_indices: torch.Tensor,   # [bs]
            step_t: int,
            conv_cat: torch.Tensor,       # [bs, 3*proj, state_len]
        ):
            """
            inter_conv supported shapes:
            - [pool, steps, 3*proj, state_len]                 (4D)
            - [layers, pool, steps, 3*proj, state_len]         (5D)
            """
            dim = inter_conv.dim()
            if dim == 4:
                target = inter_conv[pool_indices, step_t, :, :]  # [bs, 3*proj, state_len]
                if target.shape != conv_cat.shape:
                    raise RuntimeError(
                        f"[KimiLinearAttnBackend] intermediate_conv_window (4D) shape mismatch: "
                        f"target={tuple(target.shape)} vs conv={tuple(conv_cat.shape)}"
                    )
                target.copy_(conv_cat)
            elif dim == 5:
                target = inter_conv[layer_id_, pool_indices, step_t, :, :]
                if target.shape != conv_cat.shape:
                    raise RuntimeError(
                        f"[KimiLinearAttnBackend] intermediate_conv_window (5D) shape mismatch: "
                        f"target={tuple(target.shape)} vs conv={tuple(conv_cat.shape)}"
                    )
                target.copy_(conv_cat)
            else:
                raise RuntimeError(
                    f"[KimiLinearAttnBackend] intermediate_conv_window has unsupported dim={dim}, "
                    f"shape={tuple(inter_conv.shape)}. Expected 4D or 5D."
                )

        # -------------------------------------------------------------------------
        # EAGLE verify (target_verify) 分支
        # -------------------------------------------------------------------------
        if is_target_verify:
            if forward_batch.spec_info is None:
                raise RuntimeError("[KimiLinearAttnBackend] target_verify=True but forward_batch.spec_info is None")

            if getattr(forward_batch.spec_info, "topk", 1) != 1:
                raise RuntimeError(
                    "[KimiLinearAttnBackend] This implementation currently supports --speculative-eagle-topk=1 only. "
                    "For topk>1 you must add retrieve_next_token/retrieve_next_sibling/retrieve_parent_token handling."
                )

            if not isinstance(mamba_cache_params, MambaPool.SpeculativeState):
                raise RuntimeError(
                    "[KimiLinearAttnBackend] target_verify=True requires MambaPool.SpeculativeState cache, "
                    f"but got {type(mamba_cache_params)}."
                )

            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window
            



            draft_token_num = int(forward_batch.spec_info.draft_token_num)
            seq_len = int(q_proj_states.shape[0])

            if seq_len % draft_token_num != 0:
                raise RuntimeError(
                    f"[KimiLinearAttnBackend] target_verify expects seq_len divisible by draft_token_num, "
                    f"but got seq_len={seq_len}, draft_token_num={draft_token_num}"
                )

            bs = seq_len // draft_token_num
            cache_indices = cache_indices_all[:bs].contiguous()
            
            import os
            if os.getenv("SGLANG_VERIFY_DEBUG", "1") == "1" and (not _in_cuda_graph_capture()):
                if layer_id == 0:
                    _dbg_print("\n=== [Kimi target_verify] buffers ===")
                    _dbg_print("draft_token_num:", int(forward_batch.spec_info.draft_token_num))
                    _dbg_print("bs:", int(bs), "seq_len:", int(q_proj_states.shape[0]))
                    _dbg_print("cache_indices[:8]:", cache_indices[:8].tolist())
                    _dbg_print("intermediate_state_cache.shape:", tuple(intermediate_state_cache.shape),
                        "dim:", intermediate_state_cache.dim(), "dtype:", intermediate_state_cache.dtype)
                    _dbg_print("intermediate_conv_window_cache.shape:", tuple(intermediate_conv_window_cache.shape),
                        "dim:", intermediate_conv_window_cache.dim(), "dtype:", intermediate_conv_window_cache.dtype)
                    _dbg_print("=== [Kimi target_verify] end ===\n")

            conv_state_q, conv_state_k, conv_state_v = conv_states_all.split(proj_size, dim=1)

            # verify: 临时滚动状态，避免直接污染主 cache
            conv_q = conv_state_q.clone()
            conv_k = conv_state_k.clone()
            conv_v = conv_state_v.clone()
            state = ssm_states[cache_indices].contiguous()

            # reshape tokens: [bs, steps, proj]
            q_ps = q_proj_states.view(bs, draft_token_num, proj_size).contiguous()
            k_ps = k_proj_states.view(bs, draft_token_num, proj_size).contiguous()
            v_ps = v_proj_states.view(bs, draft_token_num, proj_size).contiguous()

            # gating from hidden_states
            beta_full = b_proj(hidden_states)[0].float().sigmoid()          # [seq_len, ...]
            raw_g_full = f_b_proj(f_a_proj(hidden_states)[0])[0]            # [seq_len, ...]
            g_full = fused_kda_gate(raw_g_full, A_log, head_dim, g_bias=dt_bias)  # [seq_len, ...]

            beta_full = beta_full.view(bs, draft_token_num, -1).contiguous()
            g_full = g_full.view(bs, draft_token_num, -1).contiguous()

            cu = torch.arange(0, bs + 1, dtype=torch.int32, device=q_proj_states.device)

            outs = []

            for t in range(draft_token_num):
                q_t = causal_conv1d_update(
                    q_ps[:, t, :],
                    conv_q,
                    q_conv_weights,
                    q_conv_bias,
                    activation="silu",
                    conv_state_indices=cache_indices,
                )
                k_t = causal_conv1d_update(
                    k_ps[:, t, :],
                    conv_k,
                    k_conv_weights,
                    k_conv_bias,
                    activation="silu",
                    conv_state_indices=cache_indices,
                )
                v_t = causal_conv1d_update(
                    v_ps[:, t, :],
                    conv_v,
                    v_conv_weights,
                    v_conv_bias,
                    activation="silu",
                    conv_state_indices=cache_indices,
                )

                q_t, k_t, v_t = map(
                    lambda x: rearrange(x, "b (h d) -> 1 b h d", d=head_dim),
                    (q_t, k_t, v_t),
                )

                beta_t = beta_full[:, t, :].unsqueeze(0)
                g_t = g_full[:, t, :].unsqueeze(0)

                out_t, state = fused_recurrent_kda(
                    q=q_t,
                    k=k_t,
                    v=v_t,
                    g=g_t,
                    beta=beta_t,
                    initial_state=state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu,
                )
                outs.append(out_t)

                # ---- store intermediate states ----
                state_to_store = state.to(intermediate_state_cache.dtype, copy=False)
                _store_intermediate_ssm(
                    intermediate_state_cache,
                    layer_id,
                    cache_indices,
                    t,
                    state_to_store,
                )

                conv_cat = torch.cat(
                    [conv_q[cache_indices], conv_k[cache_indices], conv_v[cache_indices]],
                    dim=1,
                ).to(intermediate_conv_window_cache.dtype, copy=False)  # [bs, 3*proj, state_len]
                _store_intermediate_conv(
                    intermediate_conv_window_cache,
                    layer_id,
                    cache_indices,
                    t,
                    conv_cat,
                )
                
                import os
                if os.getenv("SGLANG_VERIFY_DEBUG", "1") == "1" and (not _in_cuda_graph_capture()):
                    if layer_id == 0 and t < 2:
                        with torch.no_grad():
                            _dbg_print(f"[Kimi verify] t={t} state mean/std:",
                                float(state.float().mean().item()),
                                float(state.float().std().item()))
                            # 读回你刚写的 intermediate（用最保守的方式：只打印 shape，不硬编码维度）
                            try:
                                # 尝试按 “layers,pool,step,...” 读一条
                                if intermediate_state_cache.dim() == 5:
                                    # [pool, steps, ...]
                                    x = intermediate_state_cache[cache_indices[0], t]
                                elif intermediate_state_cache.dim() == 6:
                                    # [layers, pool, steps, ...]
                                    x = intermediate_state_cache[layer_id, cache_indices[0], t]
                                else:
                                    x = intermediate_state_cache[cache_indices[0], t]
                                _dbg_print(f"[Kimi verify] inter_ssm readback t={t} shape:", tuple(x.shape),
                                    "mean/std:", float(x.float().mean().item()), float(x.float().std().item()))
                            except Exception as e:
                                _dbg_print("[Kimi verify] readback inter_ssm failed:", repr(e))


            outs_stacked = torch.stack([o.squeeze(0) for o in outs], dim=1)  # [bs, steps, h, d]
            core_attn_out = outs_stacked.reshape(
                1, bs * draft_token_num, outs_stacked.shape[2], outs_stacked.shape[3]
            ).contiguous()

            return core_attn_out

        # -------------------------------------------------------------------------
        # 非 verify：走 fast extend 路径（你原来的逻辑）
        # -------------------------------------------------------------------------
        conv_state_q, conv_state_k, conv_state_v = conv_states_all.split(proj_size, dim=1)

        has_initial_state = (forward_batch.extend_prefix_lens is not None) and (forward_batch.extend_prefix_lens > 0)

        q_proj_states_t = q_proj_states.transpose(0, 1)
        k_proj_states_t = k_proj_states.transpose(0, 1)
        v_proj_states_t = v_proj_states.transpose(0, 1)

        q = causal_conv1d_fn(
            q_proj_states_t,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_states=conv_state_q,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices_all,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        k = causal_conv1d_fn(
            k_proj_states_t,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_states=conv_state_k,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices_all,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        v = causal_conv1d_fn(
            v_proj_states_t,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_states=conv_state_v,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices_all,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=head_dim),
            (q, k, v),
        )

        beta = b_proj(hidden_states)[0].float().sigmoid()
        g = f_b_proj(f_a_proj(hidden_states)[0])[0]
        g = fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)

        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)

        initial_state = ssm_states[cache_indices_all].contiguous()
        core_attn_out, last_recurrent_state = chunk_kda(
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
        ssm_states[cache_indices_all] = last_recurrent_state

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
        
        import os

        if os.getenv("SGLANG_VERIFY_DEBUG", "1") == "1" and (not _in_cuda_graph_capture()):
            _dbg_print("\n=== [update_mamba_state_after_mtp_verify] ===")
            _dbg_print("accepted_indices:", accepted_indices.tolist())
            _dbg_print("state_indices_tensor[:min]:", state_indices_tensor[:min(8, state_indices_tensor.numel())].tolist())

            _dbg_print("ssm_states.shape:", tuple(ssm_states.shape))
            _dbg_print("intermediate_state_cache.shape:", tuple(intermediate_state_cache.shape))
            _dbg_print("conv_states.shape:", tuple(conv_states.shape))
            _dbg_print("intermediate_conv_window_cache.shape:", tuple(intermediate_conv_window_cache.shape))

            _dbg_print("valid N:", int(valid_mask.sum().item()))
            if valid_mask.any():
                _dbg_print("valid_state_indices min/max:",
                    int(valid_state_indices.min().item()), int(valid_state_indices.max().item()))
                _dbg_print("last_steps min/max:",
                    int(last_steps.min().item()), int(last_steps.max().item()))

                # 关键：检查你假设的 step 维度到底是不是 dim=2
                if intermediate_state_cache.dim() >= 3:
                    step_dim = intermediate_state_cache.shape[2]  # 你现在代码就是按这个在用
                    _dbg_print("intermediate_state_cache step_dim(shape[2]):", int(step_dim))
                    _dbg_print("OOB(last_steps >= step_dim):", bool((last_steps >= step_dim).any().item()))
            _dbg_print("=== [update_mamba_state_after_mtp_verify] end ===\n")
