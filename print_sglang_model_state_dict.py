#!/usr/bin/env python
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

# === 1. 把 sglang/python 加进 sys.path，确保能 import sglang ===
REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# 触发 AutoConfig.register("qwen3_kimi", Qwen3KimiConfig) 等注册逻辑
import sglang.srt.utils.hf_transformers_utils  # noqa: F401

# sglang 自己的 Config & Model
from sglang.srt.configs.qwen3_next import Qwen3KimiConfig
from sglang.srt.models.qwen3_next import EntryClass as Qwen3NextForCausalLM

# 我们要 patch 的模块
import sglang.srt.models.qwen3_next as qwen3_mod
import sglang.srt.distributed.parallel_state as pstate
from sglang.srt.layers import dp_attention as dp_attn
import sglang.srt.models.qwen2_moe as qwen2_moe
import sglang.srt.server_args as srt_server_args
import sglang.srt.utils.offloader as offloader


def apply_single_process_stubs():
    """给 sglang 打一套单机单卡 stub，避免各种 global state 断言。"""

    # 1) PP group
    qwen3_mod.get_pp_group = lambda: SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        rank_in_group=0,
        world_size=1,
    )

    # 2) TP group + TP rank/size
    pstate.get_tp_group = lambda: SimpleNamespace(
        rank_in_group=0,
        world_size=1,
    )
    pstate.get_tensor_model_parallel_rank = lambda: 0
    pstate.get_tensor_model_parallel_world_size = lambda: 1

    # 3) DP attention 的 tp size / rank + 开关
    dp_attn._ATTN_TP_SIZE = 1
    dp_attn._ATTN_TP_RANK = 0
    dp_attn.get_attention_tp_size = lambda: 1
    dp_attn.get_attention_tp_rank = lambda: 0
    dp_attn.is_dp_attention_enabled = lambda: False

    # 4) global server args：保证 MoE 里 `get_global_server_args().ep_num_redundant_experts` 可用
    dummy_args = SimpleNamespace(ep_num_redundant_experts=0)
    # server_args 模块自身
    srt_server_args.get_global_server_args = lambda: dummy_args
    # qwen2_moe 里是 `from ... import get_global_server_args`
    qwen2_moe.get_global_server_args = lambda: dummy_args

    # 5) offloader：不做任何 CPU/GPU offload，把 generator 展开成 list
    offloader.get_offloader = lambda: SimpleNamespace(
        wrap_modules=lambda modules: list(modules)
    )



def build_sglang_model_and_get_state_dict(model_dir: str):
    print(f"[INFO] Loading Qwen3KimiConfig.from_pretrained from {model_dir}")
    config = Qwen3KimiConfig.from_pretrained(model_dir)
    print(f"[INFO] Config class: {type(config)}")
    print(f"[INFO] config.model_type: {config.model_type}")
    print(f"[INFO] config.architectures: {config.architectures}")

    torch.set_default_dtype(torch.float32)

    # 打 stub（必须在构建模型之前）
    apply_single_process_stubs()

    print("[INFO] Building sglang.srt.models.qwen3_next.EntryClass (Qwen3NextForCausalLM) ...")
    model = Qwen3NextForCausalLM(config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Built sglang model, total params: {num_params / 1e6:.1f}M")

    # 不加载 ckpt，只看“裸模型”的参数结构
    state = model.state_dict()
    return state


def main():
    parser = argparse.ArgumentParser(
        description="Print all param names + shapes from sglang Qwen3Next EntryClass (with Qwen3KimiConfig)."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to HF model directory (same as --model-path used for sglang).",
    )
    args = parser.parse_args()

    state = build_sglang_model_and_get_state_dict(args.model_dir)
    print(f"[INFO] #sglang model params: {len(state)}")
    print("============================================================")
    for name, tensor in sorted(state.items()):
        print(f"{name:80s}  {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
