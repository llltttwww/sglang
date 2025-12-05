#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import sys
from collections import Counter

import torch
from safetensors.torch import load_file as load_safetensors

# === 1. 把 sglang/python 加进 sys.path，确保能 import sglang ===
REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# 触发 sglang 那边注册 Qwen3KimiConfig（AutoConfig.register）
import sglang.srt.utils.hf_transformers_utils  # noqa: F401

from transformers import AutoConfig
# from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM
from sglang.srt.models.qwen3_next import EntryClass as Qwen3NextForCausalLM
from sglang.srt.configs.qwen3_next import Qwen3KimiConfig
import sglang.srt.models.qwen3_next as qwen3_mod
import sglang.srt.distributed.parallel_state as pstate  
from sglang.srt.layers import dp_attention as dp_attn
import sglang.srt.models.qwen2_moe as qwen2_moe
import sglang.srt.server_args as srt_server_args
# 单机单卡 stub：把 dp-attn 的 tp size / rank 固定成 1 / 0
dp_attn._ATTN_TP_SIZE = 1
dp_attn._ATTN_TP_RANK = 0
dp_attn.is_dp_attention_enabled = lambda: False
from types import SimpleNamespace

def load_ckpt_params(model_dir: str):
    """读取 ckpt 中的参数名（优先 safetensors）"""
    model_dir = Path(model_dir)
    ckpt_keys = []

    # 1) safetensors
    st_files = sorted(model_dir.glob("*.safetensors"))
    params_dict = {}
    if st_files:
        print(f"[INFO] Found safetensors files: {[f.name for f in st_files]}")
        for f in st_files:
            print(f"[INFO] Loading {f} ...")
            params = load_safetensors(str(f))
            for k in params.keys():
                params_dict[k] = None  # 只关心 key，不存值
        ckpt_keys = sorted(params_dict.keys())
        return ckpt_keys

    # 2) pytorch_model.bin
    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        print(f"[INFO] Loading {bin_path} ...")
        sd = torch.load(bin_path, map_location="cpu")
        ckpt_keys = sorted(sd.keys())
        return ckpt_keys

    raise FileNotFoundError(f"No *.safetensors or pytorch_model.bin found in {model_dir}")




def build_hf_model_and_get_keys(model_dir: str, trust_remote_code: bool = True):
    print(f"[INFO] Loading HF config via AutoConfig.from_pretrained from {model_dir}")
    config = Qwen3KimiConfig.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
    )
    print(f"[INFO] HF Config class: {type(config)}")
    print(f"[INFO] config.model_type: {getattr(config, 'model_type', None)}")
    print(f"[INFO] config.architectures: {getattr(config, 'architectures', None)}")

    torch.set_default_dtype(torch.float32)

    # ===== 单机单卡 fake sglang 环境 =====

    # 1) PP group
    qwen3_mod.get_pp_group = lambda: SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        rank_in_group=0,
        world_size=1,
    )

    # 2) TP group + rank/size
    pstate.get_tp_group = lambda: SimpleNamespace(
        rank_in_group=0,
        world_size=1,
    )
    pstate.get_tensor_model_parallel_rank = lambda: 0
    pstate.get_tensor_model_parallel_world_size = lambda: 1

    # 3) MoE expert parallel 相关
    pstate.get_moe_ep_group = lambda: SimpleNamespace(
        rank_in_group=0,
        world_size=1,
    )
    pstate.get_moe_expert_parallel_rank = lambda: 0
    pstate.get_moe_expert_parallel_world_size = lambda: 1

    # （有的地方还会用 data-parallel 版本，顺手 fake 一下，安全一点）
    pstate.get_moe_data_parallel_rank = lambda: 0
    pstate.get_moe_data_parallel_world_size = lambda: 1

    # 4) DP-attention 的 tp rank/size + enable flag
    dp_attn.get_attention_tp_rank = lambda: 0
    dp_attn.get_attention_tp_size = lambda: 1
    dp_attn.is_dp_attention_enabled = lambda: False

    # 5) global server args：给 Qwen2MoeSparseMoeBlock 用的
    from sglang.srt import server_args as s_args
    fake_args = SimpleNamespace(ep_num_redundant_experts=0)
    s_args.get_global_server_args = lambda: fake_args

    print("[INFO] Building sglang.srt.models.qwen3_next.EntryClass (Qwen3NextForCausalLM) ...")
    model = Qwen3NextForCausalLM(config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Built sglang EntryClass model, total params: {num_params / 1e6:.1f}M")

    hf_keys = sorted(name for name, _ in model.named_parameters())
    return hf_keys


def main():
    parser = argparse.ArgumentParser(
        description="Compare param names between HF/Kimi checkpoint and HF Qwen3Next implementation (with Qwen3KimiConfig)."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to HF model directory (same as --model-path used for sglang).",
    )
    args = parser.parse_args()
    model_dir = args.model_dir

    print("============================================================")
    print(f"[STEP 1] Scanning checkpoint in {model_dir}")
    ckpt_keys = load_ckpt_params(model_dir)
    print(f"[INFO] #ckpt params: {len(ckpt_keys)}")

    print("  Example ckpt keys:")
    for k in ckpt_keys[:20]:
        print("   -", k)

    print("============================================================")
    print(f"[STEP 2] Building HF Qwen3Next model and listing param names")
    hf_keys = build_hf_model_and_get_keys(model_dir)
    print(f"[INFO] #HF model params: {len(hf_keys)}")

    print("  Example HF model keys:")
    for k in hf_keys[:20]:
        print("   -", k)

    print("============================================================")
    print("[STEP 3] Compare name sets")

    set_ckpt = set(ckpt_keys)
    set_hf = set(hf_keys)

    only_in_hf = sorted(set_hf - set_ckpt)
    only_in_ckpt = sorted(set_ckpt - set_hf)

    print(f"[INFO] #params only in HF model (not in ckpt): {len(only_in_hf)}")
    for k in only_in_hf[:50]:
        print("   [HFONLY]", k)

    print("------------------------------------------------------------")
    print(f"[INFO] #params only in ckpt (not in HF model): {len(only_in_ckpt)}")
    for k in only_in_ckpt[:50]:
        print("   [CKPTONLY]", k)

    # 额外：按前缀聚合一下，帮助你看是哪些模块差异最大
    def prefix(key, n=3):
        parts = key.split(".")
        return ".".join(parts[:n])

    hf_only_prefix_cnt = Counter(prefix(k) for k in only_in_hf)
    ckpt_only_prefix_cnt = Counter(prefix(k) for k in only_in_ckpt)

    print("============================================================")
    print("[INFO] Top prefixes only in HF model:")
    for p, c in hf_only_prefix_cnt.most_common(20):
        print(f"   {p}: {c} keys")

    print("------------------------------------------------------------")
    print("[INFO] Top prefixes only in ckpt:")
    for p, c in ckpt_only_prefix_cnt.most_common(20):
        print(f"   {p}: {c} keys")

    print("============================================================")
    print("[DONE] Now you can inspect which modules differ (e.g. k_norm, v_norm, gates, delta/kda, etc.)")


if __name__ == "__main__":
    main()
