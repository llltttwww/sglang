#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors


def load_ckpt_state_dict(model_dir: str):
    model_dir = Path(model_dir)

    # 优先 *.safetensors
    st_files = sorted(model_dir.glob("*.safetensors"))
    if st_files:
        print(f"[INFO] Found safetensors files: {[f.name for f in st_files]}")
        state = {}
        for f in st_files:
            print(f"[INFO] Loading {f} ...")
            sd = load_safetensors(str(f))
            state.update(sd)
        return state

    # 其次 pytorch_model.bin
    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        print(f"[INFO] Loading {bin_path} ...")
        state = torch.load(bin_path, map_location="cpu")
        return state

    raise FileNotFoundError(f"No *.safetensors or pytorch_model.bin found in {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Print all param names + shapes from HF checkpoint"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to HF model directory (same as --model-path).",
    )
    args = parser.parse_args()

    state = load_ckpt_state_dict(args.model_dir)
    print(f"[INFO] #ckpt params: {len(state)}")
    print("============================================================")
    import re


    def param_sort_key(name: str):
        # 顶层前缀：model / mtp / 其他
        prefix = name.split('.', 1)[0]
        if prefix == "model":
            prefix_rank = 0
        elif prefix == "mtp":
            prefix_rank = 1
        else:
            prefix_rank = 2  # 其他最后

        # layer 编号
        m_layer = re.search(r"\.layers\.(\d+)\.", name)
        layer = int(m_layer.group(1)) if m_layer else 10**6  # 没有 layer 的排最后

        # expert 编号（如果有）
        m_expert = re.search(r"\.experts\.(\d+)\.", name)
        expert = int(m_expert.group(1)) if m_expert else -1

        # 最后用 name 把同层、同 expert 的按名字稳定一下
        return (prefix_rank, layer, expert, name)

    for name, tensor in sorted(state.items(), key=lambda kv: param_sort_key(kv[0])):
        print(f"{name:80s}  {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
