#!/usr/bin/env python
import os
import sys
from typing import Optional

import torch
from safetensors import safe_open

def check_file(path: str, name_filter: Optional[str] = None) -> bool:
    """
    返回值: 该文件中是否发现 NaN/Inf
    """
    print(f"[CHECK] file: {path}")
    has_bad = False
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if name_filter is not None and name_filter not in key:
                continue

            t = f.get_tensor(key)
            if not torch.is_floating_point(t):
                continue

            finite_mask = torch.isfinite(t)
            if not finite_mask.all():
                # 统计一下 NaN / Inf 数量
                nan_cnt = torch.isnan(t).sum().item()
                inf_cnt = torch.isinf(t).sum().item()
                print(f"  [BAD] {key}: nan={nan_cnt}, inf={inf_cnt}, "
                      f"shape={tuple(t.shape)}")
                # min/max 也打印一下（注意要先把非 finite 的屏蔽掉）
                finite_vals = t[finite_mask]
                if finite_vals.numel() > 0:
                    print(f"       finite_min={finite_vals.min().item():.6g}, "
                          f"finite_max={finite_vals.max().item():.6g}")
                else:
                    print("       (no finite values)")
                has_bad = True
    return has_bad


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_safetensors_nan.py <ckpt_dir_or_file> "
              "[name_filter(optional)]")
        sys.exit(1)

    target = sys.argv[1]
    name_filter = sys.argv[2] if len(sys.argv) >= 3 else None

    any_bad = False
    if os.path.isfile(target) and target.endswith(".safetensors"):
        any_bad |= check_file(target, name_filter)
    else:
        # 遍历目录下所有 .safetensors
        for root, _, files in os.walk(target):
            for fn in files:
                if not fn.endswith(".safetensors"):
                    continue
                full = os.path.join(root, fn)
                bad = check_file(full, name_filter)
                any_bad |= bad

    if any_bad:
        print("\n[RESULT] Found NaN/Inf in checkpoint tensors.")
        sys.exit(1)
    else:
        print("\n[RESULT] No NaN/Inf found in checked tensors.")
        sys.exit(0)


if __name__ == "__main__":
    main()
