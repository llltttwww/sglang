#!/usr/bin/env python3
import argparse
import copy
import sys
from pathlib import Path

# Ensure the local `python/` package root is importable when running from the repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))

# Register SGLang custom HF configs (e.g., `qwen3_kimi`) into transformers AutoConfig.
import sglang.srt.utils.hf_transformers_utils  # noqa: F401

from transformers import AutoConfig


def _summarize_layer_types(cfg):
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None:
        return None
    return {
        "len": len(layer_types),
        "head": layer_types[: min(8, len(layer_types))],
        "tail": layer_types[-min(8, len(layer_types)) :],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check Qwen3Next/Qwen3-Kimi MTP (NextN) draft config overrides used by speculative decoding."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local HF model directory (same as --model-path for sglang.launch_server).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers AutoConfig loader.",
    )
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )

    print("=== Target config ===")
    print("model_type:", getattr(cfg, "model_type", None))
    print("architectures:", getattr(cfg, "architectures", None))
    print("num_hidden_layers:", getattr(cfg, "num_hidden_layers", None))
    print("full_attention_interval:", getattr(cfg, "full_attention_interval", None))
    print("mtp_num_layers:", getattr(cfg, "mtp_num_layers", None))
    print("num_nextn_predict_layers:", getattr(cfg, "num_nextn_predict_layers", None))
    print("layer_types:", _summarize_layer_types(cfg))
    if hasattr(cfg, "full_attention_layer_ids"):
        print("full_attention_layer_ids:", cfg.full_attention_layer_ids[:16])
    if hasattr(cfg, "linear_layer_ids"):
        print("linear_layer_ids:", cfg.linear_layer_ids[:16])

    # Apply the same overrides we use for the draft (MTP) model.
    draft_cfg = copy.deepcopy(cfg)
    draft_layers = getattr(draft_cfg, "mtp_num_layers", None)
    if draft_layers is None:
        draft_layers = getattr(draft_cfg, "num_nextn_predict_layers", None)
    if draft_layers is None:
        draft_layers = 1

    draft_cfg.num_nextn_predict_layers = int(draft_layers)
    draft_cfg.num_hidden_layers = int(draft_layers)
    draft_cfg.full_attention_interval = 1
    draft_cfg.layer_types = ["full_attention"] * int(draft_layers)

    print("\n=== Draft (MTP) config after overrides ===")
    print("num_hidden_layers:", getattr(draft_cfg, "num_hidden_layers", None))
    print("num_nextn_predict_layers:", getattr(draft_cfg, "num_nextn_predict_layers", None))
    print("full_attention_interval:", getattr(draft_cfg, "full_attention_interval", None))
    print("layer_types:", _summarize_layer_types(draft_cfg))
    if hasattr(draft_cfg, "full_attention_layer_ids"):
        print("full_attention_layer_ids:", draft_cfg.full_attention_layer_ids[:16])
    if hasattr(draft_cfg, "linear_layer_ids"):
        print("linear_layer_ids:", draft_cfg.linear_layer_ids[:16])


if __name__ == "__main__":
    main()
