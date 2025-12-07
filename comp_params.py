from safetensors.torch import load_file

base_path = "/mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-Kimi-1204/snapshots/fccadc7499f8a9bcef45f30f409143e01640bef7/model-00000-of-00001.safetensors"
sft_path  = "/mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173/model-00000-of-00001.safetensors"

base_tensors = load_file(base_path)
sft_tensors  = load_file(sft_path)

base_keys = {k for k in base_tensors.keys() if "linear_attn" in k}
sft_keys  = {k for k in sft_tensors.keys()  if "linear_attn" in k}

print("[base only] ===========")
for k in sorted(base_keys - sft_keys):
    print(k)

print("\n[sft only] ===========")
for k in sorted(sft_keys - base_keys):
    print(k)

