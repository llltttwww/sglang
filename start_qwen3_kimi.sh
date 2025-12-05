CUDA_VISIBLE_DEVICES=0 \
python -m sglang.launch_server \
  --model-path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-Kimi-1204/snapshots/fccadc7499f8a9bcef45f30f409143e01640bef7 \
  --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3-Next-Kimi-1204/snapshots/fccadc7499f8a9bcef45f30f409143e01640bef7 \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16
