CUDA_VISIBLE_DEVICES=0 \
python -m sglang.launch_server \
  --model-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
  --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16
