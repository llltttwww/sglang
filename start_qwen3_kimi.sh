# CUDA_VISIBLE_DEVICES=0 \
# python -m sglang.launch_server \
#   --model-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
#   --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
#   --host 0.0.0.0 \
#   --port 30000 \
#   --dtype bfloat16


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m sglang.launch_server \
  --model-path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--gated_full/snapshots/a8fe22a49eea06152a4281978141fe38db9fd943 \
  --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--gated_full/snapshots/a8fe22a49eea06152a4281978141fe38db9fd943 \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \