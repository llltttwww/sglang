# CUDA_VISIBLE_DEVICES=0 \
# python -m sglang.launch_server \
#   --model-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
#   --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
#   --host 0.0.0.0 \
#   --port 30000 \
#   --dtype bfloat16


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m sglang.launch_server \
  --model-path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3Kimi-1209-57343/snapshots/a79c58b3d78b28b6ab64e004edac30c9b6badce9 \
  --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/hf_cache/hub/models--yuchenFan--Qwen3Kimi-1209-57343/snapshots/a79c58b3d78b28b6ab64e004edac30c9b6badce9 \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16 \
  # --speculative-algorithm EAGLE \
  # --speculative-draft-model-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
  # --speculative-num-steps 3 \
  # --speculative-eagle-topk 1 \
