# CUDA_VISIBLE_DEVICES=0 \
# python -m sglang.launch_server \
#   --model-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
#   --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/hf_checkpoints/qwen3-kimi-1204-iter0000173 \
#   --host 0.0.0.0 \
#   --port 30000 \
#   --dtype bfloat16


CUDA_VISIBLE_DEVICES=1 \
python -m sglang.launch_server \
  --model-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen3-kimi-260212-decay-30B-sft-tulu3/iter_0021863_hf \
  --tokenizer-path /mnt/shared-storage-user/p1-shared/luotianwei/pretrain/posttrain/slime/checkpoints/qwen3-kimi-260212-decay-30B-sft-tulu3/iter_0021863_hf \
  --host 0.0.0.0 \
  --port 30001 \
  --dtype bfloat16 \
  --disable-cuda-graph \
  --enable-deterministic-inference \
  --random-seed 42 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  # --disable-radix-cache \
  # If you need deterministic outputs at temperature=0 (especially for MoE/TP), enable:
  # --enable-deterministic-inference \
  # --random-seed 42 \
  # For debugging cache-related nondeterminism across repeated prompts:
  # --disable-radix-cache \

