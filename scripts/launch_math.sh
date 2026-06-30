#!/usr/bin/env bash
# Same 4-GPU background layout as launch_countdown.sh, for the GSM8K+MATH stage.
#   GPU 0      -> vLLM rollout server   |   GPU 1,2,3 -> training (3 processes)
# Larger MAXLEN because math reasoning is longer.
#
# Env knobs:  MODEL (Qwen/Qwen2.5-3B), MAXLEN (3072), SMOKE (0/1),
#             VLLM_GPU (0), TRAIN_GPUS (1,2,3), NPROC (3)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/env.sh"            # caches -> big disk
export MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
export MAXLEN="${MAXLEN:-3072}"
VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2,3}"
NPROC="${NPROC:-3}"
SMOKE_FLAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
mkdir -p logs runs checkpoints

echo "[launch] (1/2) starting vLLM rollout server on GPU $VLLM_GPU (maxlen=$MAXLEN) ..."
MODEL="$MODEL" MAXLEN="$MAXLEN" VLLM_GPU="$VLLM_GPU" bash rollout/serve_vllm.sh

CMD="CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch --config_file configs/accelerate_ddp.yaml --num_processes $NPROC train/grpo_math.py --model $MODEL $SMOKE_FLAG"
echo "[launch] (2/2) starting training: $CMD"
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s cogito "cd '$ROOT' && source scripts/env.sh && $CMD 2>&1 | tee logs/train.log"
  echo "[launch] training running in tmux session 'cogito'."
else
  nohup bash -lc "cd '$ROOT' && source scripts/env.sh && $CMD" > logs/train.log 2>&1 &
  echo $! > logs/train.pid
  echo "[launch] training running under nohup (pid $(cat logs/train.pid))."
fi
echo "[launch] Monitor:  bash scripts/monitor.sh    Stop:  bash scripts/stop.sh"
