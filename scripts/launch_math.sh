#!/usr/bin/env bash
# Same 3-GPU background layout as launch_countdown.sh, for the GSM8K+MATH stage.
# Larger MAXLEN because math reasoning is longer.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
export MAXLEN="${MAXLEN:-3072}"
TRAIN_GPUS="${TRAIN_GPUS:-1,2}"
SMOKE_FLAG=""; [ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"
mkdir -p logs runs checkpoints

echo "[launch] (1/2) starting vLLM rollout server on GPU 0 (maxlen=$MAXLEN) ..."
MODEL="$MODEL" MAXLEN="$MAXLEN" VLLM_GPU=0 bash rollout/serve_vllm.sh

CMD="CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch --config_file configs/accelerate_ddp.yaml --num_processes 2 train/grpo_math.py --model $MODEL $SMOKE_FLAG"
echo "[launch] (2/2) starting training: $CMD"
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s cogito "cd '$ROOT' && $CMD 2>&1 | tee logs/train.log"
  echo "[launch] training running in tmux session 'cogito'."
else
  nohup bash -lc "cd '$ROOT' && $CMD" > logs/train.log 2>&1 &
  echo $! > logs/train.pid
  echo "[launch] training running under nohup (pid $(cat logs/train.pid))."
fi
echo "[launch] Monitor:  bash scripts/monitor.sh    Stop:  bash scripts/stop.sh"
