#!/usr/bin/env bash
# Start the TRL/vLLM rollout server on a dedicated GPU (default GPU 0), detached.
# Used by scripts/launch_*.sh. Env: MODEL, MAXLEN, VLLM_GPU, VLLM_PORT, VLLM_GPU_MEM.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"   # caches -> big disk; inherited by the nohup'd server
MODEL="${MODEL:-Qwen/Qwen2.5-3B}"
MAXLEN="${MAXLEN:-2048}"
GPU="${VLLM_GPU:-0}"
PORT="${VLLM_PORT:-8000}"
MEM="${VLLM_GPU_MEM:-0.85}"
mkdir -p "$ROOT/logs"

echo "[serve] vLLM rollout server: model=$MODEL gpu=$GPU maxlen=$MAXLEN port=$PORT"
CUDA_VISIBLE_DEVICES="$GPU" nohup trl vllm-serve \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$MEM" \
  --max-model-len "$MAXLEN" \
  --port "$PORT" \
  > "$ROOT/logs/vllm.log" 2>&1 &
echo $! > "$ROOT/logs/vllm.pid"
PID="$(cat "$ROOT/logs/vllm.pid")"
echo "[serve] pid=$PID ; waiting for startup (up to ~15 min on first model download)..."

for i in $(seq 1 360); do
  if grep -qiE "Application startup complete|Uvicorn running|init_communicator|Starting vLLM" "$ROOT/logs/vllm.log" 2>/dev/null; then
    echo "[serve] vLLM server READY."
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[serve] ERROR: vLLM process died. Last log lines:"; tail -n 40 "$ROOT/logs/vllm.log"; exit 1
  fi
  sleep 5
done
echo "[serve] WARNING: startup not confirmed after timeout; check logs/vllm.log"
exit 0
