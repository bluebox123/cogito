#!/usr/bin/env bash
# Cleanly stop training + the vLLM rollout server.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tmux kill-session -t cogito 2>/dev/null && echo "killed tmux 'cogito'" || true
[ -f "$ROOT/logs/train.pid" ] && kill "$(cat "$ROOT/logs/train.pid")" 2>/dev/null && echo "killed trainer pid" || true
[ -f "$ROOT/logs/vllm.pid" ]  && kill "$(cat "$ROOT/logs/vllm.pid")"  2>/dev/null && echo "killed vllm pid" || true
pkill -f "vllm-serve"      2>/dev/null || true
pkill -f "grpo_countdown.py" 2>/dev/null || true
pkill -f "grpo_math.py"      2>/dev/null || true
echo "stopped."
