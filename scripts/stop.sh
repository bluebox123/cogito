#!/usr/bin/env bash
# Cleanly stop training + the vLLM rollout server.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tmux kill-session -t cogito 2>/dev/null && echo "killed tmux 'cogito'" || true
[ -f "$ROOT/logs/train.pid" ] && kill "$(cat "$ROOT/logs/train.pid")" 2>/dev/null && echo "killed trainer pid" || true
[ -f "$ROOT/logs/vllm.pid" ]  && kill "$(cat "$ROOT/logs/vllm.pid")"  2>/dev/null && echo "killed vllm pid" || true
pkill -f "vllm-serve"      2>/dev/null || true
pkill -f "grpo_countdown.py" 2>/dev/null || true
pkill -f "grpo_math.py"      2>/dev/null || true
# Hard sweep: vLLM spawns EngineCore worker subprocesses that survive the parent
# kill and keep holding GPU memory. Kill anything running from THIS repo's venv
# (safe: other users are on their own envs). Then give the GPUs a moment to free.
sleep 2
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true   # vLLM renames its GPU workers to this
pkill -9 -f "$ROOT/venv/bin/python" 2>/dev/null || true
sleep 2
echo "stopped. (if a GPU still shows ~67GB used, run:"
echo "  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F',' '\$2+0>50000{print \$1+0}' | xargs -r kill -9 )"
