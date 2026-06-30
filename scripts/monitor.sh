#!/usr/bin/env bash
# One-shot status snapshot. Safe to run anytime; does not touch training.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "==================== GPUs ===================="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi
echo; echo "==================== tmux ===================="
tmux ls 2>/dev/null || echo "(no tmux sessions)"
echo; echo "============== last training log =============="
tail -n 22 "$ROOT/logs/train.log" 2>/dev/null || echo "(no logs/train.log yet)"
echo; echo "============== recent reward / length =============="
grep -oE "reward[^,}]*" "$ROOT/logs/train.log" 2>/dev/null | tail -n 4
grep -oE "completion[^,}]*length[^,}]*" "$ROOT/logs/train.log" 2>/dev/null | tail -n 4
echo; echo "vLLM server: $( (kill -0 $(cat "$ROOT/logs/vllm.pid" 2>/dev/null) 2>/dev/null && echo UP) || echo 'down/unknown')"
echo "Live stream:  tail -f $ROOT/logs/train.log"
echo "Live curves:  tensorboard --logdir $ROOT/runs --port 6006"
echo "Attach live:  tmux attach -t cogito   (detach with Ctrl-b then d)"
