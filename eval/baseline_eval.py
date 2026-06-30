"""Measure the BASE model BEFORE any training (the 'before' column).
Run this once, right after building data, before launching training."""
import argparse

import eval_countdown
import eval_math


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["countdown", "math", "all"], default="all")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--k", type=int, default=1)
    a = p.parse_args()
    if a.task in ("countdown", "all"):
        eval_countdown.run(a.model, "base", a.limit, a.k)
    if a.task in ("math", "all"):
        eval_math.run(a.model, "base", a.limit, a.k)
    print("[baseline] done. Numbers stored under tag 'base' in results/results.json")
