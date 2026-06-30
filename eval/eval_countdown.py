"""Eval Countdown solve-rate on the held-out test split. pass@1 (greedy) and pass@k."""
import argparse
import os

from _eval_utils import read_jsonl, generate, score_countdown, passk_metrics, update_results, ROOT


def run(model, tag, limit=512, k=1, temperature=None):
    rows = read_jsonl(os.path.join(ROOT, "data", "countdown_test.jsonl"), limit)
    prompts = [r["prompt"] for r in rows]
    temp = (0.0 if k == 1 else 0.8) if temperature is None else temperature
    gens = generate(model, prompts, n=k, temperature=temp, max_tokens=1024, max_model_len=2048)
    correct = score_countdown(rows, gens)
    pass1, passk = passk_metrics(correct, k)
    metrics = {"solve_rate_pass@1": round(pass1, 4), "pass@%d" % k: round(passk, 4), "n": len(rows)}
    update_results("countdown", tag, metrics)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="cogito")
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--k", type=int, default=1)
    a = p.parse_args()
    run(a.model, a.tag, a.limit, a.k)
