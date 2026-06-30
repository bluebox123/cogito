"""Eval GSM8K + MATH-500 pass@1 (greedy) and pass@k on held-out splits."""
import argparse
import os

from _eval_utils import read_jsonl, generate, score_math, passk_metrics, update_results, ROOT


def _run_one(model, tag, filename, task, limit, k, max_tokens):
    path = os.path.join(ROOT, "data", filename)
    if not os.path.exists(path):
        print("[eval] %s missing (run data/build_math.py first) - skipping" % filename)
        return None
    rows = read_jsonl(path, limit)
    prompts = [r["prompt"] for r in rows]
    temp = 0.0 if k == 1 else 0.8
    gens = generate(model, prompts, n=k, temperature=temp, max_tokens=max_tokens, max_model_len=max_tokens + 600)
    correct = score_math(rows, gens)
    pass1, passk = passk_metrics(correct, k)
    metrics = {"pass@1": round(pass1, 4), "pass@%d" % k: round(passk, 4), "n": len(rows)}
    update_results(task, tag, metrics)
    return metrics


def run(model, tag, limit=300, k=1):
    _run_one(model, tag, "gsm8k_test.jsonl", "gsm8k", limit, k, max_tokens=1024)
    _run_one(model, tag, "math500_test.jsonl", "math500", limit, k, max_tokens=2048)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="cogito")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--k", type=int, default=1)
    a = p.parse_args()
    run(a.model, a.tag, a.limit, a.k)
