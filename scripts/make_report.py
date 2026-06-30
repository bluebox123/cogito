"""Turn results/results.json into a before/after markdown table + a HF model card."""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "results.json")

TASK_ROWS = [
    ("countdown", "solve_rate_pass@1", "Countdown solve rate (pass@1)"),
    ("gsm8k", "pass@1", "GSM8K pass@1"),
    ("math500", "pass@1", "MATH-500 pass@1"),
]


def fmt(v):
    try:
        return "%.1f%%" % (100.0 * float(v))
    except Exception:
        return "-"


def build_table(data):
    lines = ["| Benchmark (held-out) | Base | Cogito |", "|---|---|---|"]
    for task, metric, label in TASK_ROWS:
        d = data.get(task, {})
        base = d.get("base", {}).get(metric)
        cog = d.get("cogito", {}).get(metric)
        if base is None and cog is None:
            continue
        lines.append("| %s | %s | %s |" % (label, fmt(base), fmt(cog)))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="your-username/cogito-3b")
    a = p.parse_args()
    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            data = json.load(f)
    table = build_table(data)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "before_after_table.md"), "w", encoding="utf-8") as f:
        f.write("# Cogito - before/after (base vs GRPO)\n\n" + table + "\n")

    card = (
        "---\n"
        "license: apache-2.0\n"
        "base_model: Qwen/Qwen2.5-3B\n"
        "library_name: transformers\n"
        "tags:\n  - grpo\n  - reasoning\n  - rl\n  - reinforcement-learning\n  - cogito\n"
        "---\n\n"
        "# Cogito - a reasoning model trained with GRPO (RL from verifiable rewards)\n\n"
        "Cogito is `Qwen/Qwen2.5-3B` after GRPO reasoning training on Countdown + math, "
        "with no reasoning supervision - only correctness rewards. It learns to produce a "
        "`<think>...</think>` chain-of-thought before answering, and response length grows "
        "over training (the DeepSeek-R1 signature).\n\n"
        "## Results\n\n" + table + "\n\n"
        "## Method\n\n"
        "GRPO (group-relative, no critic). Per step: sample G rollouts per prompt with vLLM, "
        "score with verifiable rewards (Countdown: expression evaluates to target using each "
        "number once; math: boxed answer matches), compute group-relative advantages, update.\n\n"
        "Recipe and eval harness: see the `recipe/` folder in this repo.\n"
    )
    with open(os.path.join(ROOT, "results", "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print("[report] wrote results/before_after_table.md and results/MODEL_CARD.md")
    print(table)


if __name__ == "__main__":
    main()
