"""Shared eval utilities: load data, batch-generate with vLLM (offline), score,
and merge results into results/results.json."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reward.countdown import countdown_reward  # noqa: E402
from reward.math_answer import math_reward  # noqa: E402

RESULTS_JSON = os.path.join(ROOT, "results", "results.json")


def read_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def generate(model_path, prompts, n=1, temperature=0.0, max_tokens=1024,
             max_model_len=2048, gpu_mem=0.90, tp=None):
    """Returns list (aligned with prompts) of lists of n completion strings."""
    import torch
    from vllm import LLM, SamplingParams
    if tp is None:
        tp = max(1, torch.cuda.device_count())
    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
              max_model_len=max_model_len, tensor_parallel_size=tp)
    sp = SamplingParams(n=n, temperature=temperature,
                        top_p=(1.0 if temperature == 0 else 0.95), max_tokens=max_tokens)
    outs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outs]


def passk_metrics(correct_matrix, k):
    """correct_matrix: list per prompt of list[bool]. Returns pass@1 and pass@k."""
    n = len(correct_matrix)
    if n == 0:
        return 0.0, 0.0
    pass1 = sum(1 for row in correct_matrix if row and row[0]) / n
    passk = sum(1 for row in correct_matrix if any(row)) / n
    return pass1, passk


def score_countdown(rows, gens):
    out = []
    for r, comps in zip(rows, gens):
        rs = countdown_reward(comps, nums=[r["nums"]] * len(comps), target=[r["target"]] * len(comps))
        out.append([x > 0.5 for x in rs])
    return out


def score_math(rows, gens):
    out = []
    for r, comps in zip(rows, gens):
        rs = math_reward(comps, answer=[r["answer"]] * len(comps))
        out.append([x > 0.5 for x in rs])
    return out


def update_results(task, model_tag, metrics):
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    data = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data.setdefault(task, {})[model_tag] = metrics
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("[eval] %s / %s -> %s" % (task, model_tag, metrics))
