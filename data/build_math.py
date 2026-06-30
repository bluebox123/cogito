"""Download + build the math datasets.

TRAIN pool:  GSM8K (openai/gsm8k:main) + (best-effort) a Hendrycks MATH train
             mirror. If no MATH mirror loads, training proceeds on GSM8K alone.
EVAL only:   GSM8K test  +  MATH-500 (HuggingFaceH4/MATH-500). Never trained on.

Outputs JSONL with columns: prompt, answer, task.

Usage:
  python data/build_math.py            # full
  python data/build_math.py --smoke    # tiny (256 train, 64 eval each)
  python data/build_math.py --no-math  # GSM8K only (skip MATH train mirror)
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from reward.math_answer import extract_boxed  # noqa: E402


def load_template():
    with open(os.path.join(ROOT, "prompts", "system_math.txt"), "r", encoding="utf-8") as f:
        return f.read()


def gsm_answer(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("[math] wrote %s (%d rows)" % (os.path.basename(path), len(rows)))


def try_load_math_train(template, limit):
    """Return list of {prompt, answer, task} from the first MATH mirror that loads."""
    from datasets import load_dataset
    # (dataset_id, config_or_None, problem_col, answer_extractor)
    attempts = [
        ("nlile/hendrycks-MATH-benchmark", None, "problem", None),
        ("EleutherAI/hendrycks_math", "all", "problem", None),
    ]
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    rows = []
    for ds_id, cfg, pcol, _ in attempts:
        try:
            print("[math] trying MATH train mirror: %s" % ds_id)
            if ds_id == "EleutherAI/hendrycks_math":
                from datasets import concatenate_datasets
                parts = []
                for s in subjects:
                    parts.append(load_dataset(ds_id, s, split="train"))
                ds = concatenate_datasets(parts)
            else:
                ds = load_dataset(ds_id, split="train") if cfg is None else load_dataset(ds_id, cfg, split="train")
            for ex in ds:
                problem = ex.get(pcol) or ex.get("question")
                ans = ex.get("answer")
                if not ans:
                    ans = extract_boxed(ex.get("solution", "") or "")
                if not problem or not ans:
                    continue
                rows.append({"prompt": template.replace("{problem}", str(problem).strip()),
                             "answer": str(ans).strip(), "task": "math"})
                if limit and len(rows) >= limit:
                    break
            if rows:
                print("[math] loaded %d MATH train rows from %s" % (len(rows), ds_id))
                return rows
        except Exception as e:
            print("[math] could not load %s: %s" % (ds_id, e))
    print("[math] no MATH train mirror available -> GSM8K-only training")
    return []


def build(args):
    from datasets import load_dataset
    template = load_template()
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

    print("[math] loading GSM8K ...")
    gsm = load_dataset("openai/gsm8k", "main")

    train_rows = []
    for ex in gsm["train"]:
        train_rows.append({"prompt": template.replace("{problem}", ex["question"].strip()),
                           "answer": gsm_answer(ex["answer"]), "task": "gsm8k"})

    gsm_test = []
    for ex in gsm["test"]:
        gsm_test.append({"prompt": template.replace("{problem}", ex["question"].strip()),
                         "answer": gsm_answer(ex["answer"]), "task": "gsm8k"})

    if not args.no_math:
        train_rows += try_load_math_train(template, args.math_limit)

    # MATH-500 eval (best-effort)
    math500 = []
    try:
        m500 = load_dataset("HuggingFaceH4/MATH-500")
        split = m500["test"] if "test" in m500 else m500[list(m500.keys())[0]]
        for ex in split:
            ans = ex.get("answer") or extract_boxed(ex.get("solution", "") or "")
            if not ans:
                continue
            math500.append({"prompt": template.replace("{problem}", ex["problem"].strip()),
                            "answer": str(ans).strip(), "task": "math500"})
    except Exception as e:
        print("[math] could not load MATH-500: %s" % e)

    if args.smoke:
        train_rows = train_rows[:256]
        gsm_test = gsm_test[:64]
        math500 = math500[:64]

    import random
    random.Random(args.seed).shuffle(train_rows)
    _write(os.path.join(ROOT, "data", "math_train.jsonl"), train_rows)
    _write(os.path.join(ROOT, "data", "gsm8k_test.jsonl"), gsm_test)
    _write(os.path.join(ROOT, "data", "math500_test.jsonl"), math500)
    print("[math] DONE. train=%d  gsm8k_test=%d  math500_test=%d"
          % (len(train_rows), len(gsm_test), len(math500)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-math", action="store_true", help="skip the MATH train mirror, use GSM8K only")
    p.add_argument("--math-limit", type=int, default=0, help="cap MATH train rows (0 = all)")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    build(a)
