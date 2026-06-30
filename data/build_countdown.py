"""Generate the Countdown dataset locally (download-free, contamination-free).

Each puzzle = a list of 3-4 numbers + a target reachable by combining them with
+ - * / (each number used once). We build the target by construction so every
example is guaranteed solvable. Train/test are split AFTER de-duplicating on
(sorted numbers, target), so the held-out test set never overlaps train.

Usage:
  python data/build_countdown.py              # full: 50k train / 1k test
  python data/build_countdown.py --smoke      # tiny: 256 train / 64 test
"""
import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_template():
    with open(os.path.join(ROOT, "prompts", "system_countdown.txt"), "r", encoding="utf-8") as f:
        return f.read()


def gen_one(rng, n_min, n_max, val_max, target_max):
    while True:
        n = rng.randint(n_min, n_max)
        nums = [rng.randint(1, val_max) for _ in range(n)]
        order = nums[:]
        rng.shuffle(order)
        acc = order[0]
        for v in order[1:]:
            op = rng.choice(["+", "-", "*", "/"])
            if op == "+":
                acc = acc + v
            elif op == "-":
                acc = acc - v
            elif op == "*":
                acc = acc * v
            else:
                acc = acc // v if (v != 0 and acc % v == 0) else acc + v
        if isinstance(acc, int) and 1 <= acc <= target_max:
            return nums, acc


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(args):
    rng = random.Random(args.seed)
    template = load_template()
    seen = set()
    rows = []
    need = args.train_size + args.test_size
    attempts = 0
    while len(rows) < need and attempts < need * 100:
        attempts += 1
        nums, target = gen_one(rng, args.n_min, args.n_max, args.val_max, args.target_max)
        key = (tuple(sorted(nums)), target)
        if key in seen:
            continue
        seen.add(key)
        prompt = template.replace("{nums}", str(nums)).replace("{target}", str(target))
        rows.append({"prompt": prompt, "nums": nums, "target": target, "task": "countdown"})
    rng.shuffle(rows)
    test = rows[: args.test_size]
    train = rows[args.test_size:]
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    _write(os.path.join(ROOT, "data", "countdown_train.jsonl"), train)
    _write(os.path.join(ROOT, "data", "countdown_test.jsonl"), test)
    print("[countdown] train=%d test=%d  (disjoint by (sorted nums, target))" % (len(train), len(test)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=int, default=50000)
    p.add_argument("--test-size", type=int, default=1024)
    p.add_argument("--n-min", type=int, default=3)
    p.add_argument("--n-max", type=int, default=4)
    p.add_argument("--val-max", type=int, default=99)
    p.add_argument("--target-max", type=int, default=999)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.train_size, a.test_size = 256, 64
    build(a)
