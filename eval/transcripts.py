"""Collect 'aha moment' transcripts: sample the trained model and keep completions
that show self-correction / backtracking. Writes results/aha_transcripts.md."""
import argparse
import os

from _eval_utils import read_jsonl, generate, ROOT

PHRASES = ["wait", "let me reconsider", "let me re-check", "recheck", "actually,",
           "hmm", "that's not right", "that is not right", "i made a mistake",
           "let me try again", "on second thought", "but that", "let me double-check"]


def has_aha(text):
    tl = text.lower()
    return any(p in tl for p in PHRASES)


def run(model, n_prompts=64, samples=2, max_keep=15):
    rows = read_jsonl(os.path.join(ROOT, "data", "countdown_test.jsonl"), n_prompts)
    mp = os.path.join(ROOT, "data", "math500_test.jsonl")
    if os.path.exists(mp):
        rows += read_jsonl(mp, n_prompts)
    prompts = [r["prompt"] for r in rows]
    gens = generate(model, prompts, n=samples, temperature=0.9, max_tokens=1024, max_model_len=2560)

    keep = []
    for r, comps in zip(rows, gens):
        for c in comps:
            if has_aha(c):
                keep.append((r.get("task", "?"), r["prompt"], c))
                break
        if len(keep) >= max_keep:
            break

    out = os.path.join(ROOT, "results", "aha_transcripts.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Cogito - Aha-moment transcripts (self-correction during reasoning)\n\n")
        if not keep:
            f.write("_No self-correction phrases found in this sample. Try more prompts/samples._\n")
        for i, (task, prompt, comp) in enumerate(keep, 1):
            f.write("## %d. (%s)\n\n**Prompt**\n\n```\n%s\n```\n\n**Cogito (with <think>)**\n\n```\n<think>%s\n```\n\n---\n\n"
                    % (i, task, prompt.strip(), comp.strip()))
    print("[transcripts] wrote %d aha transcripts -> %s" % (len(keep), out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--samples", type=int, default=2)
    a = p.parse_args()
    run(a.model, a.n_prompts, a.samples)
