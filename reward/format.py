"""Small format reward — kept tiny so answer-correctness always dominates.

The training prompt already opens "<think>", so a well-formed completion closes
it with </think> and then provides an <answer>...</answer> block. Full structure
in the right order earns 0.2; just having both tags earns 0.1; otherwise 0.0.
"""
import re

_ORDERED = re.compile(r"</think>\s*<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_HAS_ANSWER = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)


def _completion_text(c):
    if isinstance(c, str):
        return c
    try:
        return c[-1]["content"]
    except Exception:
        return str(c)


def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        text = _completion_text(c)
        if _ORDERED.search(text):
            rewards.append(0.2)
        elif "</think>" in text.lower() and _HAS_ANSWER.search(text):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards
