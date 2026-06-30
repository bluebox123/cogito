"""Verifiable reward for math (GSM8K + MATH).

Extracts the model's final answer from \\boxed{...} (or <answer>...), then checks
equivalence against ground truth. Uses math_verify (symbolic/latex aware) when
available, with a numeric/string normalization fallback for plain numbers like
GSM8K. Reward = 1.0 if equivalent, else 0.0.
"""
import re

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAVE_MV = True
except Exception:
    _HAVE_MV = False


def _completion_text(c):
    if isinstance(c, str):
        return c
    try:
        return c[-1]["content"]
    except Exception:
        return str(c)


def extract_boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def extract_answer(text):
    b = extract_boxed(text)
    if b is not None:
        return b.strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1)
        bb = extract_boxed(inner)
        return (bb if bb is not None else inner).strip()
    return None


def _normalize(s):
    s = str(s).strip()
    for tok in ["\\,", "\\!", "\\ ", " ", ",", "$", "\\%", "%", "\\left", "\\right"]:
        s = s.replace(tok, "")
    return s


def _numeric_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def _verify(gt, pred):
    if _HAVE_MV:
        try:
            if bool(_mv_verify(_mv_parse(gt), _mv_parse(pred))):
                return True
        except Exception:
            pass
    a, b = _normalize(pred), _normalize(gt)
    if a != "" and a == b:
        return True
    return _numeric_equal(a, b)


def math_reward(completions, answer=None, **kwargs):
    """TRL reward_func: list[float], one per completion (1.0 correct, else 0.0)."""
    rewards = []
    for i in range(len(completions)):
        text = _completion_text(completions[i])
        pred = extract_answer(text)
        r = 0.0
        if pred is not None and answer is not None:
            try:
                if _verify(str(answer[i]), pred):
                    r = 1.0
            except Exception:
                r = 0.0
        rewards.append(r)
    return rewards
