"""Verifiable reward for the Countdown task.

Parses the arithmetic expression the model put in <answer>...</answer> (or a
\\boxed{...} fallback), checks it uses each provided number exactly once and uses
only + - * / and parentheses, then safely evaluates it and compares to target.
Reward = 1.0 if exactly correct, else 0.0. No learned reward model.
"""
import ast
import operator
import re

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError("disallowed expression")


def safe_eval(expr):
    try:
        return _eval_node(ast.parse(expr, mode="eval"))
    except Exception:
        return None


def _completion_text(c):
    if isinstance(c, str):
        return c
    try:
        return c[-1]["content"]
    except Exception:
        return str(c)


def extract_expression(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    boxed = re.findall(r"\\boxed\{(.+?)\}", text, re.DOTALL)
    if boxed:
        return boxed[-1]
    return None


def countdown_reward(completions, nums=None, target=None, **kwargs):
    """TRL reward_func: list[float], one per completion (1.0 correct, else 0.0)."""
    rewards = []
    for i in range(len(completions)):
        text = _completion_text(completions[i])
        expr = extract_expression(text)
        r = 0.0
        if expr is not None and nums is not None and target is not None:
            cand = expr.strip().strip("`").replace("=", " ")
            lines = cand.splitlines()
            cand = lines[0].strip() if lines else cand
            if cand and re.fullmatch(r"[0-9+\-*/().\s]+", cand):
                used = [int(x) for x in re.findall(r"\d+", cand)]
                want = list(nums[i])
                if sorted(used) == sorted(want):
                    val = safe_eval(cand)
                    if val is not None and abs(float(val) - float(target[i])) < 1e-6:
                        r = 1.0
        rewards.append(r)
    return rewards
