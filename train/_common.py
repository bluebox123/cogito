"""Shared helpers for the GRPO training entry points."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def have_flash_attn():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def attn_impl():
    return "flash_attention_2" if have_flash_attn() else "sdpa"


def latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not cks:
        return None
    cks.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, cks[-1])


def load_jsonl_dataset(path):
    from datasets import load_dataset
    return load_dataset("json", data_files=path, split="train")
