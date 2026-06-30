"""GRPO training on GSM8K + MATH (Qwen2.5-3B) using TRL + vLLM (server mode).

Launch via scripts/launch_math.sh (starts vLLM rollout server on GPU 0 first).
Longer completions than Countdown, so per_device is smaller and the vLLM server
must be started with a larger --max-model-len (the launch script handles this).
"""
import argparse
import os
import sys

from _common import ROOT, attn_impl, latest_checkpoint, load_jsonl_dataset

sys.path.insert(0, ROOT)
from reward.math_answer import math_reward  # noqa: E402
from reward.format import format_reward  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--data", default=os.path.join(ROOT, "data", "math_train.jsonl"))
    p.add_argument("--output", default=os.path.join(ROOT, "checkpoints", "math"))
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()

    from trl import GRPOConfig, GRPOTrainer

    # num_generations=6 so per_device(4) x num_processes(3) = 12 is divisible by it
    # (works under every TRL batch-divisibility rule for the 3-GPU training split).
    num_generations, per_device, grad_accum, max_completion = 6, 4, 8, 2048
    max_steps = a.max_steps
    if a.smoke:
        num_generations, per_device, grad_accum, max_completion, max_steps = 4, 4, 1, 768, 5

    dataset = load_jsonl_dataset(a.data)

    cfg = GRPOConfig(
        output_dir=a.output,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_prompt_length=512,
        max_completion_length=max_completion,
        temperature=0.9,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        beta=0.001,
        max_steps=max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=2,
        save_steps=(5 if a.smoke else 50),
        save_strategy="steps",
        save_total_limit=3,
        report_to="tensorboard",
        logging_dir=os.path.join(ROOT, "runs", "math"),
        seed=42,
        log_completions=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="127.0.0.1",
        vllm_server_port=8000,
        model_init_kwargs={"torch_dtype": "bfloat16", "attn_implementation": attn_impl()},
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=a.model,
        args=cfg,
        reward_funcs=[math_reward, format_reward],
        train_dataset=dataset,
    )
    resume = None if (a.no_resume or a.smoke) else latest_checkpoint(a.output)
    if resume:
        print("[train] resuming from %s" % resume)
    trainer.train(resume_from_checkpoint=resume)
    final_dir = os.path.join(a.output, "final")
    trainer.save_model(final_dir)
    print("[train] DONE. final model at %s" % final_dir)


if __name__ == "__main__":
    main()
