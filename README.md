# Cogito — a reasoning model trained with GRPO (RL from verifiable rewards)

Take `Qwen2.5-3B` (a base model that can't reason well) and use **GRPO** —
reinforcement learning from *verifiable* rewards — to make it teach itself
step-by-step reasoning. No reasoning supervision: only "did it get the right
answer". Reasoning emerges, response length grows over training (the DeepSeek-R1
signature), and we measure a clean before/after on held-out math + Countdown.

This repo is built to run on a **3× A100 80GB** Jupyter box. You build nothing
locally except this folder — push it to GitHub, `git clone` on the box, run a few
commands. **Training runs in the background and survives closing the browser.**

---

## How the 3 GPUs are used

```
GPU 0  ─────────────►  vLLM rollout server  (generates the G samples per prompt — the GPU sink)
GPU 1  ─┐
GPU 2  ─┴────────────►  GRPO training        (accelerate DDP, bf16, gradient checkpointing)
```

vLLM and the trainer MUST be on separate GPUs (TRL requirement). The launch
scripts set this up for you automatically.

---

## Prerequisites (one-time)

1. A **GitHub account** + an empty repo (e.g. `cogito`). Create it at github.com (no README).
2. A **Hugging Face account** + a **write token**: huggingface.co → Settings → Access Tokens → New token (role: *Write*). You'll paste it on the box.
3. A cloud GPU box with **3× A100 80GB** and a **JupyterLab terminal** (RunPod, Lambda, Vast, Paperspace, etc.).

---

## STEP 1 — push this folder to GitHub (on your local Windows, PowerShell)

```powershell
cd "C:\Users\samarth\Desktop\Coursework\final_year_projects\Cogito"
git init
git add .
git commit -m "Cogito GRPO pipeline"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/cogito.git
git push -u origin main
```

If `git push` asks you to log in, use your GitHub username + a **Personal Access
Token** as the password (github.com → Settings → Developer settings → Tokens).

> Whenever I update files later, you just run these two locally:
> `git add . ; git commit -m "update" ; git push` — then `git pull` on the box.

---

## STEP 2 — on the Jupyter box (open a Terminal)

```bash
git clone https://github.com/<YOUR-USERNAME>/cogito.git
cd cogito
bash setup.sh                 # installs vllm/trl/etc, checks the 3 GPUs (~10–20 min first time)
huggingface-cli login         # paste your HF *write* token
```

`setup.sh` prints your GPU count at the end — confirm it says **GPU count: 3**.

---

## STEP 3 — smoke test FIRST (mandatory, ~10 min)

This proves the *entire* pipeline works (data → vLLM server → training →
checkpoint) on a tiny model + 5 steps, before you spend hours. Do not skip it.

```bash
python data/build_countdown.py --smoke
SMOKE=1 MODEL=Qwen/Qwen2.5-Math-1.5B MAXLEN=1024 bash scripts/launch_countdown.sh
bash scripts/monitor.sh
```

Watch `bash scripts/monitor.sh` until you see steps progressing and all 3 GPUs
busy. When it finishes (5 steps), clean up before the real run:

```bash
bash scripts/stop.sh
```

If the smoke test fails, read **Troubleshooting** below — fixing it now is cheap.

---

## STEP 4 — Countdown training (the real run, in the background)

```bash
python data/build_countdown.py                       # ~50k train / 1k held-out test
python eval/baseline_eval.py --task countdown        # measures the BASE model (the "before")
bash scripts/launch_countdown.sh                     # trains on 3 GPUs, in the background
```

Now you can **close the browser / shut your laptop** — training keeps running.
Reconnect anytime and check:

```bash
bash scripts/monitor.sh          # one-shot status snapshot
tail -f logs/train.log           # live stream (Ctrl-C just stops watching, not training)
tmux attach -t cogito            # watch live inside tmux (detach: Ctrl-b then d)
```

Healthy signs: **reward trending up** within ~100 steps, **completion length
growing**, all 3 GPUs busy. Full run ≈ 6 hours.

---

## STEP 5 — Math training (GSM8K + MATH)

After Countdown finishes (reward has plateaued):

```bash
python data/build_math.py                            # downloads GSM8K + MATH (+ MATH-500 for eval)
python eval/baseline_eval.py --task math             # base "before" for GSM8K + MATH-500
bash scripts/launch_math.sh                          # ~16–18 hours, background
bash scripts/monitor.sh
```

---

## STEP 6 — finalize: push to HF, download model, wipe scratch

```bash
HF_REPO=<YOUR-HF-USERNAME>/cogito-3b bash scripts/finalize.sh
```

This evaluates Cogito, builds the curves + before/after table + aha transcripts,
**pushes the model + recipe to your HF repo**, and packages `cogito_release.zip`.

Download `cogito_release.zip` to your computer via the **JupyterLab file browser**
(right-click the file → Download). It contains the model weights + all results.

Only once you've downloaded the zip AND confirmed your HF page looks right, free
the box's disk by re-running with the wipe flag:

```bash
HF_REPO=<YOUR-HF-USERNAME>/cogito-3b CONFIRM_WIPE=1 bash scripts/finalize.sh
```

---

## Monitoring cheat-sheet

| Want to…                | Command |
|---|---|
| Quick status            | `bash scripts/monitor.sh` |
| Live training log       | `tail -f logs/train.log` |
| Live curves (browser)   | `tensorboard --logdir runs --port 6006` |
| Watch inside tmux       | `tmux attach -t cogito` (detach: `Ctrl-b` then `d`) |
| GPU usage               | `nvidia-smi` |
| Stop everything         | `bash scripts/stop.sh` |
| Resume after a restart  | re-run the same `bash scripts/launch_*.sh` (auto-resumes from last checkpoint) |

---

## Troubleshooting

- **CUDA out of memory (training):** lower `per_device_train_batch_size` (and keep
  `per_device * gradient_accumulation_steps` divisible by `num_generations`) in
  `train/grpo_*.py`, or switch to ZeRO-2: edit the launch script's
  `--config_file` to `configs/accelerate_zero2.yaml`.
- **OOM on the vLLM server (GPU 0):** lower `VLLM_GPU_MEM` (e.g. `VLLM_GPU_MEM=0.7 bash scripts/launch_countdown.sh`).
- **`vllm` version error from TRL:** TRL supports vLLM 0.13–0.23. `pip install "vllm>=0.20,<0.23.1"`, then re-run.
- **flash-attn didn't install:** fine — the code auto-falls back to `sdpa` attention (a bit slower).
- **`num_generations` must divide batch:** ensure `per_device_train_batch_size * gradient_accumulation_steps` is a multiple of `num_generations` (defaults already satisfy this).
- **Port 8000 already in use:** `bash scripts/stop.sh` first, or set `VLLM_PORT=8001` for both the server and `vllm_server_port` in the train config.
- **Server never becomes ready:** `tail -n 50 logs/vllm.log` — usually a model download still in progress or an OOM.
- **Box got killed mid-run:** just re-run `bash scripts/launch_*.sh`; it resumes from the latest checkpoint in `checkpoints/`.

---

## Repo map

```
data/      build_countdown.py, build_math.py    # download/generate data + held-out splits
reward/    countdown.py, math_answer.py, format.py  # verifiable rewards (1.0 correct / 0.0)
prompts/   system_countdown.txt, system_math.txt    # R1-style think/answer templates
train/     grpo_countdown.py, grpo_math.py       # TRL GRPOTrainer + vLLM server config
rollout/   serve_vllm.sh                          # vLLM rollout server (GPU 0)
eval/      baseline_eval, eval_countdown, eval_math, track_length, transcripts
scripts/   launch_countdown.sh, launch_math.sh, monitor.sh, stop.sh, finalize.sh
configs/   accelerate_ddp.yaml, accelerate_zero2.yaml
```

Built for the Cogito project: GRPO reasoning training, reproducible and rerunnable.
