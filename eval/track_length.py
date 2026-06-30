"""Build the signature curves from TensorBoard logs written during training:
  results/reward_curve.png   (mean reward vs step)
  results/length_curve.png   (mean completion length vs step  ->  'it learns to think longer')
Robust to TRL tag-name differences across versions (matches by substring)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _event_dirs(runs_dir):
    dirs = []
    for cur, _, files in os.walk(runs_dir):
        if any(f.startswith("events.out.tfevents") for f in files):
            dirs.append(cur)
    return dirs


def _scalars(ev_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(ev_dir)
    ea.Reload()
    return ea, ea.Tags().get("scalars", [])


def _pick(tags, must, must_not=()):
    for t in tags:
        tl = t.lower()
        if all(m in tl for m in must) and not any(b in tl for b in must_not):
            return t
    return None


def _series(ea, tag):
    pts = ea.Scalars(tag)
    return [p.step for p in pts], [p.value for p in pts]


def plot_metric(must, must_not, title, ylabel, outfile):
    runs_dir = os.path.join(ROOT, "runs")
    if not os.path.isdir(runs_dir):
        print("[track] no runs/ dir yet"); return
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for ev in _event_dirs(runs_dir):
        ea, tags = _scalars(ev)
        tag = _pick(tags, must, must_not)
        if tag is None:
            continue
        steps, vals = _series(ea, tag)
        if steps:
            label = os.path.relpath(ev, runs_dir).split(os.sep)[0]
            plt.plot(steps, vals, label="%s:%s" % (label, tag))
            plotted = True
    if not plotted:
        print("[track] no matching scalar for %s" % "+".join(must)); plt.close(); return
    plt.xlabel("training step"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(ROOT, "results", outfile)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130); plt.close()
    print("[track] wrote %s" % out)


if __name__ == "__main__":
    plot_metric(["reward"], ["std", "min", "max"], "GRPO reward over training", "mean reward", "reward_curve.png")
    plot_metric(["length"], [], "Mean response length over training", "completion length (tokens)", "length_curve.png")
