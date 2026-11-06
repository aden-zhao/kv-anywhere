#!/usr/bin/env python3
"""README figures from sweep results.

    python3 figures.py results/conv_qwen25 --out docs/figures [--load 5]
    python3 figures.py results/agents_qwen25 --agents
    python3 figures.py results/agents_gc_least results/agents_gc_prefix --gc

ttft.png    median and p99 TTFT vs load for the four main arms
agents.png  (--agents) mean session time and throughput vs concurrent agents
gc.png      (--gc) session time vs tier size for each tier GC policy: the model's curves
            (capacity.py) and the measured runs
tokens.png  where each arm's prompt tokens came from at one load:
            GPU prefix cache, the KV tier, or recomputed
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import analyze

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Fixed categorical order (blue, orange, aqua, yellow, magenta, green); each arm keeps its color.
ARMS = [("gpu", "least", "least-loaded, GPU only", "#2a78d6"),
        ("gpu", "prefix", "prefix affinity, GPU only", "#eb6834"),
        ("gpu", "prefix_cap", "capped affinity, GPU only", "#1baf7a"),
        ("private", "prefix_cap", "capped affinity + private tiers", "#eda100"),
        ("private", "least", "least-loaded + private tiers", "#e87ba4"),
        ("shared", "least", "least-loaded + shared tier", "#008300"),
        ("shared", "prefix_cap", "capped affinity + shared tier", "#4a3aa7")]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# The TTFT chart shows four arms; pure prefix affinity is off the scale (see the table).
MAIN = [("gpu", "least"), ("gpu", "prefix_cap"), ("private", "least"), ("shared", "least")]


def ttft_figure(rows, out, arms):
    arms = [a for a in arms if (a[0], a[1]) in MAIN]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, q, title in [(axes[0], "p50", "Median TTFT"), (axes[1], "p99", "p99 TTFT")]:
        style(ax)
        for mode, pol, label, color in arms:
            pts = sorted((r["load"], r[q]) for r in rows if r["mode"] == mode and r["policy"] == pol)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("load (trace speedup)", color=INK2, fontsize=9)
        ax.set_ylabel("seconds", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def token_figure(rows, out, load, arms):
    sel = []
    for mode, pol, label, _ in arms:
        r = [r for r in rows if r["mode"] == mode and r["policy"] == pol and r["load"] == load]
        if r:
            sel.append((label, r[0]))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(sel) + 1.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    parts = [("gpu_hit", "GPU cache", "#2a78d6"), ("tier_hit", "KV tier", "#1baf7a")]
    for i, (label, r) in enumerate(sel):
        left = 0
        for key, name, color in parts:
            ax.barh(i, r[key] * 100, left=left, color=color, height=0.6,
                    edgecolor=SURFACE, linewidth=2, label=name if i == 0 else None)
            left += r[key] * 100
        ax.text(left + 0.8, i, f"{left:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(sel)), [s[0] for s in sel], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("% of prompt tokens not recomputed", color=INK2, fontsize=9)
    ax.set_title(f"Prompt tokens served from cache (load {load:g}x)", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    ax.set_xlim(0, max(r[1]["gpu_hit"] + r[1]["tier_hit"] for r in sel) * 100 * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


AGENT_ARMS = [("gpu", "least", "least-loaded, GPU only", "#2a78d6"),
              ("gpu", "prefix", "prefix affinity, GPU only", "#eb6834"),
              ("shared", "least", "least-loaded + /dev/shm tier", "#008300"),
              ("shared", "prefix", "prefix affinity + /dev/shm tier", "#4a3aa7"),
              ("lustre-big", "least", "least-loaded + big Lustre tier", "#eda100")]


def agents_figure(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, key, title, unit in [(axes[0], "session_mean", "Mean time to finish an agent session", "seconds"),
                                 (axes[1], "steps_per_s", "Throughput", "agent steps per second")]:
        style(ax)
        for mode, pol, label, color in AGENT_ARMS:
            pts = sorted((r["load"], r[key]) for r in rows if r["mode"] == mode and r["policy"] == pol)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks([16, 32, 48])
        ax.set_xlabel("concurrent agents (4 GPUs)", color=INK2, fontsize=9)
        ax.set_ylabel(unit, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


GC_POLICIES = [("oldest", "oldest written", "#2a78d6"),
               ("lru", "least recently used", "#eb6834"),
               ("protect", "protect live sessions", "#1baf7a")]


def gc_figure(dirs, capacity, out):
    """Model curves of session time vs tier size (capacity.py), with the measured runs."""
    model = json.load(open(capacity))
    runs = analyze.load(dirs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE, sharey=True)
    for ax, policy, title in [(axes[0], "least", "Least-loaded routing"),
                              (axes[1], "prefix", "Prefix-affinity routing")]:
        style(ax)
        for gc, label, color in GC_POLICIES:
            pts = [(r["tier_gb"], r["session_mean"]) for r in model["rows"]
                   if r["policy"] == policy and r["gc"] == gc]
            ax.plot(*zip(*pts), color=color, linewidth=2, label=label)
            meas = [(r["tier_gb"], r["session_mean"]) for r in runs
                    if r["policy"] == policy and r["gc"] == gc]
            ax.plot(*zip(*meas), "o", color=color, markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2)
        floor = min(r["session_mean"] for r in model["rows"] if r["policy"] == policy)
        ax.axhline(floor, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(250, floor + 25, "nothing evicted", ha="right", fontsize=8, color=INK2)
        ax.set_xlabel("shared tier size (GB)", color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(f"mean session time (s), {model['agents']} agents", color=INK2, fontsize=9)
    axes[0].plot([], [], "o", color=INK2, markersize=8, label="measured")
    axes[0].plot([], [], color=INK2, linewidth=2, label="model")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--load", type=float, default=None, help="trace speedup for tokens.png")
    ap.add_argument("--agents", action="store_true", help="agent sweep: draw agents.png only")
    ap.add_argument("--gc", action="store_true", help="GC sweeps: draw gc.png only")
    ap.add_argument("--capacity", default="results/gc_capacity.json", help="capacity.py output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.gc:
        gc_figure(a.dirs, a.capacity, f"{a.out}/gc.png")
        print("wrote", a.out)
        return
    rows = analyze.load(a.dirs)
    if a.agents:
        agents_figure(rows, f"{a.out}/agents.png")
        print("wrote", a.out)
        return
    load = a.load or max(r["load"] for r in rows)
    ttft_figure(rows, f"{a.out}/ttft.png", ARMS)
    token_figure(rows, f"{a.out}/tokens.png", load, ARMS)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
