#!/usr/bin/env python3
"""README figures from sweep results.

    python3 figures.py results/conv_qwen25 --out docs/figures [--speed 7]

ttft.png    median and p99 TTFT vs load for the four main arms
tokens.png  where each arm's prompt tokens came from at one load:
            GPU prefix cache, the KV tier, or recomputed
"""
import argparse
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
            pts = sorted((r["speed"], r[q]) for r in rows if r["mode"] == mode and r["policy"] == pol)
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


def token_figure(rows, out, speed, arms):
    sel = []
    for mode, pol, label, _ in arms:
        r = [r for r in rows if r["mode"] == mode and r["policy"] == pol and r["speed"] == speed]
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
    ax.set_title(f"Prompt tokens served from cache (load {speed:g}x)", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    ax.set_xlim(0, max(r[1]["gpu_hit"] + r[1]["tier_hit"] for r in sel) * 100 * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--speed", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = analyze.load(a.dirs)
    speed = a.speed or max(r["speed"] for r in rows)
    ttft_figure(rows, f"{a.out}/ttft.png", ARMS)
    token_figure(rows, f"{a.out}/tokens.png", speed, ARMS)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
