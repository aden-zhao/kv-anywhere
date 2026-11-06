#!/usr/bin/env python3
"""Summarize sweep results: one row per run.

    python3 analyze.py results/conv [more dirs] [--csv summary.csv]

Columns: TTFT percentiles, hit shares from vLLM's counters (GPU prefix cache vs
the external tier), and load imbalance (busiest replica's prompt tokens / mean).
"""
import argparse
import csv
import glob
import json
import os


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    meta = json.load(open(path.replace(".csv", ".json")))
    ttft = [float(r["ttft"]) for r in rows if r["ttft"] not in ("", "None")]
    c = meta["counters"]
    queries = sum(x["vllm:prefix_cache_queries_total"] for x in c)
    local = sum(x["vllm:prefix_cache_hits_total"] for x in c)
    ext = sum(x["vllm:external_prefix_cache_hits_total"] for x in c)
    prompt = [x["vllm:prompt_tokens_total"] for x in c]
    agentic = meta["trace"] == "agents"
    return dict(
        run=os.path.basename(path)[:-4], mode=meta["mode"], policy=meta["policy"],
        load=meta["agents"] if agentic else meta["speed"],  # agents, or trace speedup
        tier_gb=meta.get("tier_gb"), gc=meta.get("gc", "oldest"),
        n=len(rows), failed=len(rows) - len(ttft),
        p50=pct(ttft, 0.5), p90=pct(ttft, 0.9), p99=pct(ttft, 0.99),
        mean=sum(ttft) / max(1, len(ttft)),
        gpu_hit=local / queries if queries else 0, tier_hit=ext / queries if queries else 0,
        imbalance=max(prompt) / (sum(prompt) / len(prompt)) if sum(prompt) else 0,
        sent="/".join(str(s) for s in meta["router"]["sent"]),
        **(agent_times(rows) if agentic else {}))


def agent_times(rows):
    """Closed-loop runs: how long each agent took from its first step to its last."""
    span = {}
    for r in rows:
        t0, t1 = float(r["sent"]), float(r["sent"]) + float(r["e2e"])
        a, b = span.get(r["session"], (t0, t1))
        span[r["session"]] = (min(a, t0), max(b, t1))
    times = [b - a for a, b in span.values()]
    end = max(b for _, b in span.values())
    return dict(session_mean=sum(times) / len(times), steps_per_s=len(rows) / end)


def load(dirs):
    """Summaries of every finished run (a CSV with its JSON) in the given directories."""
    return [summarize(p) for d in dirs for p in sorted(glob.glob(os.path.join(d, "*.csv")))
            if os.path.exists(p.replace(".csv", ".json"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    out = load(a.dirs)
    out.sort(key=lambda r: (r["load"], r["mode"], r["policy"]))
    print(f"{'load':>5} {'mode':8s}{'policy':11s}{'n':>5}{'fail':>5}{'p50':>7}{'p90':>7}{'p99':>7}"
          f"{'gpu hit':>8}{'tier hit':>9}{'imbal':>6}  sent")
    for r in out:
        print(f"{r['load']:5g} {r['mode']:8s}{r['policy']:11s}{r['n']:5d}{r['failed']:5d}"
              f"{r['p50']:7.2f}{r['p90']:7.2f}{r['p99']:7.2f}{r['gpu_hit']:8.1%}{r['tier_hit']:9.1%}"
              f"{r['imbalance']:6.2f}  {r['sent']}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
