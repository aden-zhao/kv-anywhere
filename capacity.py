#!/usr/bin/env python3
"""What an eviction policy is worth in tier capacity, according to the model.

    python3 capacity.py --calib results/calib_agents/calib_shared.json --out results/gc_capacity.json

Simulates 48 agents on a shared tier of each size, for every routing and GC policy,
with the agent sweep's fitted overhead (1.1). figures.py --gc draws the curves with
the measured runs on top.
"""
import argparse
import json

import validate

GPU_TOKENS = 381664  # per replica, from the agent sweeps' vLLM logs
KV_BYTES = 57344     # Qwen2.5-7B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--agents", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overhead", type=float, default=1.1)
    ap.add_argument("--gbs", type=float, nargs="+", default=[10 * i for i in range(2, 26)])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cost = dict(validate.load_costs(a.calib), overhead=a.overhead)
    meta = dict(trace="agents", max_steps=30, gap_cap=30.0, seed=a.seed)
    rows = []
    for policy in ["least", "prefix"]:
        for gc in ["oldest", "lru", "protect"]:
            for gb in a.gbs:
                run = dict(mode="shared", policy=policy, load=a.agents, gc=gc)
                m = validate.simulate_run(run, meta, cost, {}, GPU_TOKENS, int(gb * 1e9 / KV_BYTES))
                rows.append(dict(policy=policy, gc=gc, tier_gb=gb, session_mean=m["session_mean"],
                                 hits=m["gpu_hit"] + m["tier_hit"]))
            print(policy, gc, " ".join(f"{r['session_mean']:.0f}" for r in rows[-len(a.gbs):]))
    json.dump(dict(agents=a.agents, seed=a.seed, overhead=a.overhead, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
