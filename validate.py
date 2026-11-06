#!/usr/bin/env python3
"""Check the simulator against a sweep, run by run: hit rates, TTFT, and (for agents)
session time.

    python3 validate.py results/conv_qwen25 --calib results/calib_qwen25/calib.json \
        results/decode_Qwen2.5-7B-Instruct/calib.json --kv-bytes 57344

Everything in the simulator comes from calibrate.py except `overhead`, which is
fitted here on one arm (least-loaded, GPU only) and then held fixed for the rest.
"""
import argparse
import glob
import json
import math
import os
import re

import agents
import analyze
import sim
import trace


def load_costs(paths):
    cost = dict(sim.COST)
    for p in paths:
        cost.update(json.load(open(p))["cost"])
    return cost


def gpu_kv_tokens(results):
    log = open(glob.glob(os.path.join(results, "logs", "*server0.log"))[0]).read()
    return int(re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)[-1].replace(",", ""))


def simulate_run(run, meta, cost, lustre_cost, gpu_tokens, tier_tokens):
    """Simulate one measured run with the same workload, routing, and tier."""
    server, gc_interval = run["mode"], 1.0
    if server.startswith("lustre"):  # the shared tier, with Lustre's copy cost
        cost = dict(cost, load_base=lustre_cost["load_base"], load_per_tok=lustre_cost["load_per_tok"])
        server, gc_interval = "shared", 30.0  # sweep.py: scanning Lustre takes a while
    kw = dict(gpu_tokens=gpu_tokens, tier_tokens=tier_tokens, cost=cost,
              gc=run.get("gc") or "oldest", gc_interval=gc_interval)  # what tier_gc.py did
    if meta["trace"] == "agents":
        work = agents.workload(int(run["load"]), max_steps=meta["max_steps"],
                               gap_cap=meta["gap_cap"], seed=meta["seed"])
        return sim.simulate_agents(work, run["policy"], server, **kw)
    reqs = trace.workload(meta["trace"], frac=meta["frac"], dur_s=meta["dur"])
    return sim.simulate(reqs, run["policy"], server, speed=run["load"], **kw)


def fit_overhead(runs, simulate):
    """The overhead (1.0 to 1.4) whose simulated p50 and p99 TTFT best match these runs."""
    def loss(k):
        err = []
        for r in runs:
            m = simulate(r, k)
            err += [math.log(m["p50"] / r["p50"]), math.log(m["p99"] / r["p99"])]
        return sum(e * e for e in err) / len(err)
    return min([1 + i / 40 for i in range(17)], key=loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--calib-lustre", help="calibration measured with the tier on Lustre")
    ap.add_argument("--kv-bytes", type=int, required=True, help="KV bytes per token")
    ap.add_argument("--tier-gb", type=float, default=100, help="for runs that didn't record it")
    ap.add_argument("--fit-arm", default="gpu/least")
    ap.add_argument("--overhead", type=float, help="use this instead of fitting (sweeps without the fit arm)")
    a = ap.parse_args()
    cost = load_costs(a.calib)
    lustre_cost = json.load(open(a.calib_lustre))["cost"] if a.calib_lustre else {}
    runs = sorted(analyze.load([a.results]),
                  key=lambda r: (r["mode"], r["policy"], r["tier_gb"] or 0, r["gc"], r["load"]))
    meta = json.load(open(glob.glob(os.path.join(a.results, "*.json"))[0]))
    gpu_tokens = gpu_kv_tokens(a.results)
    tier_tokens = lambda r: int((r["tier_gb"] or a.tier_gb) * 1e9 / a.kv_bytes)
    simulate = lambda r, k: simulate_run(r, meta, dict(cost, overhead=k), lustre_cost,
                                         gpu_tokens, tier_tokens(r))

    if a.overhead:
        k = a.overhead
        print(f"overhead {k:.3f} (given)   (GPU KV {gpu_tokens} tok/replica)")
    else:
        fit_mode, fit_policy = a.fit_arm.split("/")
        k = fit_overhead([r for r in runs if r["mode"] == fit_mode and r["policy"] == fit_policy], simulate)
        print(f"overhead fitted on {a.fit_arm}: {k:.3f}   (GPU KV {gpu_tokens} tok/replica)")

    agentic = meta["trace"] == "agents"
    print(f"{'arm':30s}{'load':>5}  {'cached sim/meas':>16}  {'p50 sim/meas':>14}  {'p99 sim/meas':>14}"
          + (f"  {'session sim/meas':>16}" if agentic else ""))
    out = []
    for r in runs:
        m = simulate(r, k)
        hit_s, hit_m = m["gpu_hit"] + m["tier_hit"], r["gpu_hit"] + r["tier_hit"]
        arm = f"{r['mode']} {r['policy']}" + (f" {r['gc']} {r['tier_gb']:g}GB" if r["tier_gb"] else "")
        print(f"{arm:30s}{r['load']:5g}  {hit_s:7.1%} /{hit_m:7.1%}  "
              f"{m['p50']:6.2f} /{r['p50']:6.2f}  {m['p99']:6.2f} /{r['p99']:6.2f}"
              + (f"  {m['session_mean']:7.0f} /{r['session_mean']:6.0f}" if agentic else ""))
        row = dict(mode=r["mode"], policy=r["policy"], load=r["load"], sim_hit=hit_s, meas_hit=hit_m,
                   **({"gc": r["gc"], "tier_gb": r["tier_gb"]} if r["tier_gb"] else {}),
                   sim_p50=m["p50"], meas_p50=r["p50"], sim_p99=m["p99"], meas_p99=r["p99"])
        if agentic:
            row.update(sim_session=m["session_mean"], meas_session=r["session_mean"])
        out.append(row)
    with open(os.path.join(a.results, "validate.json"), "w") as f:
        json.dump(dict(overhead=k, gpu_tokens=gpu_tokens, rows=out), f, indent=1)


if __name__ == "__main__":
    main()
