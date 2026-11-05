#!/usr/bin/env python3
"""Check the simulator against a sweep: hit rates and TTFT, arm by arm.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--kv-bytes", type=int, required=True, help="KV bytes per token")
    ap.add_argument("--tier-gb", type=float, default=100)
    ap.add_argument("--fit-arm", default="gpu/least")
    ap.add_argument("--calib-lustre", help="calibration measured with the tier on Lustre")
    a = ap.parse_args()
    cost = dict(sim.COST)
    for p in a.calib:
        cost.update(json.load(open(p))["cost"])
    rows = [analyze.summarize(p) for p in sorted(glob.glob(os.path.join(a.results, "*.csv")))]
    meta = json.load(open(glob.glob(os.path.join(a.results, "*.json"))[0]))
    log = open(glob.glob(os.path.join(a.results, "logs", "*server0.log"))[0]).read()
    gpu_tokens = int(re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)[-1].replace(",", ""))
    tier_tokens = int(a.tier_gb * 1e9 / a.kv_bytes)
    lustre = json.load(open(a.calib_lustre))["cost"] if a.calib_lustre else {}

    def run(r, c):
        if r["mode"].startswith("lustre"):  # the shared tier, with Lustre's load cost
            c = dict(c, load_base=lustre["load_base"], load_per_tok=lustre["load_per_tok"])
        server = "shared" if r["mode"].startswith("lustre") else r["mode"]
        tier = int(r["tier_gb"] * 1e9 / a.kv_bytes) if r.get("tier_gb") else tier_tokens
        kw = dict(gpu_tokens=gpu_tokens, tier_tokens=tier, cost=c, tier_eviction="fifo")  # as gc.py
        if meta["trace"] == "agents":
            work = agents.workload(int(r["speed"]), max_steps=meta["max_steps"],
                                   gap_cap=meta["gap_cap"], seed=meta["seed"])
            return sim.simulate_agents(work, r["policy"], server, **kw)
        reqs = trace.workload(meta["trace"], frac=meta["frac"], dur_s=meta["dur"])
        return sim.simulate(reqs, r["policy"], server, speed=r["speed"], **kw)

    fit_mode, fit_policy = a.fit_arm.split("/")
    fit_rows = [r for r in rows if r["mode"] == fit_mode and r["policy"] == fit_policy]

    def loss(k):
        c = dict(cost, overhead=k)
        e = []
        for r in fit_rows:
            m = run(r, c)
            e += [math.log(m["p50"] / r["p50"]), math.log(m["p99"] / r["p99"])]
        return sum(x * x for x in e) / len(e)

    k = min((loss(k), k) for k in [1 + i / 40 for i in range(17)])[1]
    cost["overhead"] = k
    print(f"overhead fitted on {a.fit_arm}: {k:.3f}   (GPU KV {gpu_tokens} tok, tier {tier_tokens} tok)")
    agentic = meta["trace"] == "agents"
    print(f"{'arm':22s}{'load':>5}  {'cached sim/meas':>16}  {'p50 sim/meas':>14}  {'p99 sim/meas':>14}"
          + (f"  {'session sim/meas':>16}" if agentic else ""))
    out = []
    for r in sorted(rows, key=lambda r: (r["mode"], r["policy"], r["speed"])):
        m = run(r, cost)
        hit_s, hit_m = m["gpu_hit"] + m["tier_hit"], r["gpu_hit"] + r["tier_hit"]
        print(f"{r['mode'] + ' ' + r['policy']:22s}{r['speed']:5g}  {hit_s:7.1%} /{hit_m:7.1%}  "
              f"{m['p50']:6.2f} /{r['p50']:6.2f}  {m['p99']:6.2f} /{r['p99']:6.2f}"
              + (f"  {m['session_mean']:7.0f} /{r['session_mean']:6.0f}" if agentic else ""))
        out.append(dict(mode=r["mode"], policy=r["policy"], speed=r["speed"],
                        **({"sim_session": m["session_mean"], "meas_session": r["session_mean"]} if agentic else {}),
                        sim_hit=hit_s, meas_hit=hit_m, sim_p50=m["p50"], meas_p50=r["p50"],
                        sim_p99=m["p99"], meas_p99=r["p99"]))
    with open(os.path.join(a.results, "validate.json"), "w") as f:
        json.dump(dict(overhead=k, gpu_tokens=gpu_tokens, tier_tokens=tier_tokens, rows=out), f, indent=1)


if __name__ == "__main__":
    main()
