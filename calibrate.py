#!/usr/bin/env python3
"""Measure the simulator's costs on one vLLM server that has a file-system KV tier.

    python3 calibrate.py --backend http://localhost:8100 --model <m> --out results/calib.json

Prefill and tier loads. For each prompt length n, four 1-token requests (so the
time to first token is the prefill or load time):
  compute  a fresh prompt, nothing cached               -> prefill n tokens, no context
  extend   the same prompt plus n more tokens           -> prefill n tokens after n cached
  gpu      the same prompt again                        -> GPU prefix hit (the floor)
  tier     clear GPU + CPU caches, same prompt again    -> load n tokens from the tier
Decode. B concurrent requests with C-token prompts each generate 128 tokens. Once
their prefills are done, the gap between streamed tokens is one decode step.

Fits (least squares):
  t_prefill(new, ctx) = prefill_base + prefill_per_tok * new + prefill_attn * new * (ctx + new / 2)
  t_load(n)           = load_base + load_per_tok * n
  t_step(B, C)        = step_base + step_per_seq * B + step_per_ctx * B * C
"""
import argparse
import concurrent.futures
import json
import random
import time

import numpy as np
import requests


def prompt(rng, n):
    return [rng.randrange(1000, 100000) for _ in range(n)]


def first_token_time(url, model, toks):
    t = time.perf_counter()
    requests.post(url + "/v1/completions", json={
        "model": model, "prompt": toks, "max_tokens": 1, "temperature": 0}).raise_for_status()
    return time.perf_counter() - t


def clear_caches(url):
    """GPU prefix cache and the CPU staging tier; the file-system tier keeps its files."""
    requests.post(url + "/reset_prefix_cache", params={"reset_external": "true"}).raise_for_status()


def token_gaps(url, model, toks, n_out=128):
    r = requests.post(url + "/v1/completions", stream=True, json={
        "model": model, "prompt": toks, "max_tokens": n_out, "ignore_eos": True,
        "temperature": 0, "stream": True})
    stamps = [time.perf_counter() for line in r.iter_lines() if line.startswith(b"data: {")]
    return [b - a for a, b in zip(stamps, stamps[1:])]


def step_time(url, model, B, C, rng):
    prompts = [prompt(rng, C) for _ in range(B)]
    with concurrent.futures.ThreadPoolExecutor(B) as ex:
        gaps = list(ex.map(lambda p: token_gaps(url, model, p), prompts))
    late = sorted(g for gs in gaps for g in gs[len(gs) // 2:])  # second half: prefills are done
    return late[len(late) // 2]


def kv_capacity(url):
    """GPU KV capacity in tokens, from vLLM's cache_config_info metric labels."""
    line = next(l for l in requests.get(url + "/metrics").text.splitlines()
                if l.startswith("vllm:cache_config_info"))
    labels = dict(kv.split("=", 1) for kv in line[line.index("{") + 1:line.index("}")].split(","))
    return int(labels["num_gpu_blocks"].strip('"')) * int(labels["block_size"].strip('"'))


def measure_prefill(a, rng):
    rows = []
    for n in a.lens:  # `extend` sends 2n tokens, so 2n must fit in the 32k context
        for _ in range(a.reps):
            p = prompt(rng, n)
            clear_caches(a.backend)
            row = dict(n=n, compute=first_token_time(a.backend, a.model, p))
            row["extend"] = first_token_time(a.backend, a.model, p + prompt(rng, n))
            row["gpu"] = first_token_time(a.backend, a.model, p)
            time.sleep(1.0)  # let the tier writes land
            clear_caches(a.backend)
            row["tier"] = first_token_time(a.backend, a.model, p)
            rows.append(row)
            print(row, flush=True)
    return rows


def measure_decode(a, rng):
    steps, cap = [], kv_capacity(a.backend)
    for C, Bs in [(2048, [1, 8, 32, 96]), (8192, [1, 8, 32])]:
        for B in Bs:
            if B * (C + 128) > 0.8 * cap:  # vLLM would preempt; not a clean step
                continue
            clear_caches(a.backend)
            steps.append(dict(B=B, C=C, step=step_time(a.backend, a.model, B, C, rng)))
            print(steps[-1], flush=True)
    return steps


def fit(X, y):
    return np.linalg.lstsq(np.array(X, float), np.array(y, float), rcond=None)[0].tolist()


def fit_prefill(rows):
    X, y = [], []
    for r in rows:
        n = r["n"]
        X += [[1, n, n * (0 + n / 2)], [1, n, n * (n + n / 2)]]  # ctx = 0, then ctx = n
        y += [r["compute"], r["extend"]]
    base, per_tok, attn = fit(X, y)
    load_base, load_per_tok = fit([[1, r["n"]] for r in rows], [r["tier"] - r["gpu"] for r in rows])
    return dict(prefill_base=base, prefill_per_tok=per_tok, prefill_attn=attn,
                load_base=load_base, load_per_tok=load_per_tok)


def fit_decode(steps):
    base, per_seq, per_ctx = fit([[1, s["B"], s["B"] * s["C"]] for s in steps], [s["step"] for s in steps])
    return dict(step_base=base, step_per_seq=per_seq, step_per_ctx=per_ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://localhost:8100")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 15360])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(1)
    first_token_time(a.backend, a.model, prompt(rng, 64))  # warm up
    out = dict(cost={})
    if not a.decode_only:
        out["rows"] = measure_prefill(a, rng)
        out["cost"].update(fit_prefill(out["rows"]))
    out["steps"] = measure_decode(a, rng)
    out["cost"].update(fit_decode(out["steps"]))
    print("fit:", out["cost"])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
