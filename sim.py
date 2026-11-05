#!/usr/bin/env python3
"""Simulate routing policies on a trace: cache hits, load balance, and TTFT.

    python3 sim.py --calib results/calib_qwen25/calib.json [--frac 0.3] [--speed 3 5 7]

Caches. Each replica has an LRU prompt cache on the GPU. A request's prompt splits
into three parts:
    gpu hit   blocks already on the chosen replica's GPU   (free)
    tier hit  blocks found in the host tier                (loaded, see below)
    miss      everything else                              (recomputed)
Server configs:
    gpu      no host tier
    private  each replica has its own host tier (tier-tokens / n each)
    shared   one host tier that every replica reads (tier-tokens total)

Timing. Each replica runs vLLM's loop, one engine step at a time:
  - every running request decodes one token;
  - the rest of the step's token budget goes to prefill chunks, oldest first;
  - a request starts prefill only if its whole context fits in the GPU KV cache;
  - a tier hit is copied in (DMA, overlapped with compute) before prefill starts.
  step time = step_base + step_per_seq * B + step_per_ctx * (KV tokens read by decodes)
            + prefill_per_tok * P + prefill_attn * sum(chunk * (ctx + chunk / 2))
for B decodes and P prefill tokens. calibrate.py measures all the coefficients from
isolated requests; `overhead` scales every step for what a loaded server adds on top
(scheduler and API-server CPU time, mixed-batch inefficiency). It is the one
parameter fitted to a sweep (validate.py does this, on a single arm).
"""
import argparse
import json
from collections import deque

import policies
import trace

BLOCK = trace.BLOCK
BUDGET = 2048  # vLLM's max_num_batched_tokens for these runs

# Seconds. Placeholders; pass --calib to use measured values.
COST = dict(prefill_base=0.009, prefill_per_tok=5.9e-5, prefill_attn=2.4e-9,
            load_base=0.009, load_per_tok=1.2e-5,
            step_base=0.012, step_per_seq=2e-4, step_per_ctx=5e-9, overhead=1.0)


def t_load(tokens, c=COST):
    return c["load_base"] + c["load_per_tok"] * tokens if tokens > 0 else 0.0


class Job:
    """One request inside a replica: waiting, then prefilling, then decoding."""

    def __init__(self, req, ready_at, miss, cached, arrival, est_miss):
        self.req, self.arrival, self.est_miss = req, arrival, est_miss
        self.ready_at = ready_at        # after any tier load
        self.prefill_left = miss        # tokens still to compute
        self.ctx = cached               # tokens of context already in GPU memory
        self.decode_left = req.output_len - 1
        self.kv = req.input_len + req.output_len  # KV reserved until the request ends


class Engine:
    """One replica running vLLM's loop: a FIFO of waiting jobs, then prefill, then decode."""

    def __init__(self, kv_tokens, cost):
        self.kv, self.c = kv_tokens, cost
        self.clock = 0.0
        self.waiting = deque()
        self.prefilling = []
        self.decoding = []
        self.used = 0                   # KV tokens reserved by prefilling + decoding jobs
        self.busy = 0.0

    def idle(self):
        return not (self.waiting or self.prefilling or self.decoding)

    def admit(self):
        """Start waiting jobs (oldest first) while their KV fits."""
        while self.waiting and self.waiting[0].ready_at <= self.clock:
            j = self.waiting[0]
            if self.used + j.kv > self.kv and (self.decoding or self.prefilling):
                return
            self.prefilling.append(self.waiting.popleft())
            self.used += j.kv

    def step(self, on_first_token):
        c = self.c
        self.admit()
        if not (self.prefilling or self.decoding):
            self.clock = max(self.clock, self.waiting[0].ready_at)  # nothing to run until a load lands
            return
        # every decoding job emits one token; leftover budget goes to prefill chunks
        B = len(self.decoding)
        t = c["step_base"] + c["step_per_seq"] * B + c["step_per_ctx"] * sum(j.ctx for j in self.decoding)
        budget = BUDGET - B
        done_prefill = []
        for j in self.prefilling:
            if budget <= 0:
                break
            chunk = min(budget, j.prefill_left)
            t += c["prefill_per_tok"] * chunk + c["prefill_attn"] * chunk * (j.ctx + chunk / 2)
            j.prefill_left -= chunk
            j.ctx += chunk
            budget -= chunk
            if j.prefill_left == 0:
                done_prefill.append(j)
        t *= c.get("overhead", 1.0)
        self.clock += t
        self.busy += t

        for j in self.decoding:
            j.ctx += 1
            j.decode_left -= 1
        for j in [j for j in self.decoding if j.decode_left <= 0]:
            self.decoding.remove(j)
            self.used -= j.kv
        for j in done_prefill:
            self.prefilling.remove(j)
            on_first_token(self, j)
            if j.decode_left > 0:
                self.decoding.append(j)
            else:
                self.used -= j.kv


def simulate(reqs, policy, server="gpu", n=4, speed=1.0, gpu_tokens=380_000,
             tier_tokens=1_700_000, cap=16384, cost=COST):
    gpu = [policies.Replica(gpu_tokens // BLOCK) for _ in range(n)]
    if server == "shared":
        tier = [policies.Replica(tier_tokens // BLOCK)] * n  # one tier, n references
    elif server == "private":
        tier = [policies.Replica(tier_tokens // n // BLOCK) for _ in range(n)]
    else:
        tier = None
    # The router's estimate of each replica, as in router.py: what it sent there,
    # sized to the GPU cache plus any tier only that replica can read.
    view_tokens = gpu_tokens + (tier_tokens // n if server == "private" else 0)
    view = [policies.Replica(view_tokens // BLOCK) for _ in range(n)]
    engines = [Engine(gpu_tokens, cost) for _ in range(n)]
    tok = dict(gpu=0, tier=0, miss=0)
    ttft = []

    def first_token(eng, job):
        view[engines.index(eng)].pending -= job.est_miss
        ttft.append(eng.clock - job.arrival)

    def advance(until):
        for e in engines:
            while not e.idle() and e.clock < until:
                e.step(first_token)
            if e.idle():
                e.clock = max(e.clock, until)

    for r in sorted(reqs, key=lambda r: r.t):
        now = r.t / speed
        advance(now)
        keys = r.hash_ids  # stand-in for policies.block_keys: same chaining, cheaper
        j = policies.choose(policy, view, keys, cap)
        est_miss = r.input_len - min(view[j].match(keys) * BLOCK, r.input_len)
        g = min(gpu[j].match(keys) * BLOCK, r.input_len)
        t = max(0, min(tier[j].match(keys) * BLOCK, r.input_len) - g) if tier else 0
        miss = max(1, r.input_len - g - t)  # vLLM always recomputes at least one token
        view[j].pending += est_miss
        engines[j].waiting.append(Job(r, now + t_load(t, cost), miss, g + t, now, est_miss))
        for c in [view[j], gpu[j]] + ([tier[j]] if tier else []):
            c.insert(keys)
        tok["gpu"] += g
        tok["tier"] += t
        tok["miss"] += r.input_len - g - t
    advance(float("inf"))
    ttft.sort()
    total = sum(tok.values())
    pct = lambda q: ttft[min(len(ttft) - 1, int(q * len(ttft)))]
    busy = [e.busy for e in engines]
    return dict(policy=policy, server=server,
                gpu_hit=tok["gpu"] / total, tier_hit=tok["tier"] / total,
                p50=pct(0.5), p90=pct(0.9), p99=pct(0.99),
                imbalance=max(busy) / (sum(busy) / n))


ARMS = [("least", "gpu"), ("prefix", "gpu"), ("prefix_cap", "gpu"),
        ("least", "private"), ("prefix_cap", "private"),
        ("least", "shared"), ("prefix_cap", "shared")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="conversation")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--speed", type=float, nargs="+", default=[3.0, 5.0, 7.0])
    ap.add_argument("--gpu-tokens", type=int, default=380_000)
    ap.add_argument("--tier-tokens", type=int, default=1_700_000)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--calib", nargs="*", default=[], help="calibrate.py JSONs (later ones win)")
    a = ap.parse_args()
    cost = dict(COST)
    for path in a.calib:
        cost.update(json.load(open(path))["cost"])
    reqs = trace.workload(a.trace, frac=a.frac)
    print(f"{len(reqs)} requests, frac={a.frac}, gpu {a.gpu_tokens} tok/replica, tier {a.tier_tokens} tok")
    for s in a.speed:
        print(f"\nspeed {s}x")
        print(f"{'policy':12s}{'server':9s}{'gpu hit':>9s}{'tier hit':>9s}{'p50 s':>8s}{'p90 s':>8s}"
              f"{'p99 s':>8s}{'imbal':>7s}")
        for p, srv in ARMS:
            m = simulate(reqs, p, srv, speed=s, gpu_tokens=a.gpu_tokens,
                         tier_tokens=a.tier_tokens, cap=a.cap, cost=cost)
            print(f"{p:12s}{srv:9s}{m['gpu_hit']:9.1%}{m['tier_hit']:9.1%}{m['p50']:8.2f}"
                  f"{m['p90']:8.2f}{m['p99']:8.2f}{m['imbalance']:7.2f}")


if __name__ == "__main__":
    main()
