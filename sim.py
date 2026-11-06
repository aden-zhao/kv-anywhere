#!/usr/bin/env python3
"""Simulate routing policies on a trace: cache hits, load balance, and TTFT.

    python3 sim.py --calib results/calib_qwen25/calib.json [--frac 0.3] [--speed 3 5 7]

Caches. Each replica has a prompt cache on the GPU that evicts like vLLM's (LRU,
tail of a prompt first). A request's prompt splits into three parts:
    gpu hit   blocks already on the chosen replica's GPU   (free)
    tier hit  blocks after those, found in the host tier   (copied in)
    miss      everything else                              (recomputed, then stored)
Server configs: gpu (no tier), private (one tier per replica, tier_tokens / n each),
shared (one tier every replica reads). The tier runs tier_gc.py's own deletion code
every gc_interval seconds, with policy gc ("oldest", "lru" or "protect").

Timing. Each replica runs vLLM's loop, one engine step at a time:
  - every running request decodes one token;
  - the rest of the step's token budget goes to prefill chunks, oldest first;
  - a request starts prefill only once its whole context fits in the GPU KV cache;
  - a tier hit is copied in first (DMA, overlapped with compute), one copy at a
    time per replica.
  step time = step_base + step_per_seq * B + step_per_ctx * (context tokens of the B decodes)
            + prefill_per_tok * P + prefill_attn * sum(chunk * (ctx + chunk / 2))
for B decoding requests and P prefill tokens. calibrate.py measures the coefficients
on isolated requests. `overhead` scales every step for what a loaded server adds on
top (scheduler and API-server CPU, mixed batches); validate.py fits it on one arm.
"""
import argparse
import heapq
import itertools
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


class Job:
    """One request inside a replica: waiting, then prefilling, then decoding."""

    def __init__(self, req, arrival, ready_at, miss, cached, est_miss):
        self.req, self.arrival, self.est_miss = req, arrival, est_miss
        self.ready_at = ready_at        # after any tier copy
        self.prefill_left = miss        # tokens still to compute
        self.ctx = cached               # tokens of context already in GPU memory
        self.decode_left = req.output_len - 1
        self.kv = req.input_len + req.output_len  # KV reserved until the request ends


class Engine:
    """One replica running vLLM's loop: a FIFO of waiting jobs, then prefill, then decode."""

    def __init__(self, i, kv_tokens, cost):
        self.i, self.kv, self.c = i, kv_tokens, cost
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

    def step(self, on_first_token, on_done):
        c = self.c
        self.admit()
        if not (self.prefilling or self.decoding):
            self.clock = max(self.clock, self.waiting[0].ready_at)  # nothing to run until a copy lands
            return
        # every decoding job emits one token; the leftover budget goes to prefill chunks
        B = len(self.decoding)
        t = c["step_base"] + c["step_per_seq"] * B + c["step_per_ctx"] * sum(j.ctx for j in self.decoding)
        budget = BUDGET - B
        prefilled = []
        for j in self.prefilling:
            if budget <= 0:
                break
            chunk = min(budget, j.prefill_left)
            t += c["prefill_per_tok"] * chunk + c["prefill_attn"] * chunk * (j.ctx + chunk / 2)
            j.prefill_left -= chunk
            j.ctx += chunk
            budget -= chunk
            if j.prefill_left == 0:
                prefilled.append(j)
        t *= c["overhead"]
        self.clock += t
        self.busy += t

        for j in self.decoding:
            j.ctx += 1
            j.decode_left -= 1
        for j in [j for j in self.decoding if j.decode_left <= 0]:
            self.decoding.remove(j)
            self.used -= j.kv
            on_done(self, j)
        for j in prefilled:
            self.prefilling.remove(j)
            on_first_token(self, j)
            if j.decode_left > 0:
                self.decoding.append(j)
            else:
                self.used -= j.kv
                on_done(self, j)


class Cluster:
    """n replicas, their caches, and the router's view of them (as in router.py)."""

    def __init__(self, policy, server="gpu", n=4, gpu_tokens=380_000, tier_tokens=1_700_000,
                 cap=16384, cost=COST, gc="oldest", gc_interval=1.0):
        self.policy, self.cap, self.cost = policy, cap, cost
        self.engines = [Engine(i, gpu_tokens, cost) for i in range(n)]
        self.gpu = [policies.ChainCache(gpu_tokens // BLOCK) for _ in range(n)]
        if server == "shared":
            self.tier = [policies.Tier(tier_tokens // BLOCK, gc, gc_interval)] * n  # one tier, n references
        elif server == "private":
            self.tier = [policies.Tier(tier_tokens // n // BLOCK, gc, gc_interval) for _ in range(n)]
        else:
            self.tier = None
        # The router's guess of each replica: what it sent there, sized to the GPU
        # cache plus any tier only that replica can read.
        view_tokens = gpu_tokens + (tier_tokens // n if server == "private" else 0)
        self.view = [policies.Replica(view_tokens // BLOCK) for _ in range(n)]
        self.link_free = [0.0] * n  # when each replica's host link finishes its current copy
        self.tokens = dict(gpu=0, tier=0, miss=0)
        self.ttft = []

    def route(self, now, req):
        """Pick a replica for req and queue it there."""
        keys = req.hash_ids  # stand-in for policies.block_keys: same chaining, cheaper
        i = policies.choose(self.policy, self.view, keys, self.cap)
        n = req.input_len
        est_miss = n - min(self.view[i].match(keys) * BLOCK, n)
        g = t = self.gpu[i].match(keys)  # blocks on the GPU, and where the tier's run ends
        if self.tier:
            self.tier[i].collect(now)
            t = self.tier[i].match(keys, g)
        gpu_hit = min(g * BLOCK, n)
        tier_hit = min(t * BLOCK, n) - gpu_hit
        miss = max(1, n - gpu_hit - tier_hit)  # vLLM always recomputes at least one token
        self.view[i].pending += est_miss
        self.view[i].insert(keys)
        self.gpu[i].insert(keys)
        if self.tier:
            self.tier[i].access(keys, g, t, now)
        self.tokens["gpu"] += gpu_hit
        self.tokens["tier"] += tier_hit
        self.tokens["miss"] += n - gpu_hit - tier_hit
        eng = self.engines[i]
        if eng.idle():
            eng.clock = max(eng.clock, now)
        eng.waiting.append(Job(req, now, self.copy_done(i, now, tier_hit), miss, gpu_hit + tier_hit, est_miss))

    def copy_done(self, i, now, tokens):
        """When copying `tokens` from the tier into replica i finishes (copies queue)."""
        if tokens == 0:
            return now
        c = self.cost
        self.link_free[i] = max(now, self.link_free[i]) + c["load_base"] + c["load_per_tok"] * tokens
        return self.link_free[i]

    def first_token(self, eng, job):
        self.view[eng.i].pending -= job.est_miss
        self.ttft.append(eng.clock - job.arrival)

    def summary(self):
        ttft = sorted(self.ttft)
        pct = lambda q: ttft[min(len(ttft) - 1, int(q * len(ttft)))]
        total = sum(self.tokens.values())
        busy = [e.busy for e in self.engines]
        return dict(gpu_hit=self.tokens["gpu"] / total, tier_hit=self.tokens["tier"] / total,
                    p50=pct(0.5), p90=pct(0.9), p99=pct(0.99),
                    imbalance=max(busy) / (sum(busy) / len(busy)))


def run(arrivals, follow, policy, server="gpu", **kw):
    """Event loop: always advance whichever comes first, the next arrival or the
    replica furthest behind. follow(req, t_done) gives the next (time, request), if any."""
    cluster = Cluster(policy, server, **kw)
    order = itertools.count()  # breaks ties between events at the same time
    queue = [(t, next(order), r) for t, r in arrivals]
    heapq.heapify(queue)

    def done(eng, job):
        nxt = follow(job.req, eng.clock)
        if nxt:
            heapq.heappush(queue, (nxt[0], next(order), nxt[1]))

    while queue or any(not e.idle() for e in cluster.engines):
        busy = [e for e in cluster.engines if not e.idle()]
        e = min(busy, key=lambda e: e.clock) if busy else None
        if e and (not queue or e.clock < queue[0][0]):
            e.step(cluster.first_token, done)
        else:
            now, _, req = heapq.heappop(queue)
            cluster.route(now, req)
    return cluster.summary()


def simulate(reqs, policy, server="gpu", speed=1.0, **kw):
    """Open loop: every request arrives at its trace time / speed."""
    return run([(r.t / speed, r) for r in reqs], lambda req, t: None, policy, server, **kw)


def simulate_agents(agents, policy, server="gpu", **kw):
    """Closed loop: each agent sends its next step `gap` seconds after the last one finishes."""
    steps_done = [0] * len(agents)
    finish = [0.0] * len(agents)

    def follow(step, t):
        s = step.session
        steps_done[s] += 1
        finish[s] = t
        if steps_done[s] < len(agents[s].steps):
            nxt = agents[s].steps[steps_done[s]]
            return t + nxt.gap, nxt
        return None

    out = run([(a.start, a.steps[0]) for a in agents], follow, policy, server, **kw)
    sessions = sorted(finish[s] - a.start for s, a in enumerate(agents))
    out.update(session_mean=sum(sessions) / len(sessions),
               session_p90=sessions[int(0.9 * (len(sessions) - 1))], session_max=sessions[-1],
               steps_per_s=sum(len(a.steps) for a in agents) / max(finish))
    return out


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
            m = simulate(reqs, p, srv, speed=s, gpu_tokens=a.gpu_tokens, tier_tokens=a.tier_tokens, cost=cost)
            print(f"{p:12s}{srv:9s}{m['gpu_hit']:9.1%}{m['tier_hit']:9.1%}{m['p50']:8.2f}"
                  f"{m['p90']:8.2f}{m['p99']:8.2f}{m['imbalance']:7.2f}")


if __name__ == "__main__":
    main()
