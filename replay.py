#!/usr/bin/env python3
"""Replay a workload against the router and record per-request latency.

    python3 replay.py --backends <urls> --model <m> --trace conversation --frac 0.3 --speed 4 --out run.csv
    python3 replay.py --backends <urls> --model <m> --trace agents --agents 32 --out run.csv

Mooncake traces (conversation, toolagent) are open loop: requests go out at their
trace arrival times / --speed. Agent sessions (FreeInference) are closed loop: each
agent sends its next step only after the previous one finishes, plus the recorded
gap. Prompts are token ids, with `ignore_eos` and the trace's output length.

Each CSV row is one request: send time, TTFT, end-to-end time, replica, prompt
length, cached tokens, session. A JSON next to the CSV holds each backend's
prefix-cache hit counters (local GPU vs external tier) over the run.
"""
import argparse
import asyncio
import csv
import json
import time

import aiohttp

import agents
import trace

COUNTERS = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
            "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]


class Prompts:
    """Request bodies built from pre-encoded 512-token blocks. An agent prompt can be
    128k tokens; encoding it from scratch would stall the event loop for tens of ms."""

    def __init__(self, model):
        self.model, self.blocks = model, {}

    def prepare(self, reqs):
        for r in reqs:
            for h in r.hash_ids:
                if h not in self.blocks:
                    self.blocks[h] = ",".join(map(str, trace.block_tokens(h)))

    def body(self, req):
        full, rest = divmod(req.input_len, trace.BLOCK)
        parts = [self.blocks[h] for h in req.hash_ids[:full]]
        if rest:  # the last block is partial
            parts.append(",".join(self.blocks[req.hash_ids[full]].split(",")[:rest]))
        return ('{"model": %s, "prompt": [%s], "max_tokens": %d, "ignore_eos": true, '
                '"temperature": 0, "stream": true, "stream_options": {"include_usage": true}}'
                % (json.dumps(self.model), ",".join(parts), req.output_len)).encode()


async def scrape(session, url):
    async with session.get(url + "/metrics") as r:
        text = await r.text()
    out = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        name = line.split("{")[0].split(" ")[0]
        if name in out:
            out[name] += float(line.rsplit(" ", 1)[1])
    return out


async def send(session, a, prompts, req, t0, rows):
    """One request, now. Appends its row; returns when the response has finished."""
    sent = time.perf_counter()
    ttft, cached, n_out, err, replica = None, None, 0, "", -1
    try:
        async with session.post(a.router + "/v1/completions", data=prompts.body(req),
                                headers={"Content-Type": "application/json"}) as r:
            replica = int(r.headers.get("X-Replica", -1))
            async for line in r.content:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                msg = json.loads(line[6:])
                if ttft is None and msg.get("choices"):
                    ttft = time.perf_counter() - sent
                if msg.get("usage"):
                    n_out = msg["usage"]["completion_tokens"]
                    cached = (msg["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    except Exception as e:  # keep going; the row records the failure
        err = repr(e)[:200]
    rows.append(dict(sent=round(sent - t0, 4), ttft=ttft, e2e=time.perf_counter() - sent,
                     replica=replica, input_len=req.input_len, output_len=n_out,
                     cached=cached, session=req.session, err=err))


async def open_loop(session, a, prompts, req, t0, rows):
    await asyncio.sleep(max(0.0, t0 + req.t / a.speed - time.perf_counter()))
    await send(session, a, prompts, req, t0, rows)


async def agent_loop(session, a, prompts, agent, t0, rows):
    await asyncio.sleep(max(0.0, t0 + agent.start - time.perf_counter()))
    for i, step in enumerate(agent.steps):
        if i:
            await asyncio.sleep(step.gap)  # the agent runs its tool (or the user types)
        await send(session, a, prompts, step, t0, rows)


async def run(a):
    prompts = Prompts(a.model)
    if a.trace == "agents":
        work = agents.workload(a.agents, max_steps=a.max_steps, gap_cap=a.gap_cap, seed=a.seed)
        prompts.prepare(s for ag in work for s in ag.steps)
        tasks = lambda s, t0, rows: [agent_loop(s, a, prompts, ag, t0, rows) for ag in work]
    else:
        work = trace.workload(a.trace, start_s=a.start, dur_s=a.dur, frac=a.frac, seed=a.seed)
        prompts.prepare(work)
        tasks = lambda s, t0, rows: [open_loop(s, a, prompts, r, t0, rows) for r in work]
    rows = []
    # read_bufsize: vLLM can stream a single line of over 512 KB for a long prompt
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), read_bufsize=1 << 23,
                                     timeout=aiohttp.ClientTimeout(total=None)) as s:
        before = [await scrape(s, b) for b in a.backends]
        t0 = time.perf_counter() + 1.0
        await asyncio.gather(*tasks(s, t0, rows))
        wall = time.perf_counter() - t0
        after = [await scrape(s, b) for b in a.backends]
        async with s.get(a.router + "/stats") as r:
            stats = await r.json()
    rows.sort(key=lambda r: r["sent"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(vars(a), wall=wall, router=stats,
                counters=[{k: after[i][k] - before[i][k] for k in COUNTERS}
                          for i in range(len(a.backends))])
    with open(a.out.replace(".csv", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    ok = sorted(r["ttft"] for r in rows if r["ttft"] is not None)
    print(f"{a.out}: {len(ok)}/{len(rows)} ok, wall {wall:.0f}s, "
          f"TTFT p50 {ok[len(ok) // 2]:.2f}s p99 {ok[int(len(ok) * 0.99)]:.2f}s, sent {stats['sent']}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="http://localhost:8000")
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", default="conversation", choices=["conversation", "toolagent", "agents"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="", help="server config label, saved with the results")
    ap.add_argument("--policy", default="", help="routing policy label, saved with the results")
    ap.add_argument("--tier-gb", type=float, default=None, help="tier budget label, saved with the results")
    g = ap.add_argument_group("Mooncake traces (open loop)")
    g.add_argument("--start", type=float, default=600)
    g.add_argument("--dur", type=float, default=600)
    g.add_argument("--frac", type=float, default=0.3)
    g.add_argument("--speed", type=float, default=1.0)
    g = ap.add_argument_group("agent sessions (closed loop)")
    g.add_argument("--agents", type=int, default=16, help="concurrent agent sessions")
    g.add_argument("--max-steps", type=int, default=30)
    g.add_argument("--gap-cap", type=float, default=30.0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
