#!/usr/bin/env python3
"""Replay a trace workload against the router and record per-request latency.

    python3 replay.py --router http://localhost:8000 --backends <urls> \
        --trace conversation --frac 0.3 --speed 4 --out results/run.csv

Requests are sent open-loop at their trace arrival times (divided by --speed),
with the prompt as token ids, `ignore_eos`, and the trace's output length. Each
CSV row is one request: arrival, TTFT, end-to-end time, replica, prompt length,
and cached tokens. A JSON next to the CSV holds each backend's prefix-cache hit
counters (local GPU vs external tier) over the run, scraped from /metrics.
"""
import argparse
import asyncio
import csv
import json
import time

import aiohttp

import trace

COUNTERS = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
            "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
            "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]


async def scrape(session, url):
    async with session.get(url + "/metrics") as r:
        text = await r.text()
    out = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        name = line.split("{")[0].split(" ")[0]
        if name in out:
            out[name] += float(line.rsplit(" ", 1)[1])
    return out


async def one(session, router, req, model, t0, rows):
    await asyncio.sleep(max(0.0, t0 + req.send_at - time.perf_counter()))
    body = {"model": model, "prompt": req.tokens, "max_tokens": req.output_len,
            "ignore_eos": True, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    sent = time.perf_counter()
    ttft, cached, n_out, err = None, None, 0, ""
    try:
        async with session.post(router + "/v1/completions", json=body) as r:
            replica = int(r.headers.get("X-Replica", -1))
            async for line in r.content:
                if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                    continue
                msg = json.loads(line[6:])
                if ttft is None and msg.get("choices"):
                    ttft = time.perf_counter() - sent
                if msg.get("usage"):
                    n_out = msg["usage"]["completion_tokens"]
                    details = msg["usage"].get("prompt_tokens_details") or {}
                    cached = details.get("cached_tokens")
    except Exception as e:  # keep going; the row records the failure
        replica, err = -1, repr(e)[:200]
    rows.append(dict(arrival=round(req.send_at, 4), ttft=ttft, e2e=time.perf_counter() - sent,
                     replica=replica, input_len=req.input_len, output_len=n_out,
                     cached=cached, session=req.session, err=err))


async def run(a):
    reqs = trace.workload(a.trace, start_s=a.start, dur_s=a.dur, frac=a.frac, seed=a.seed)
    for r in reqs:
        r.send_at = r.t / a.speed
        r.tokens = trace.token_ids(r)  # build prompts up front, off the send path
    rows = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                     timeout=aiohttp.ClientTimeout(total=None)) as s:
        before = [await scrape(s, b) for b in a.backends]
        t0 = time.perf_counter() + 1.0
        await asyncio.gather(*(one(s, a.router, r, a.model, t0, rows) for r in reqs))
        wall = time.perf_counter() - t0
        after = [await scrape(s, b) for b in a.backends]
        async with s.get(a.router + "/stats") as r:
            stats = await r.json()
    rows.sort(key=lambda r: r["arrival"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(vars(a), wall=wall, router=stats,
                counters=[{k: after[i][k] - before[i][k] for k in COUNTERS}
                          for i in range(len(a.backends))])
    with open(a.out.replace(".csv", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    ok = [r["ttft"] for r in rows if r["ttft"] is not None]
    ok.sort()
    print(f"{a.out}: {len(ok)}/{len(rows)} ok, wall {wall:.0f}s, "
          f"TTFT p50 {ok[len(ok) // 2]:.2f}s p99 {ok[int(len(ok) * 0.99)]:.2f}s, sent {stats['sent']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="http://localhost:8000")
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", default="conversation")
    ap.add_argument("--start", type=float, default=600)
    ap.add_argument("--dur", type=float, default=600)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
