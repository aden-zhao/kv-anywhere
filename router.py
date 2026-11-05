#!/usr/bin/env python3
"""A routing proxy in front of N vLLM servers.

    python3 router.py --backends http://localhost:8100 http://localhost:8101 ... \
        --policy least --port 8000

Clients POST /v1/completions with `prompt` as a list of token ids. The router
picks a backend with policies.choose(), forwards the request, and streams the
response back with an X-Replica header. It only sees what it sends, so its view
of each replica (pending prefill tokens, cached prefix blocks) is an estimate.

POST /config {"policy": ..., "cap": ..., "capacity_blocks": ...} switches the
policy and clears the router's state between runs. GET /stats shows the state.
"""
import argparse

import aiohttp
from aiohttp import web

import policies


class Router:
    def __init__(self, backends, policy, cap, capacity_blocks):
        self.backends = backends
        self.configure(policy, cap, capacity_blocks)

    def configure(self, policy, cap, capacity_blocks):
        self.policy, self.cap, self.capacity_blocks = policy, cap, capacity_blocks
        self.replicas = [policies.Replica(capacity_blocks) for _ in self.backends]
        self.sent = [0] * len(self.backends)

    async def completions(self, request):
        body = await request.json()
        tokens = body["prompt"]
        keys = policies.block_keys(tokens)
        j = policies.choose(self.policy, self.replicas, keys, self.cap)
        rep = self.replicas[j]
        miss = len(tokens) - rep.match(keys) * policies.BLOCK
        rep.pending += miss
        rep.insert(keys)
        self.sent[j] += 1

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                           "X-Replica": str(j)})
        await resp.prepare(request)
        first = True
        try:
            for attempt in range(2):
                try:
                    async with self.session.post(self.backends[j] + "/v1/completions", json=body) as up:
                        async for chunk in up.content.iter_any():
                            if first:
                                rep.pending -= miss
                                first = False
                            await resp.write(chunk)
                    break
                except aiohttp.ServerDisconnectedError:
                    # A pooled keep-alive connection was closed under us. Safe to retry
                    # only if nothing has reached the client yet.
                    if not first or attempt == 1:
                        raise
        finally:
            if first:
                rep.pending -= miss
        await resp.write_eof()
        return resp

    async def config(self, request):
        c = await request.json()
        self.configure(c.get("policy", self.policy), c.get("cap", self.cap),
                       c.get("capacity_blocks", self.capacity_blocks))
        return web.json_response({"ok": True, "policy": self.policy})

    async def stats(self, request):
        return web.json_response({
            "policy": self.policy, "cap": self.cap, "sent": self.sent,
            "pending": [r.pending for r in self.replicas],
            "cached_blocks": [len(r.cache) for r in self.replicas]})

    async def start(self, app):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None))

    async def stop(self, app):
        await self.session.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--policy", default="least", choices=["least", "prefix", "prefix_cap"])
    ap.add_argument("--cap", type=int, default=16384, help="prefix_cap balance threshold (tokens)")
    ap.add_argument("--capacity-blocks", type=int, default=1000,
                    help="router's guess of each replica's cache size, in 512-token blocks")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    r = Router(a.backends, a.policy, a.cap, a.capacity_blocks)
    app = web.Application(client_max_size=1 << 28)
    app.on_startup.append(r.start)
    app.on_cleanup.append(r.stop)
    app.router.add_post("/v1/completions", r.completions)
    app.router.add_post("/config", r.config)
    app.router.add_get("/stats", r.stats)
    web.run_app(app, port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
