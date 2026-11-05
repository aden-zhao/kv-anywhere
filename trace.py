"""Mooncake FAST'25 traces -> replayable workloads with real prefix sharing.

Each trace line is {timestamp (ms), input_length, output_length, hash_ids}. A hash
id names one 512-token block, and the ids are chained: two requests with the same
first k ids share their first k blocks. We turn ids back into token ids (one fixed
random block per id), so the replayed prompts share exactly what the real ones did.

The trace comes from a large cluster with thousands of live sessions. A 4-GPU
cluster can't hold that many, so we sample whole sessions (`frac`) to shrink the
working set, and compress time (`speed`) to set the load. Sampling sessions instead
of requests keeps every multi-turn conversation intact.
"""
import json
import os
import random
import urllib.request
from collections import Counter
from dataclasses import dataclass

BLOCK = 512  # tokens per trace hash id
URL = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/{}_trace.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Request:
    t: float          # arrival, seconds from the window start (before speedup)
    input_len: int
    output_len: int
    hash_ids: list
    session: int


def load(name="toolagent"):
    path = os.path.join(HERE, "traces", f"{name}_trace.jsonl")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL.format(name), path)
    with open(path) as f:
        return [json.loads(line) for line in f]


def sessions(reqs, hot_count=200):
    """Union requests that share any block other than the few hot system prompts."""
    cnt = Counter(h for r in reqs for h in r["hash_ids"])
    hot = {h for h, c in cnt.items() if c > hot_count}
    parent = list(range(len(reqs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for i, r in enumerate(reqs):
        for h in r["hash_ids"]:
            if h in hot:
                continue
            if h in owner:
                parent[find(i)] = find(owner[h])
            else:
                owner[h] = i
    return [find(i) for i in range(len(reqs))]


def workload(name="toolagent", start_s=600, dur_s=600, frac=1.0, max_len=32768,
             max_out=1024, seed=0):
    reqs = load(name)
    sess = sessions(reqs)
    keep = random.Random(seed)
    chosen = {s for s in sorted(set(sess)) if keep.random() < frac}
    out = []
    for r, s in zip(reqs, sess):
        t = r["timestamp"] / 1000 - start_s
        if not (0 <= t < dur_s) or s not in chosen or r["input_length"] > max_len:
            continue
        out.append(Request(t, r["input_length"], min(r["output_length"], max_out),
                           r["hash_ids"], s))
    return out


def block_tokens(h, lo=1000, hi=100000):
    """The token ids for hash id h: a fixed random block, the same every time."""
    rng = random.Random(h)
    return [rng.randrange(lo, hi) for _ in range(BLOCK)]


def token_ids(req):
    """A request's prompt: its blocks' tokens, cut to its length."""
    return [t for h in req.hash_ids for t in block_tokens(h)][:req.input_len]


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "toolagent"
    for frac in [1.0, 0.3, 0.1]:
        w = workload(name, frac=frac)
        toks = sum(r.input_len for r in w)
        print(f"{name} frac={frac}: {len(w)} reqs, {toks / 600:.0f} input tok/s at 1x, "
              f"{len({r.session for r in w})} sessions")
