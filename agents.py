"""FreeInference agent traces -> closed-loop agent sessions with real prefix sharing.

FreeInference (Harvard, CC-BY-4.0) logs coding agents (codex, opencode, ...) in
production. Each session is a list of steps; a step's prompt is the previous step's
prompt plus the latest tool result, so ~99.6% of it is shared with the step before.
Every request lists ids for its 16-token blocks, unique within a week, so sharing
across sessions (common system prompts) is kept too.

We regroup those ids into chained 512-token ids (id i names the whole prefix up to
block 32*(i+1)), so trace.token_ids, policies.py and sim.py work unchanged.

Replay is closed-loop: an agent sends step k+1 only after step k finishes, plus the
recorded gap between them (tool time, or a human typing; capped by `gap_cap`).
"""
import hashlib
import io
import json
import os
import random
import urllib.request
from dataclasses import dataclass, field

import trace

WEEK = "2026-08-30_2026-09-05"
URL = ("https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/resolve/main/"
       "release/{week}/traces/{day}.jsonl.zst")
GROUP = trace.BLOCK // 16  # 16-token trace blocks per 512-token id


@dataclass
class Step:
    input_len: int
    output_len: int
    hash_ids: list
    gap: float          # seconds between the previous step finishing and this one starting
    session: int = 0
    t: float = 0.0      # filled in at replay time


@dataclass
class Agent:
    sid: str
    start: float        # seconds from the window start (first step's arrival)
    steps: list = field(default_factory=list)


def regroup(block_ids):
    """16-token block ids -> chained 512-token ids (a partial last group gets its own id)."""
    out, h = [], hashlib.sha1()
    for i in range(0, len(block_ids), GROUP):
        h.update(json.dumps(block_ids[i:i + GROUP]).encode())
        out.append(int.from_bytes(h.copy().digest()[:8], "little"))
    return out


def _download(day):
    path = os.path.join(trace.HERE, "traces", f"freeinference_{day}.jsonl.zst")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL.format(week=WEEK, day=day), path)
    return path


def load_day(day="2026-09-01", min_peak=32768, max_len=131072, min_steps=10):
    """Top-level sessions of one day whose context peaks within [min_peak, max_len].
    Cached as compact JSON next to the raw file (parsing a day takes a while)."""
    cache = os.path.join(trace.HERE, "traces", f"agents_{day}_{min_peak}_{max_len}_{min_steps}.json")
    if os.path.exists(cache):
        return [Agent(a["sid"], a["start"], [Step(**s) for s in a["steps"]]) for a in json.load(open(cache))]
    import zstandard
    agents, t0 = [], None
    with open(_download(day), "rb") as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"):
            s = json.loads(line)
            reqs = sorted(s["requests"], key=lambda r: r["step"])
            if len(reqs) < min_steps:
                continue
            lens = [len(r["block_ids"]) * 16 for r in reqs]
            if not (min_peak <= max(lens) <= max_len):
                continue
            steps, prev_end = [], None
            for r, n in zip(reqs, lens):
                out = (r.get("provider_tokens") or {}).get("completion") or 1
                gap = 0.0 if prev_end is None else max(0.0, (r["start_ms"] - prev_end) / 1000)
                steps.append(Step(n, out, regroup(r["block_ids"]), gap))
                prev_end = r["end_ms"]
            agents.append(Agent(s["session_id"], reqs[0]["start_ms"] / 1000, steps))
    agents.sort(key=lambda a: a.start)
    t0 = agents[0].start if agents else 0
    for a in agents:
        a.start -= t0
    with open(cache, "w") as f:
        json.dump([dict(sid=a.sid, start=a.start, steps=[s.__dict__ for s in a.steps]) for a in agents], f)
    return agents


def workload(n_agents=16, day="2026-09-01", max_steps=30, gap_cap=30.0, max_out=256,
             stagger=60.0, seed=0, **filters):
    """n_agents sessions, each truncated to its first max_steps steps, starting
    evenly over `stagger` seconds. Gaps are capped at gap_cap (a human may pause
    for minutes); outputs at max_out (the next prompt comes from the trace, not
    from our output, so this doesn't change what gets reused)."""
    pool = load_day(day, **filters)
    agents = random.Random(seed).sample(pool, min(n_agents, len(pool)))
    for i, a in enumerate(agents):
        a.start = stagger * i / max(1, len(agents))
        a.steps = a.steps[:max_steps]
        for s in a.steps:
            s.gap = min(s.gap, gap_cap)
            s.output_len = min(s.output_len, max_out)
            s.session = i
    return agents


if __name__ == "__main__":
    import statistics as st
    pool = load_day()
    steps = [s for a in pool for s in a.steps]
    print(f"{len(pool)} sessions, {len(steps)} steps; steps/session median "
          f"{st.median(len(a.steps) for a in pool)}; context median {st.median(s.input_len for s in steps)}, "
          f"max {max(s.input_len for s in steps)}; gap median {st.median(s.gap for s in steps if s.gap):.1f}s")
