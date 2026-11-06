#!/usr/bin/env python3
"""Keep a KV tier directory under a byte budget by deleting blocks.

    python3 tier_gc.py --budget-gb 100 --policy protect /dev/shm/kv/shared

vLLM's file-system tier never evicts. DeepSeek runs a GC worker on each storage
node; this is the same idea, with three choices of what to delete first:
  oldest   oldest-written blocks. Works with plain vLLM, but tmpfs doesn't record
           reads, so a block in constant use looks as old as a dead one.
  lru      least recently used. Needs kvtier.ReadStampedTier to record reads.
  protect  like lru, but never a block used within the last window (about three
           of a session's steps). If every block is that fresh, delete newcomers
           that haven't been read yet. Under pressure this keeps some sessions
           whole instead of every session partly evicted (Denning's working set,
           plus bypassing newcomers). The window is measured, not set:
           3 x the median time between reads of a block.
A block deleted between lookup and load just gets recomputed (kv_load_failure_policy).
"""
import argparse
import os
import time
from dataclasses import dataclass


@dataclass
class Block:
    path: str
    size: int
    written: float      # mtime
    last_used: float    # last read (atime, stamped by kvtier.py) or write
    read: bool          # read since it was written?


def scan(root):
    blocks = []
    for d, _, names in os.walk(root):
        for n in names:
            if n.endswith(".bin"):
                p = os.path.join(d, n)
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    continue
                read = st.st_atime > st.st_mtime
                blocks.append(Block(p, st.st_size, st.st_mtime, max(st.st_atime, st.st_mtime), read))
    return blocks


class Window:
    """3 x the median time between consecutive uses of a block: about three steps."""

    def __init__(self, start=60.0):
        self.seconds, self.prev, self.gaps = start, {}, []

    def update(self, blocks):
        for b in blocks:
            before = self.prev.get(b.path, b.written)
            if b.read and b.last_used > before + 1:
                self.gaps = (self.gaps + [b.last_used - before])[-500:]
            self.prev[b.path] = b.last_used
        if len(self.gaps) >= 20:
            self.seconds = 3 * sorted(self.gaps)[len(self.gaps) // 2]


def deletion_order(blocks, policy, window, now):
    if policy == "oldest":
        return sorted(blocks, key=lambda b: b.written)
    if policy == "lru":
        return sorted(blocks, key=lambda b: b.last_used)
    stale = sorted((b for b in blocks if now - b.last_used > window.seconds), key=lambda b: b.last_used)
    fresh = [b for b in blocks if now - b.last_used <= window.seconds]
    newcomers = sorted((b for b in fresh if not b.read), key=lambda b: -b.written)  # newest first
    in_use = sorted((b for b in fresh if b.read), key=lambda b: b.last_used)
    return stale + newcomers + in_use


def to_delete(blocks, budget, policy, window, now):
    """Nothing while under budget; once over, enough blocks to get down to 90% of it
    (the slack keeps it from deleting a few blocks every second). sim.py runs this too."""
    window.update(blocks)
    total = sum(b.size for b in blocks)
    if total <= budget:
        return []
    out = []
    for b in deletion_order(blocks, policy, window, now):
        if total <= 0.9 * budget:
            break
        out.append(b)
        total -= b.size
    return out


def collect(root, budget, policy, window):
    for b in to_delete(scan(root), budget, policy, window, time.time()):
        try:
            os.remove(b.path)
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, required=True, help="per directory")
    ap.add_argument("--policy", default="oldest", choices=["oldest", "lru", "protect"])
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    budget = a.budget_gb * 1e9
    windows = {d: Window() for d in a.dirs}
    while True:
        for d in a.dirs:
            collect(d, budget, a.policy, windows[d])
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
