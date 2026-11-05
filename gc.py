#!/usr/bin/env python3
"""Keep a KV tier directory under a byte budget by deleting the oldest blocks.

    python3 gc.py --budget-gb 100 /dev/shm/kv/shared

vLLM's file-system tier never evicts. DeepSeek runs a GC worker on each storage
node that drops least-recently-used KV under a watermark; this is the same idea
with oldest-written-first (tmpfs doesn't track reads). A block deleted between
lookup and load just gets recomputed (kv_load_failure_policy=recompute).
"""
import argparse
import os
import time


def scan(root):
    files = []
    for d, _, names in os.walk(root):
        for n in names:
            if n.endswith(".bin"):
                p = os.path.join(d, n)
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
    return files


def collect(root, budget):
    files = scan(root)
    total = sum(f[1] for f in files)
    if total <= budget:
        return 0
    freed = 0
    for _, size, p in sorted(files):
        if total - freed <= 0.9 * budget:
            break
        try:
            os.remove(p)
            freed += size
        except FileNotFoundError:
            pass
    return freed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, required=True, help="per directory")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    budget = a.budget_gb * 1e9
    while True:
        for d in a.dirs:
            collect(d, budget)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
