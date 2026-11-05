#!/usr/bin/env python3
"""Run every (server config, policy, load) combination on one 4-GPU node.

    python3 sweep.py --model Qwen/Qwen2.5-7B-Instruct --trace conversation --frac 0.3 \
        --speeds 3 5 7 9 --out results/conv_qwen25

Server configs (how the 4 vLLM replicas are started):
    gpu      GPU prefix cache only
    private  plus a file-system KV tier in /dev/shm, one directory per replica
    shared   plus one file-system KV tier that all four replicas read and write
Each config starts the replicas once and runs every policy x load on them. Before
each run, every cache is cleared: GPU and CPU caches over HTTP, the tier by deleting
its files, the router's estimate by /config.
Run it inside one srun step (Slingshot allows only 3 network applications per node).
"""
import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
TIER = "/dev/shm/kv"
ROUTER = "http://localhost:8000"
POLICIES = {"gpu": ["least", "prefix", "prefix_cap"],
            "private": ["least", "prefix_cap"],
            "shared": ["least", "prefix_cap"]}


def port(i):
    return 8100 + i


def tier_dirs(mode, n):
    if mode == "private":
        return [f"{TIER}/r{i}" for i in range(n)]
    if mode == "shared":
        return [f"{TIER}/shared"]
    return []


def serve_cmd(a, mode, i):
    cmd = ["vllm", "serve", a.model, "--port", str(port(i)),
           "--max-model-len", "32768", "--gpu-memory-utilization", str(a.gpu_util),
           "--enable-prompt-tokens-details", "--seed", "0", "--generation-config", "vllm"]
    if mode == "gpu":
        return cmd
    root = tier_dirs(mode, a.n)[i if mode == "private" else 0]
    tier = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",  # a block the GC just deleted is recomputed
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": int(a.cpu_gb * 1e9),  # pinned staging area on the host
                "block_size": a.chunk,                    # tokens per tier file
                "secondary_tiers": [{"type": "fs", "root_dir": root}]}}
    return cmd + ["--kv-transfer-config", json.dumps(tier)]


def responding(p):
    try:
        requests.get(f"http://localhost:{p}/health", timeout=2)
        return True
    except requests.RequestException:
        return False


def stop(procs):
    """Kill each process group (vLLM's engine runs in a child process)."""
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    time.sleep(5)


@contextlib.contextmanager
def replicas(a, mode, logdir):
    """Start n vLLM servers for one config; yield the GPU KV cache size (tokens)."""
    busy = [p for p in [8000] + [port(i) for i in range(a.n)] if responding(p)]
    if busy:  # a crashed earlier run would silently serve this one
        sys.exit(f"ports {busy} already serving; kill leftover servers first")
    shutil.rmtree(TIER, ignore_errors=True)
    for d in tier_dirs(mode, a.n):
        os.makedirs(d)
    procs = []
    try:
        for i in range(a.n):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), VLLM_SERVER_DEV_MODE="1")
            log = open(f"{logdir}/{mode}_server{i}.log", "w")
            procs.append(subprocess.Popen(serve_cmd(a, mode, i), env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True))
        t = time.time()
        for i, p in enumerate(procs):
            while not responding(port(i)):
                if p.poll() is not None:
                    sys.exit(f"server {i} exited; see {logdir}/{mode}_server{i}.log")
                time.sleep(1)
        log = open(f"{logdir}/{mode}_server0.log").read()
        kv = int(re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)[-1].replace(",", ""))
        print(f"[{mode}] servers up in {time.time() - t:.0f}s, GPU KV {kv} tokens/replica", flush=True)
        yield kv
    finally:
        stop(procs)


@contextlib.contextmanager
def helpers(a, mode, logdir):
    """The router, plus the tier's GC when there is a tier."""
    backends = [f"http://localhost:{port(i)}" for i in range(a.n)]
    cmds = [[sys.executable, f"{HERE}/router.py", "--backends", *backends]]
    dirs = tier_dirs(mode, a.n)
    if dirs:
        cmds.append([sys.executable, f"{HERE}/gc.py", "--budget-gb", str(a.tier_gb / len(dirs)), *dirs])
    log = open(f"{logdir}/{mode}_router.log", "w")
    procs = [subprocess.Popen(c, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
             for c in cmds]
    time.sleep(3)
    try:
        yield backends
    finally:
        stop(procs)


def clear_caches(a, mode):
    for i in range(a.n):
        requests.post(f"http://localhost:{port(i)}/reset_prefix_cache",
                      params={"reset_external": "true"}).raise_for_status()
    for d in tier_dirs(mode, a.n):  # delete block files, keep vLLM's directories
        for root, _, names in os.walk(d):
            for n in names:
                if n.endswith(".bin"):
                    os.remove(os.path.join(root, n))


def kv_bytes_per_token(model):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return c.num_hidden_layers * 2 * c.num_key_value_heads * head_dim * 2  # K and V, bf16


def run(a, mode, policy, speed, kv, backends):
    clear_caches(a, mode)
    # The router guesses what each replica caches: its GPU cache, plus the private
    # tier that only that replica can read. (A shared tier is everyone's.)
    blocks = kv // 512
    if mode == "private":
        blocks += int(a.tier_gb / a.n * 1e9 / kv_bytes_per_token(a.model) / 512)
    requests.post(ROUTER + "/config", json={
        "policy": policy, "cap": a.cap, "capacity_blocks": blocks}).raise_for_status()
    name = f"{a.trace}_f{a.frac}_{mode}_{policy}_s{speed:g}"
    subprocess.run([sys.executable, f"{HERE}/replay.py", "--backends", *backends,
                    "--model", a.model, "--trace", a.trace, "--frac", str(a.frac),
                    "--dur", str(a.dur), "--speed", str(speed), "--out", f"{a.out}/{name}.csv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--modes", nargs="+", default=["gpu", "private", "shared"])
    ap.add_argument("--policies", nargs="+", default=None, help="override POLICIES")
    ap.add_argument("--trace", default="conversation")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--dur", type=float, default=600)
    ap.add_argument("--speeds", type=float, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--cpu-gb", type=float, default=4, help="per-replica CPU staging tier")
    ap.add_argument("--chunk", type=int, default=256, help="tokens per tier file")
    ap.add_argument("--tier-gb", type=float, default=100, help="total tier budget")
    ap.add_argument("--calibrate", action="store_true", help="only run calibrate.py (shared config)")
    ap.add_argument("--calib-args", default="", help="extra arguments for calibrate.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logdir = os.path.join(a.out, "logs")
    os.makedirs(logdir, exist_ok=True)

    if a.calibrate:
        with replicas(a, "shared", logdir):
            subprocess.run([sys.executable, f"{HERE}/calibrate.py", "--backend", f"http://localhost:{port(0)}",
                            "--model", a.model, "--out", f"{a.out}/calib.json", *a.calib_args.split()],
                           check=True)
        return

    for mode in a.modes:
        with replicas(a, mode, logdir) as kv, helpers(a, mode, logdir) as backends:
            for policy in a.policies or POLICIES[mode]:
                for speed in a.speeds:
                    run(a, mode, policy, speed, kv, backends)


if __name__ == "__main__":
    main()
