#!/bin/bash
# Tier GC policies under memory pressure: 48 agents, shared /dev/shm tier of 50 or 100 GB,
# deleting oldest-written / least-recently-used / protect-active-sessions blocks first.
bash perlmutter/run.sh --model Qwen/Qwen2.5-7B-Instruct --context 131072 --cpu-gb 8 --read-stamps \
    --trace agents --agents 48 --modes shared --policies least \
    --gc-policies oldest lru protect --tier-gbs 50 100 --out results/agents_gc_least
