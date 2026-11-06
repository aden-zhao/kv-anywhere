#!/bin/bash
# Repeat of the GC comparison (LRU vs protect) on a different sample of 48 agents.
common="--model Qwen/Qwen2.5-7B-Instruct --context 131072 --cpu-gb 8 --read-stamps --trace agents
    --agents 48 --modes shared --gc-policies lru protect --seed 1 --out results/agents_gc_seed1"
bash perlmutter/run.sh $common --policies least --tier-gbs 50 100
bash perlmutter/run.sh $common --policies prefix --tier-gbs 50
