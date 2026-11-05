#!/bin/bash
# Agents with a Lustre tier big enough that nothing is evicted: does capacity beat distance?
bash perlmutter/run.sh --model Qwen/Qwen2.5-7B-Instruct --context 131072 --cpu-gb 8 \
    --trace agents --agents 32 48 --modes lustre-big --tier-gb 2000 --out results/agents_qwen25
