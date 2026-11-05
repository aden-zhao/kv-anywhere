#!/bin/bash
# Agent sessions (FreeInference), 16/32/48 concurrent agents, server config: gpu.
bash perlmutter/run.sh --model Qwen/Qwen2.5-7B-Instruct --context 131072 --cpu-gb 8 \
    --trace agents --agents 16 32 48 --modes gpu --policies least prefix prefix_cap --out results/agents_qwen25
