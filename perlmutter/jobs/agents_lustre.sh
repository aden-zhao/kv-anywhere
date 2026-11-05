#!/bin/bash
# Agent sessions (FreeInference), 16/32/48 concurrent agents, server config: lustre.
bash perlmutter/run.sh --model Qwen/Qwen2.5-7B-Instruct --context 131072 --cpu-gb 8 \
    --trace agents --agents 16 32 48 --modes lustre --policies least prefix --out results/agents_qwen25
