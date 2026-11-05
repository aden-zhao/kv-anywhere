#!/bin/bash
# Same conversation sweep with Qwen3-8B: 2.6x more KV bytes per token than Qwen2.5-7B,
# so the same 100 GB tier holds 2.6x fewer tokens and each load moves more bytes.
R="bash perlmutter/run.sh --model Qwen/Qwen3-8B"
$R --calibrate --out results/calib_qwen3
$R --trace conversation --frac 0.3 --modes gpu shared --policies least prefix_cap --speeds 2 3 4 5 --out results/conv_qwen3
