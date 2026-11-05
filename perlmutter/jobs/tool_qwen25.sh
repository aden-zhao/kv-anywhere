#!/bin/bash
# Toolagent trace: most reuse is 3 hot system prompts, plus shorter multi-turn sessions.
# Pure prefix affinity runs at one load only (it collapses onto one replica at any load).
R="bash perlmutter/run.sh --model Qwen/Qwen2.5-7B-Instruct --trace toolagent --frac 0.3"
$R --modes gpu --policies prefix --speeds 2 --out results/tool_qwen25
$R --modes gpu private shared --policies least prefix_cap --speeds 2 3 4 5 --out results/tool_qwen25
