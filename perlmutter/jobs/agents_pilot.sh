#!/bin/bash
# Pilot for the agent extension: does 128k serving work, what do long prompts cost to
# recompute and to load from each tier, and how long does an agent run take?
R="bash perlmutter/run.sh --model Qwen/Qwen2.5-7B-Instruct --context 131072 --cpu-gb 8"
CAL="--calib-args=--lens 8192 32768 65536 120000 --reps 2"
$R --calibrate --calib-mode shared "$CAL" --out results/calib_agents
$R --calibrate --calib-mode lustre "$CAL" --out results/calib_agents
$R --trace agents --agents 16 --modes gpu shared --policies least prefix --out results/agents_pilot
