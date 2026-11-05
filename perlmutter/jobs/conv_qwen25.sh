#!/bin/bash
# Calibration, then the main sweep: conversation trace, Qwen2.5-7B, 3 server configs x policies x 4 loads.
R="bash perlmutter/run.sh --model Qwen/Qwen2.5-7B-Instruct"
$R --calibrate --out results/calib_qwen25
$R --trace conversation --frac 0.3 --speeds 3 5 7 9 --out results/conv_qwen25
