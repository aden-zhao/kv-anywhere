#!/bin/bash
# Decode step time vs batch size and context, for both models (one GPU).
for M in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
  bash perlmutter/run.sh --model $M --n 1 --calibrate --calib-args=--decode-only \
      --out results/decode_$(basename $M)
done
