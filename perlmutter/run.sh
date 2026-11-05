#!/bin/bash
# Run sweep.py on a GPU node, from inside an allocation:
#   srun --jobid=<id> -N1 -n1 --gpus=4 bash perlmutter/run.sh --model ... --out results/...
source $SCRATCH/vllm-dev/env.sh          # vLLM venv + caches (cds into the vLLM tree)
export HF_HUB_OFFLINE=1
cd $SCRATCH/any-prefix
nvidia-smi -L | head -1
pkill -u $USER -f "vllm serve" && sleep 10   # servers left over from a crashed step
python sweep.py "$@"
