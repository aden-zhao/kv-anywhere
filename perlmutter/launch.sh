#!/bin/bash
# Allocate a node (or part of one), run a job script on it, and release it when the script ends.
#   bash perlmutter/launch.sh <time, e.g. 03:00:00> perlmutter/jobs/<name>.sh [gpus=4]
# 4 GPUs: a whole 40GB-A100 node (interactive QOS). Fewer: shared_interactive, billed per GPU.
# Output goes to results/<name>.out. The allocation is cancelled even if the script fails.
cd $SCRATCH/any-prefix
gpus=${3:-4}
qos=interactive; [ "$gpus" -lt 4 ] && qos=shared_interactive
jid=$(salloc --no-shell -A m4999_g -C "gpu&hbm40g" -q $qos -N 1 -G $gpus -t $1 2>&1 |
      grep -o "Granted job allocation [0-9]*" | awk '{print $4}')
[ -z "$jid" ] && { echo "allocation failed"; exit 1; }
name=$(basename $2 .sh)
echo "job $jid -> results/$name.out"
setsid nohup bash -c "srun --jobid=$jid -N1 -n1 --gpus=$gpus bash $2; scancel $jid" \
    > results/$name.out 2>&1 < /dev/null &
