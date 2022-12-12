#!/bin/bash
# NUM_PROC=$1
# shift
# python -m torch.distributed.launch --nproc_per_node=$NUM_PROC channel_search_distributed.py "$@"

set -x

PARTITION=$1
JOB_NAME=$2
# CONFIG=$3
# WORK_DIR=$4
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE channel_search_distributed.py ${PY_ARGS}