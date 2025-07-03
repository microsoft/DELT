#!/bin/bash

BASE_PATH=${1-"/home/data_efficacy"}
CONFIG_FILE=${2-"./data_scoring/config/lqs.yaml"}
MASTER_PORT=${3-2030}
GPUS_PER_NODE=${4-8}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16

CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_scorer.py --config ${CONFIG_FILE} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p $(yq eval .runtime.save_path ${CONFIG_FILE}) 
${CMD}