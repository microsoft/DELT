#!/bin/bash

export BASE_PATH=$PWD

CONFIG_FILE=${1-"./data_scoring/config/lqs.yaml"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-1}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/data_scorer/lqs/train_scorer.py --config ${CONFIG_FILE} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p $(yq eval .runtime.save_path ${CONFIG_FILE}) 
${CMD}