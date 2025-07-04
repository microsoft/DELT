#!/bin/bash

CONFIG_FILE=${1-"./data_scoring/config/lqs.yaml"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-8}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export BASE_PATH=$PWD
export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16

CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_scorer.py --lqs-process annotation_data --config ${CONFIG_FILE} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p $(yq eval .runtime.save_path ${CONFIG_FILE}) 
${CMD}



python data_scorer/lqs/tools/prepare_data_scorer_train_data.py \
    --base-path $BASE_PATH \
    --lqs-process annotation_data \
    --config-path $CONFIG_PATH \