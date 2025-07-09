#!/bin/bash

INPUT_DATA_PATH=${1-'./data/ordered_data.jsonl'}
INPUT_MODEL_PATH=${2-'./model/input_model'}
OUTPUT_MODEL_PATH=${3-'./model/output_model'}
METHOD=${4-'pretrain'}
CONFIG_PATH=${5-'pre_train.yaml'}

MASTER_PORT=${6-2030}
GPUS_PER_NODE=${7-1}
NNODES=${8-1}

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
# export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=16

CMD="deepspeed ${DISTRIBUTED_ARGS} model_train/entry.py \
    --data_path ${INPUT_DATA_PATH} \
    --model_path ${INPUT_MODEL_PATH} \
    --save ${OUTPUT_MODEL_PATH} \
    --method ${METHOD} \
    --config_path ${CONFIG_PATH}"

echo ${CMD}
mkdir -p ${OUTPUT_MODEL_PATH}
${CMD}
