#!/bin/bash

CONFIG_PATH=${1-"./data_scoring/config/lqs.yaml"}

MASTER_PORT=${2-2031}
GPUS_PER_NODE=${3-1}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3

# infer
cmd="deepspeed ${DISTRIBUTED_ARGS} data_scoring/lqs/infer_data_scorer.py --lqs-process scorer_data_infer --config ${CONFIG_PATH}"

echo ${cmd}
${cmd}

# convert tokenize
# python data_scoring/lqs/tools/convert_tokenization.py \
#     --lqs-process scorer_data_infer \
#     --config-path $CONFIG_PATH

# # convert to jsonl
# python data_scorer/lqs/tools/scorerd_data_bin2json.py \
#     --lqs-process scorer_data_infer \
#     --config-file $CONFIG_PATH \
