#!/bin/bash

CONFIG_PATH=${1-"data_scoring/config/lqs.yaml"}
OUTPUT_DATA_PATH=${2-"data/cc/lqs_scored_data.jsonl"}

MASTER_PORT=${3-2031}
GPUS_PER_NODE=${4-1}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
    
# convert tokenize
python data_scoring/lqs/tools/convert_tokenization.py \
    --lqs-process scorer_data_infer \
    --config-path $CONFIG_PATH

# infer
cmd="deepspeed ${DISTRIBUTED_ARGS} data_scoring/lqs/infer_data_scorer.py --lqs-process scorer_data_infer --config ${CONFIG_PATH}"

echo ${cmd}
${cmd}

# convert to jsonl
python data_scoring/lqs/tools/scorerd_data_bin2json.py \
    --lqs-process scorer_data_infer \
    --jsonl-output-path $OUTPUT_DATA_PATH \
    --config-path $CONFIG_PATH
