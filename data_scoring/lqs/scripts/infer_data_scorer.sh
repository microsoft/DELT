#!/bin/bash

CONFIG_FILE=${1-"./data_scoring/config/lqs.yaml"}

MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-1}
NNODES=1
DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export BASE_PATH=$PWD
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export NCCL_DEBUG=""

# convert tokenize
python data_scorer/lqs/tools/convert_tokenization.py \
    --base-path $BASE_PATH \
    --lqs-process scorer_data_infer \
    --config-file $CONFIG_FILE \ 

# infer
cmd="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/data_scorer/lqs/infer_data_scorer.py --base-path ${BASE_PATH} --lqs-process scorer_data_infer --config ${CONFIG_FILE} $@"

echo ${cmd}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p $(yq eval .save-path ${config-file}) 
${cmd}

# convert to jsonl
python data_scorer/lqs/tools/scorerd_data_bin2json.py \
    --lqs-process scorer_data_infer \
    --config-file $CONFIG_FILE \
