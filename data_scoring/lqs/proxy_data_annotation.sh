bash $BASE_PATH/data_scorer/lqs/scripts/proxy_data/proxy_data_annotation.sh $BASE_PATH $CONFIG_PATH # proxy data annotation
bash $BASE_PATH/data_scorer/lqs/scripts/tools/prepare_lqs_data_scorer_train_data.sh $BASE_PATH # prepare sample-score GT pair for data scorer training

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

CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_scorer.py --lqs-process annotation_data --config ${CONFIG_FILE} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p $(yq eval .runtime.save_path ${CONFIG_FILE}) 
${CMD}



python data_scorer/lqs/tools/prepare_data_scorer_train_data.py \
    --base-path $BASE_PATH \
    --lqs-process annotation_data \
    --config-path $CONFIG_PATH \