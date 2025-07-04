export BASE_PATH=$PWD

CONFIG_FILE=${1-"./data_scoring/config/lqs.yaml"}

MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-1}
NNODES=1

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python data_scorer/lqs/tools/convert_tokenization.py \
    --lqs-process scorer_data_infer \
    --config-file $CONFIG_FILE \ 

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""

cmd="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/data_scorer/lqs/infer_data_scorer.py  --lqs-process scorer_data_infer --config ${CONFIG_FILE} $@"

echo ${cmd}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p $(yq eval .save-path ${config-file}) 
${cmd}


python data_scorer/lqs/tools/cc_bin2jsonl.py \
    --lqs-process scorer_data_infer \
    --config-file $CONFIG_FILE \ 