BASE_PATH=${1}
DATA_PATH=${2-"${BASE_PATH}/pretrain_data/redpajama_sample_1B/cc_en_head/"}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 data_scorer/lqs/tools/process_data/cc.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-name cc \
    --model-path checkpoints/mistral/160M \
    --data-dir $DATA_PATH \
    --save processed_data/data_scorer/ \
    --max-length 1025 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --model-type mistral \
    --chunk-num-per-shard 1000000