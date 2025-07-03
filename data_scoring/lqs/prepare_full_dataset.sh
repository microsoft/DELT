export BASE_PATH=$PWD

DATA_PATH=${1-"${BASE_PATH}/pretrain_data/redpajama_sample_1B/cc_en_head/"}
CONFIG_PATH=${2-"./data_scoring/config/lqs.yaml"}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# download dataset
python utils.py --content=model --id=$HF_MODEL_ID --save_dir=$OUTPUT_MODEL_PATH

# process data
python data_scoring/lqs/tools/process_data/pretrain_data_process.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-name cc \
    --model-path checkpoints/mistral/160M \
    --data-dir $DATA_PATH \
    --save processed_data/data_scorer/lqs \
    --max-length 1025 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --model-type mistral \
    --chunk-num-per-shard 1000000