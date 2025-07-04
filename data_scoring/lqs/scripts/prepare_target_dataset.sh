#!/bin/bash

DATA_PATH=${1-"${BASE_PATH}/pretrain_data/redpajama_sample_1B/cc_en_head/"}
CONFIG_PATH=${2-"./data_scoring/config/lqs.yaml"}

export BASE_PATH=$PWD
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# download dataset
python data_scoring/lqs/tools/hf_download.py \
    --lqs-process target_data \
    --content model \
    --config-path $CONFIG_PATH \ 

# process data (lima)
python data_scoring/lqs/tools/process_data/hf_data_process.py \
    --base-path $BASE_PATH \
    --lqs-process target_data \
    --config-path $CONFIG_PATH \