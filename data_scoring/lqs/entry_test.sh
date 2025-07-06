#!/bin/bash

DATA_PATH=${1-"data/redpajama_sample_1B/cc_en_head/"}
CONFIG_PATH=${2-"./data_scoring/config/lqs.yaml"}

export BASE_PATH=$PWD
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# download model
python data_scoring/lqs/tools/hf_download.py \
    --lqs-process full_data \
    --content model \
    --config-path $CONFIG_PATH