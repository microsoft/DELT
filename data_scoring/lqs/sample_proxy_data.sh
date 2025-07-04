CONFIG_PATH=${1-"./data_scoring/config/lqs.yaml"}

export BASE_PATH=$PWD
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 data_scorer/lqs/tools/sample_proxy_data.py \
    --base-path $BASE_PATH \
    --lqs-process proxy_data \
    --config-path $CONFIG_PATH \