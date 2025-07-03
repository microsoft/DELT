BASE_PATH=${1}
SELECT_RATIO=${2-'0.9'}
FOLDING_NUM=${3-'3'}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 data_scorer/lqs/tools/select_pretrain_data_score_order.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-dir  $BASE_PATH/processed_data/pretrain/cc/mistral-1025 \
    --save $BASE_PATH/processed_data/pretrain/ \
    --model-type mistral \
    --model-path $BASE_PATH/checkpoints/mistral/160M \
    --data-scorer-model-type fairseq \
    --data-scorer-tokenizer-path $BASE_PATH/checkpoints/fairseq/125M \
    --ds-score-path $BASE_PATH/results/data_scorer_infer/cc/cc-160M-lima-lqs \
    --ds-ratio $SELECT_RATIO \
    --ds-gumbel-temperature 0.0 \
    --data-name cc-sgd100-160M-lima-lqs \
    --ascend True \
    --folding-order $FOLDING_NUM \