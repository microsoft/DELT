export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 data_scorer/lqs/tools/cc_bin2jsonl.py \
    --bin-dir $BASE_PATH/processed_data/pretrain/cc-10b-random-r0.1 \
    --ds-score-path $BASE_PATH/results/data_scorer_infer/cc/cc-160M-lima-ms-official \
    --output-file $BASE_PATH/results/decoded_data/10b-random-r0.1.jsonl \
    --tokenizer-path $BASE_PATH/checkpoints/mistral/160M \
    --model-type mistral \
    --score-name score_lqs \