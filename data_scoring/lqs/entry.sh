#!/bin/bash

INPUT_DATA_PATH=${1-"./data/original_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/scored_data.jsonl"}
METHOD=${3-"lqs"} 
CONFIG_PATH=${4-"./data_scoring/config/lqs.yaml"}

# Step 1: prepare full dataset.
bash $BASE_PATH/data_scorer/lqs/scripts/tools/prepare_full_dataset.sh $DATA_PATH $CONFIG_PATH # tokenize cc use fairseq 125M


# Step 2: prepare target dataset.
bash $BASE_PATH/data_scorer/lqs/prepare_target_dataset.sh $CONFIG_PATH # tokenize lima


# Step 3: prepare (sampling and annotation) training dataset for scoring model.
# Step 3.1: proxy data sampling.
bash $BASE_PATH/data_scorer/lqs/scripts/tools/sample_proxy_data.sh $CONFIG_PATH # sample proxy data from cc
# Step 3.2: proxy data annotation.
bash $BASE_PATH/data_scorer/lqs/scripts/proxy_data/proxy_data_annotation.sh $CONFIG_PATH # proxy data annotation

# Step 4: data scorer training.
bash $BASE_PATH/data_scorer/lqs/scripts/train_data_scorer.sh $CONFIG_PATH # data scorer training

# Step 5: full dataset scoring.
bash $BASE_PATH/data_scorer/lqs/scripts/infer_data_scorer.sh $CONFIG_PATH # data scorer infer (data scoring) 