#!/bin/bash

INPUT_DATA_PATH=${1-"./data/original_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/scored_data.jsonl"}
CONFIG_PATH=${3-"./data_scoring/config/lqs.yaml"}

# Step 1: prepare full dataset.
bash $BASE_PATH/data_scorer/lqs/scripts/prepare_full_dataset.sh $DATA_PATH $CONFIG_PATH

# Step 2: prepare target dataset.
bash $BASE_PATH/data_scorer/lqs/scripts/prepare_target_dataset.sh $CONFIG_PATH

# Step 3: prepare (sampled and annotated) training dataset for scoring model.
# Step 3.1: proxy data sampling.
bash $BASE_PATH/data_scorer/lqs/scripts/proxy_data_sampling.sh $CONFIG_PATH
# Step 3.2: proxy data annotation.
bash $BASE_PATH/data_scorer/lqs/scripts/proxy_data_annotation.sh $CONFIG_PATH

# Step 4: data scorer training.
bash $BASE_PATH/data_scorer/lqs/scripts/train_data_scorer.sh $CONFIG_PATH

# Step 5: full dataset scoring.
bash $BASE_PATH/data_scorer/lqs/scripts/infer_data_scorer.sh $CONFIG_PATH $OUTPUT_DATA_PATH
