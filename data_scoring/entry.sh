#!/bin/bash

INPUT_DATA_PATH=${1-"./data/original_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/scored_data.jsonl"}
METHOD=${3-"lqs"} 
CONFIG_PATH=${4-"./data_scoring/config/lqs.yaml"}


if [ -z "$1" ]; then
  SUPPORTED_METHODS=("lqs")
  echo "Error: No method specified. Please use one of the following: ${SUPPORTED_METHODS[*]}"
  exit 1
fi

if [ "$1" == "--help" ]; then
  echo "Usage: $0 <method>"
  echo "Available methods:"
  echo "  lqs  - Run LQS scoring script."
  exit 0
fi

if [ "$METHOD" == "lqs" ]; then
  if [ ! -x "./data_scorer/lqs/entry.sh" ]; then
    echo "Error: Script .sh does not exist or is not executable."
    exit 1
  fi
  bash ./data_scorer/lqs/entry_test.sh $INPUT_MODEL_PATH $OUTPUT_DATA_PATH $METHOD $CONFIG_PATH
fi
