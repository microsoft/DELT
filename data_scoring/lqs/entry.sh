export BASE_PATH=$PWD

INPUT_DATA_PATH=${1-"./data/original_data.jsonl"}
OUTPUT_DATA_PATH=${2-"./data/scored_data.jsonl"}
METHOD=${3-"lqs"} 
CONFIG_PATH=${4-"./data_scoring/config/lqs.yaml"}

# Step 1: prepare full dataset.
python $BASE_PATH/data_scorer/lqs/tools/get_checkpoints_lqs.py # Base model for annotation (Mistral 160M) and base model of the data scorer (fairseq 125M)
bash $BASE_PATH/data_scorer/lqs/scripts/tools/process_cc_lqs.sh $BASE_PATH $DATA_PATH # tokenize cc use fairseq 125M

# Step 2: prepare target dataset.
python $BASE_PATH/data_scorer/lqs/tools/get_data_lqs.py # lima dataset to calculate the downstream loss
bash $BASE_PATH/data_scorer/lqs/scripts/tools/process_lima_lqs.sh $BASE_PATH # tokenize lima

# Step 3: prepare (sampling and annotation) training dataset for scoring model.
# Step 3.1: proxy data sampling.
bash $BASE_PATH/data_scorer/lqs/scripts/tools/sample_proxy_data.sh $BASE_PATH # sample proxy data from cc
# Step 3.2: proxy data annotation.
bash $BASE_PATH/data_scorer/lqs/scripts/proxy_data/160M.sh $BASE_PATH # proxy data annotation
bash $BASE_PATH/data_scorer/lqs/scripts/tools/prepare_lqs_data_scorer_train_data.sh $BASE_PATH # prepare sample-score GT pair for data scorer training

# Step 4: data scorer training.
bash $BASE_PATH/data_scorer/lqs/scripts/train.sh $BASE_PATH # data scorer training

# Step 5: full dataset scoring.
bash $BASE_PATH/data_scorer/lqs/scripts/tools/convert_tokenization.sh $BASE_PATH # convert tokenized data for data scorer inference 
bash $BASE_PATH/data_scorer/lqs/scripts/infer.sh $BASE_PATH $SCORER_PATH # data scorer infer (data scoring) 
python $BASE_PATH/data_scorer/lqs/scripts/tools/scorerd_data_bin2json.sh $BASE_PATH $SCORER_PATH # convert bin to jsonl format
