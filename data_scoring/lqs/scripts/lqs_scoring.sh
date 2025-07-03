export BASE_PATH=$PWD

python3 $BASE_PATH/data_scorer/lqs/tools/get_checkpoints_lqs.py # Base model for annotation (Mistral 160M) and base model of the data scorer (fairseq 125M)
bash $BASE_PATH/data_scorer/lqs/scripts/tools/process_cc_lqs.sh $BASE_PATH $DATA_PATH # tokenize cc use fairseq 125M

python3 $BASE_PATH/data_scorer/lqs/tools/get_data_lqs.py # lima dataset to calculate the downstream loss
bash $BASE_PATH/data_scorer/lqs/scripts/tools/process_lima_lqs.sh $BASE_PATH # tokenize lima

bash $BASE_PATH/data_scorer/lqs/scripts/tools/sample_proxy_data.sh $BASE_PATH # sample proxy data from cc
bash $BASE_PATH/data_scorer/lqs/scripts/proxy_data/160M.sh $BASE_PATH # proxy data annotation
bash $BASE_PATH/data_scorer/lqs/scripts/tools/prepare_lqs_data_scorer_train_data.sh $BASE_PATH # prepare sample-score GT pair for data scorer training

bash $BASE_PATH/data_scorer/lqs/scripts/train.sh $BASE_PATH # data scorer training
bash $BASE_PATH/data_scorer/lqs/scripts/tools/convert_tokenization.sh $BASE_PATH # convert tokenized data for data scorer inference 
bash $BASE_PATH/data_scorer/lqs/scripts/infer.sh $BASE_PATH $SCORER_PATH # data scorer infer (data scoring) 

python3 $BASE_PATH/data_scorer/lqs/scripts/tools/scorerd_data_bin2json.sh $BASE_PATH $SCORER_PATH # convert bin to jsonl format