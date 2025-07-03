# Data Scoring based on Learnability-Quality Score (LQS)
This instruction shows the detailed implementation of Learnability-Quality Score (LQS).

## 0. Data Scorer Preparation
#### Dataset and model preparation
1. First download dataset [Redpajama CC](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T). Then run tokenization.
```bash
bash $BASE_PATH/data_scorer/lqs/scripts/tools/process_cc_lqs.sh $BASE_PATH $DATA_PATH

# Convert tokenized data for data scorer inference (if the data scorer and pre-training use different tokenizer).
bash $BASE_PATH/data_scorer/lqs/scripts/tools/convert_tokenization.sh $BASE_PATH
```

2. Get checkpoints for initialization
```bash
python3 $BASE_PATH/data_scorer/lqs/tools/get_checkpoints_lqs.py # Base model for annotation (Mistral 160M) and base model of the data scorer (fairseq 125M)
```

#### 0.1 Data scorer initialization
Pre-train the small model for the initialization of LM for data annotation (also is the 160M conventional baseline) or download the [checkpoint](https://huggingface.co/Data-Selection/BSL-160M) instead.
```bash
bash $BASE_PATH/scripts/pretrain/160M_bsl.sh $BASE_PATH
```

#### 0.2 Prepare data for downstream loss calculation
First download [lima](https://huggingface.co/datasets/GAIR/lima). Then run tokenization.
```bash
# python3 $BASE_PATH/data_scorer/lqs/tools/get_data_lqs.py
bash $BASE_PATH/data_scorer/lqs/scripts/tools/process_lima_lqs.sh $BASE_PATH
```
#### 0.3 Sample proxy data from CC
```bash
bash $BASE_PATH/data_scorer/lqs/scripts/tools/sample_proxy_data.sh $BASE_PATH
```

#### 0.4 Proxy data annotation
```bash
bash $BASE_PATH/data_scorer/lqs/scripts/proxy_data/160M.sh $BASE_PATH
```

## 1. Train data scorer
#### 1.1 Prepare data for data scorer training
```bash
bash $BASE_PATH/data_scorer/lqs/scripts/tools/prepare_lqs_data_scorer_train_data.sh $BASE_PATH
```

#### 1.2 Train data scorer
```bash
bash $BASE_PATH/data_scorer/lqs/scripts/train.sh $BASE_PATH
```

#### 1.3 Use the data scorer to score examples
```bash
bash $BASE_PATH/data_scorer/lqs/scripts/infer.sh $BASE_PATH $SCORER_PATH
```

#### 1.4 Convert bin to jsonl format
```bash
python3 $BASE_PATH/data_scorer/lqs/scripts/tools/scorerd_data_bin2json.sh $BASE_PATH $SCORER_PATH
```