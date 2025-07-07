# Data Scoring based on Learnability-Quality Score (LQS)
This instruction shows the detailed implementation of Learnability-Quality Score (LQS).

## 0. Data Scorer Preparation
#### Dataset and model preparation
1. First download dataset [Redpajama CC](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T). Then run tokenization.
```bash
bash data_scoring/lqs/scripts/prepare_full_dataset.sh $DATA_PATH $CONFIG_PATH

# e.g. bash data_scoring/lqs/scripts/prepare_full_dataset.sh data/cc data_scoring/config/lqs.yaml
```

2. Get checkpoints for initialization
```bash
bash data_scoring/lqs/scripts/prepare_target_dataset.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/prepare_target_dataset.sh data_scoring/config/lqs.yaml
```

```bash
bash data_scoring/lqs/scripts/proxy_data_sampling.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/proxy_data_sampling.sh data_scoring/config/lqs.yaml
```

```bash
bash data_scoring/lqs/scripts/proxy_data_annotation.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/proxy_data_annotation.sh data_scoring/config/lqs.yaml
```

```bash
bash data_scoring/lqs/scripts/proxy_data_annotation.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/proxy_data_annotation.sh data_scoring/config/lqs.yaml 
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