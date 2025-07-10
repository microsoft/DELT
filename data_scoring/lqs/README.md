# Data Scoring based on Learnability-Quality Score (LQS)
This instruction shows the detailed implementation of Learnability-Quality Score (LQS).

## 1.  Dataset and model preparation
Download the [Mistral 160M](https://huggingface.co/Data-Selection/BSL-160M) model. Then run tokenization for CC.
```bash
bash data_scoring/lqs/scripts/prepare_full_dataset.sh $DATA_PATH $CONFIG_PATH

# e.g. bash data_scoring/lqs/scripts/prepare_full_dataset.sh data/cc/1b_original_data.jsonl data_scoring/config/lqs.yaml
```

## 2. Prepare data for downstream loss calculation
Download the [lima](https://huggingface.co/Data-Selection/BSL-160M) dataset for downstream loss calculation. Then run tokenization for lima.
```bash
bash data_scoring/lqs/scripts/prepare_target_dataset.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/prepare_target_dataset.sh data_scoring/config/lqs.yaml
```

## 3. Sample proxy data from CC
```bash
bash data_scoring/lqs/scripts/proxy_data_sampling.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/proxy_data_sampling.sh data_scoring/config/lqs.yaml
```

## 4. Proxy data annotation
```bash
bash data_scoring/lqs/scripts/proxy_data_annotation.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/proxy_data_annotation.sh data_scoring/config/lqs.yaml
```

## 5. Train data scorer
```bash
bash data_scoring/lqs/scripts/train_data_scorer.sh $CONFIG_PATH
# e.g. bash data_scoring/lqs/scripts/train_data_scorer.sh data_scoring/config/lqs.yaml 
```

## 6. Use the data scorer to score examples and convert to .jsonl format
```bash
bash data_scoring/lqs/scripts/infer_data_scorer.sh $CONFIG_PATH $OUTPUT_DATA_PATH
# e.g. bash data_scoring/lqs/scripts/infer_data_scorer.sh data_scoring/config/lqs.yaml data/cc/lqs_scored_data.jsonl
```