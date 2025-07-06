# Data Efficacy for Language Model Training

<p align="center">
 <img src="https://img.shields.io/badge/Task-LM_Efficacy-orange" alt="Task" /> 
 <img src="https://img.shields.io/badge/Paper-Published-green" alt="Paper" /> 
 <img src="https://img.shields.io/badge/License-MIT-blue" alt="License" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.21545"><b>[📜 Paper]</b></a> •
  <a href="https://github.com/microsoft/DELT"><b>[🐱 GitHub Code]</b></a>
</p>

<figure>
  <img src="./figures/fig1_teaser.jpg" alt="Figure 1" style="width: 95%;">
  <figcaption style="color: gray;">
    <div><small><em>Figure 1. Average result across 8 benchmarks for different methods. High performance at the same selection ratio indicates high efficacy, while achieving similar performance with a smaller selection ratio demonstrates high efficiency. Our method excels in both efficacy and efficiency.</em></small></div>
  </figcaption>
</figure>

## 🌟 Introduction
Data is fundamental to the training of language models (LM). Recent research has been dedicated to data efficiency, which aims to maximize performance by selecting a minimal or optimal subset of training data. Techniques such as data filtering, sampling, and selection play a crucial role in this area. To complement it, we define Data Efficacy, which focuses on maximizing performance by optimizing the organization of training data and remains relatively underexplored. This work introduces a general paradigm, DELT, for considering data efficacy in LM training, which highlights the significance of training data organization. DELT comprises three components: Data Scoring, Data Selection, and Data Ordering.

<figure>
  <img src="./figures/data_efficacy_paradigm.png" alt="Figure 2" style="width: 95%;">
  <figcaption style="color: gray;">
    <div align="center"><small><em>Figure 2. DELT paradigm.</em></small></div>
  </figcaption>
</figure>

<br>

For data scoring, we design **Learnability-Quality Scoring (LQS)** method, which considers both the learnability and quality of each data sample from the gradient consistency perspective.

<figure>
  <img src="./figures/fig2_score.jpg" alt="Figure 3" style="width: 95%;">
  <figcaption style="color: gray;">
    <div align="center"><small><em>Figure 3. Learnability-Quality Scoring (LQS).</em></small></div>
  </figcaption>
</figure>

<br>

For data ordering, we devise **Folding Ordering (FO)** method, which addresses issues such as model forgetting and data distribution bias.

<figure>
  <img src="./figures/fig3_order.jpg" alt="Figure 4" style="width: 95%;">
  <figcaption style="color: gray; text-align: center;">
    <div align="center"><small><em>Figure 4. Folding Ordering (FO).</em></small></div>
  </figcaption>
</figure>


## 📢 News and Updates

Done
- [x] 2025/06/28: 💥[Arxiv paper](https://arxiv.org/abs/2506.21545) released.
- [x] 2025/06/28: 💥We have released the DELT code for pre-training on general data.

TBD
- [ ] Release the trained data scorer checkpoint .
- [ ] Release the DELT code for post-training on specific domain data.


## ⚙️ Environment Installation

```bash
conda create -n data_efficacy python=3.10
conda activate data_efficacy
bash install.sh
```

## 💾 Preparation.

<details open>
<summary>Dataset Preparation</summary>

```bash
python utils.py --content dataset --id $HF_DATASET_ID --save_dir $OUTPUT_DATA_PATH

# e.g. python utils.py --content dataset --data-name common_crawl --id togethercomputer/RedPajama-Data-1T --save-dir data/cc_original_data.jsonl --split-name train --sample-size 100000
# You could also replace it with your own dataset under jsonl format. 

# python utils.py --content dataset --data-name plain_text --id togethercomputer/RedPajama-Data-1T-Sample --save-dir data/cc_original_data.jsonl --split-name train --sample-size 100000
```
</details>

<details open>
<summary>Model Preparation</summary>

```bash
python utils.py --content=model --id $HF_MODEL_ID --save_dir $OUTPUT_MODEL_PATH

# e.g. python utils.py --content=model --id=Data-Selection/BSL-160M --save_dir=model/input_model
# You could also replace it with your own model under hf format.
```
</details>

## ⏩ Quick Start.

<details open>
<summary>Data Scoring</summary>

Existing scoring method: KenLM (`kenlm`), PDS (`pds`), and **Learnability-Quality Score** (`lqs`).
For more details about LQS, please refer to [this guide](./data_scoring/lqs/README_LQS.md).

```bash
bash data_scoring/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $METHOD $CONFIG_PATH

# e.g. bash data_scoring/entry.sh data/original_data.jsonl data/scored_data.jsonl lqs data_scoring/config/lqs.yaml

#####
# e.g. bash data_scoring/entry.sh data/cc_original_data.jsonl data/scored_data.jsonl lqs data_scoring/config/lqs.yaml
# e.g. bash data_scoring/lqs/entry_test.sh data/cc_original_data.jsonl lqs data_scoring/config/lqs.yaml
```
</details>

<details open>
<summary>Data Selection</summary>

Existing selection method: **Top-R** (`top-r`), Top-K (`top-k`), and Threshold (`threshold`).

```bash
bash data_selection/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $METHOD $CONFIG_PATH

# e.g. bash data_selection/entry.sh data/scored_data.jsonl data/selected_data.jsonl top-r data_selection/config/top-r.yaml
```
</details>

<details open>
<summary>Data Ordering</summary>

Existing ordering method: Shuffle (`shuffle`), Sorting (`sorting`), and **Folding Ordering (FO)** (`folding`).

```bash
bash data_ordering/entry.sh $INPUT_DATA_PATH $OUTPUT_DATA_PATH $METHOD $CONFIG_PATH

# e.g. bash data_ordering/entry.sh data/selected_data.jsonl data/ordered_data.jsonl folding data_ordering/config/folding.yaml
```
</details>


<details open>
<summary>Model Training</summary>

```bash
bash model_train/entry.sh $INPUT_DATA_PATH $INPUT_MODEL_PATH $OUTPUT_MODEL_PATH $METHOD $CONFIG_PATH

# e.g. bash model_train/entry.sh data/ordered_data.jsonl model/input_model model/output_model pretrain model_train/config/pre_train.yaml
```
</details>


<details open>
<summary>Model Evaluation</summary>

```bash
bash model_eval/entry.sh $INPUT_MODEL_PATH $OUTPUT_RESULT_PATH $METHOD $CONFIG_PATH

# e.g. bash model_eval/entry.sh model/output_model model/result.yaml lm_evaluation_harness model_eval/config/general.yaml
```
</details>


## 🔗 Citation
```
@article{dai2025data,
  title={Data Efficacy for Language Model Training},
  author={Yalun Dai and Yangyu Huang and Xin Zhang and Wenshan Wu and Chong Li and Wenhui Lu and Shijie Cao and Li Dong and Scarlett Li},
  journal={arXiv preprint arXiv:2506.21545},
  year={2025}
}
```

## 👀 License
This repository is licensed under the [MIT](https://github.com/microsoft/DELT/blob/main/LICENSE) License.

