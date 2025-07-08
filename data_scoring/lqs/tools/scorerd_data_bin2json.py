import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import json
import torch
import argparse
from tqdm import tqdm
from model_train.data_utils import DistributedMMapIndexedDataset 
from utils import add_args, load_yaml, get_tokenizer, load_yaml, add_args


def load_scores(score_dir):
    state_path = os.path.join(score_dir, "state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"state.json not found in {score_dir}")

    with open(state_path, "r") as f:
        state = json.load(f)

    scores = []
    for sidx in tqdm(range(state["idx"]), desc="Loading Scores"):
        score_path = os.path.join(score_dir, f"scores_{sidx}.pt")
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"Score file {score_path} not found")
        _scores = torch.load(score_path, map_location="cpu")
        scores.append(_scores)

    scores = torch.cat(scores, dim=0)
    return scores


def decode_bin_to_jsonl_with_scores(bin_data_path, pt_score_path, jsonl_output_path, tokenizer, save_score_name):
    data = DistributedMMapIndexedDataset(bin_data_path, "data")
    if len(data) == 0:
        print("No data found in the specified directory.")
        return

    scores = load_scores(pt_score_path)
    if len(scores) != len(data):
        raise ValueError("The number of scores does not match the number of data entries.")

    print(f"Found {len(data)} entries in the dataset. Decoding to {jsonl_output_path}...")

    with open(jsonl_output_path, "w") as jsonl_file:
        for doc_id, tokens in enumerate(data):
           
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

            score = scores[doc_id].item()

            json_line = {
                "id": doc_id,
                "input": "",
                "text": text,
                save_score_name: score
            }
            jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

            if doc_id % 10000 == 0:
                print(f"Decoded {doc_id}/{len(data)} documents...")

    print(f"Decoding complete. All data saved to {save_score_name}")


def main(args):
    tokenizer = get_tokenizer(args, model_path=args.jsonl_tokenizer_path, model_type=args.jsonl_model_type)

    decode_bin_to_jsonl_with_scores(
        bin_data_path=args.bin_data_path,
        pt_score_path=args.pt_score_path,
        jsonl_output_path=args.jsonl_output_path,
        tokenizer=tokenizer,
        save_score_name=args.save_score_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample proxy data for annotation.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="scorer_data_infer", help="The content to be downloaded.")
    parser.add_argument("--jsonl-output-path", type=str, required=True, help="Output path.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    
    main(args)