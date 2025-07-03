import json
import numpy as np
import argparse
from data_utils import DistributedMMapIndexedDataset 
from utils import get_tokenizer 
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .bin files back to a single .jsonl file with scores")
    parser.add_argument("--bin-dir", required=True, help="Directory containing the .bin files")
    parser.add_argument("--score-file", required=True, help="Path to the scores .pt file")
    parser.add_argument("--output-file", required=True, help="Path to save the final .jsonl file")
    parser.add_argument("--tokenizer-path", required=True, help="Path to the tokenizer model")
    parser.add_argument("--model-type", required=True, help="Model type used during processing")
    parser.add_argument("--score-name", default="score_lqs", help="Name of the score field in the output JSONL")
    return parser.parse_args()


def decode_bin_to_jsonl_with_scores(bin_dir, score_file, output_file, tokenizer, score_name):
    # Load dataset
    data = DistributedMMapIndexedDataset(bin_dir, "data")
    if len(data) == 0:
        print("No data found in the specified directory.")
        return

    # Load scores
    scores = torch.load(score_file, map_location="cpu")
    if len(scores) != len(data):
        raise ValueError("The number of scores does not match the number of data entries.")

    print(f"Found {len(data)} entries in the dataset. Decoding to {output_file}...")

    # Open the output .jsonl file
    with open(output_file, "w") as jsonl_file:
        for doc_id, tokens in enumerate(data):

            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)

            score = scores[doc_id].item()

            json_line = {
                "id": doc_id,
                "input": "",
                "text": text,
                score_name: score
            }
            jsonl_file.write(json.dumps(json_line, ensure_ascii=False) + "\n")

            if doc_id % 10000 == 0:
                print(f"Decoded {doc_id}/{len(data)} documents...")

    print(f"Decoding complete. All data saved to {output_file}")


def main():
    args = parse_args()

    tokenizer = get_tokenizer(args, model_path=args.tokenizer_path, model_type=args.model_type)

    decode_bin_to_jsonl_with_scores(
        bin_dir=args.bin_dir,
        score_file=args.score_file,
        output_file=args.output_file,
        tokenizer=tokenizer,
        score_name=args.score_name
    )


if __name__ == "__main__":
    main()