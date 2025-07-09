import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)


import argparse
import shuffle
import sorting
import folding
from utils import load_yaml, load_jsonl, add_args, write_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data ordering.")
    parser.add_argument("--input_data_path", type=str, help="Path to the input .jsonl file.")
    parser.add_argument("--output_data_path", type=str, help="Path to the output .jsonl file.")
    parser.add_argument("--method", type=str, choices=["shuffle", "sorting", "folding"], default="folding",
                        help="Ordering method: 'shuffle', 'sorting', and 'folding'. Defaults to 'folding'.")
    parser.add_argument("--config_path", type=str, default="./config/folding.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()

    args = add_args(args, load_yaml(args.config_path))

    print(f"  Arguments received:")
    print(f"  Input data path: {args.input_data_path}")
    print(f"  Selection method: {args.method}")
    print(f"  Score field: {args.score_field}")

    in_data = load_jsonl(args.input_data_path)
    if args.method == "shuffle":
        out_data = shuffle.order(in_data, args)
        print(f"  Random seed: {args.seed}")

    if args.method == "sorting":
        out_data = sorting.order(in_data, args)
        print(f"  Ascending: {args.ascending}")

    if args.method == "folding":
        out_data = folding.order(in_data, args)
        print(f"  Folding layer: {args.folding_layer}")


    write_jsonl(args.output_data_path, out_data)
