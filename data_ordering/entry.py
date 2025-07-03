import argparse
import shuffle
import sorting
import folding
from ..utils import load_yaml, load_jsonl, write_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data ordering.")
    parser.add_argument("--input_data_path", type=str, help="Path to the input .jsonl file.")
    parser.add_argument("--output_data_path", type=str, help="Path to the output .jsonl file.")
    parser.add_argument("--method", type=str, choices=["shuffle", "sorting", "folding"], default="folding",
                        help="Ordering method: 'shuffle', 'sorting', and 'folding'. Defaults to 'folding'.")
    parser.add_argument("--config", type=str, default="./config/folding.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()

    method_params = load_yaml(args.config)
    in_data = load_jsonl(args.input_data_path)
    if args.method == "shuffle":
        out_data = shuffle.order(in_data, method_params)
    if args.method == "sorting":
        out_data = sorting.order(in_data, method_params)
    if args.method == "folding":
        out_data = folding.order(in_data, method_params)
    write_jsonl(args.output_data_path, out_data)
