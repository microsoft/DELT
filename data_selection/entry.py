import argparse
import top_k
import top_r
import threshold
from ..utils import load_yaml, load_jsonl, write_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data selection.")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input .jsonl file.")
    parser.add_argument("--output_data_path", type=str, required=True, help="Path to the output .jsonl file.")
    parser.add_argument("--method", type=str, choices=["top-k", "top-r", "threshold"], default="top-r",
                        help="Selection method: 'top-k', 'top-r', and 'threshold'. Defaults to 'top-r'.")
    parser.add_argument("--config", type=str, default="./config/top-r.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()

    method_params = load_yaml(args.config)
    in_data = load_jsonl(args.input_data_path)
    if args.method == "top-k":
        out_data = top_k.select(in_data, method_params)
    if args.method == "top-r":
        out_data = top_r.select(in_data, method_params)
    if args.method == "threshold":
        out_data = threshold.select(in_data, method_params)
    write_jsonl(args.output_data_path, out_data)
