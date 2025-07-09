import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import argparse
import lm_evaluation_harness
from utils import load_yaml, write_yaml, add_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation.")
    parser.add_argument("--input_model_path", type=str, required=True, help="The path of model to be evaluated.")
    parser.add_argument("--output_result_path", type=str, required=True, help="The path of result.")
    parser.add_argument("--method", type=str, choices=["lm_evaluation_harness"], default="lm_evaluation_harness",
                        help="Evaluation method: 'lm_evaluation_harness'. Defaults to 'lm_evaluation_harness'.")
    parser.add_argument("--config_path", type=str, default="./config/general.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path))

    if args.method == "lm_evaluation_harness":
        out_result = lm_evaluation_harness.eval(args.input_model_path, args)
    write_yaml(args.output_result_path, out_result)
