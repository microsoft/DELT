import argparse
from utils import load_yaml, add_args, download_model, download_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF dataset or model.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data, target_data, proxy_data, annotation_data, scorer_data"], default="full_data", help="The content to be downloaded.")
    parser.add_argument("--content", type=str, required=True, help="Input dataset id or model id.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)

    if args.content == "model":
        download_model(args.hf_model_id, args.output_model_path)

    if args.content == "dataset":
        download_data(args.id, args.save_dir, args.split_name, args.sample_size)