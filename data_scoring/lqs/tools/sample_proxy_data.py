import os
import argparse
import numpy as np
from tqdm import tqdm

from data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder, best_fitting_dtype
from arguments import add_data_args, add_runtime_args, add_pmp_solver_args
from utils import add_args, load_yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_pmp_solver_args(add_data_args(add_runtime_args(parser)))
    args = parser.parse_args()
    return args


def main():
    
    np.random.seed(args.seed)
    output_dir = os.path.join(args.save, f"{args.data_name}", f"{args.proxy_num}")
    os.makedirs(output_dir, exist_ok=True)
        
    data = DistributedMMapIndexedDataset(args.data_dir, "data", min_state=args.min_state, max_state=args.max_state)
    dtype = data[0].dtype.type
    builder = ChunkedDatasetBuilder(args.base_path, output_dir, dtype)
    
    data_num = len(data)
    
    all_indices = set()
    for _ in tqdm(range(args.proxy_num)):
        idx = np.random.randint(data_num)
        while idx in all_indices:
            idx = np.random.randint(data_num)
        all_indices.add(idx)
    
    all_indices = list(all_indices)
    all_indices = sorted(all_indices)
    print("First 10 indices", list(all_indices)[:10])

    for idx in tqdm(all_indices):
        builder.add_np_item(data[idx])
        
    builder.finalize()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF dataset or model.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data, target_data, proxy_data, annotation_data, scorer_data"], default="full_data", help="The content to be downloaded.")
    parser.add_argument("--config-path", type=str, required=True, help="Input dataset id or model id.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    
    main(args)