import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import argparse
import numpy as np
from tqdm import tqdm

from model_train.data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder
from utils import add_args, load_yaml



def main(args):
    
    np.random.seed(args.seed)
    output_dir = os.path.join(args.save, f"{args.data_name}", f"{args.proxy_num}")
    os.makedirs(output_dir, exist_ok=True)
        
    data = DistributedMMapIndexedDataset(args.data_path, "data", min_state=args.min_state, max_state=args.max_state)
    dtype = data[0].dtype.type
    builder = ChunkedDatasetBuilder(base_path, output_dir, dtype)
    
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
    parser = argparse.ArgumentParser(description="Sample proxy data for annotation.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data"], default="full_data", help="The content to be downloaded.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    
    main(args)