import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import BOS_MODELS, get_tokenizer, load_yaml, add_args
from model_train.data_utils import DistributedMMapIndexedDataset, ChunkedDatasetBuilder, best_fitting_dtype



def normalize(scores):
    scores = scores - np.mean(scores)
    scores = scores / np.std(scores)
    scores = np.clip(scores, -3, 3)
    return scores


def main(args):

    output_dir = os.path.join(args.proxy_save, f"{args.proxy_data_name}-{args.proxy_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = get_tokenizer(args)
    dtype = best_fitting_dtype(tokenizer.vocab_size)

    tokenizer_cls = get_tokenizer(
        args, model_path=args.data_scorer_tokenizer_path, model_type=args.data_scorer_model_type)

    data_bin = DistributedMMapIndexedDataset(args.proxy_data_dir, "data", do_probe=True)
    data = []
    data_num = args.proxy_num if args.proxy_num is not None else len(data_bin)
    for i in tqdm(range(data_num)):
        data.append(data_bin[i].astype(int))

    scores = torch.load(os.path.join(args.proxy_score_path, "grad_gamma.pt"), map_location="cpu").cpu().numpy()
    scores = normalize(scores)
    
    all_data = {
        "valid": (data[:args.proxy_dev_num], scores[:args.proxy_dev_num]),
        "train": (data[args.proxy_dev_num:], scores[args.proxy_dev_num:])
    }

    max_length_no_trunc = 0
    min_length_no_trunc = 1000000
    mean_length = 0

    for split in ["valid", "train"]:
        builder = ChunkedDatasetBuilder(base_path, output_dir, dtype, split=split)
        x, y = all_data[split]
        new_y = []
        for lid, (xx, yy) in enumerate(zip(tqdm(x), y)):
            new_y.append(yy)
            eos_poses = np.where(xx == tokenizer.eos_token_id)[0]
            start = 0
            split_xx = []
            for p in eos_poses:
                split_xx.append(xx[start:p])
                start = p + 1
            split_xx.append(xx[start:])
            tokens = []
            for sxx in split_xx:
                s = tokenizer.decode(sxx, skip_special_tokens=True)
                _tokens = tokenizer_cls.encode(s, add_special_tokens=False)
                tokens.extend(_tokens)
                tokens.append(tokenizer_cls.eos_token_id)
            tokens.pop() # pop the last eos_token_id
            max_length_no_trunc = max(max_length_no_trunc, len(tokens))
            min_length_no_trunc = min(min_length_no_trunc, len(tokens))
            
            if args.data_scorer_model_type in BOS_MODELS:
                tokens = [tokenizer_cls.bos_token_id] + tokens[:args.max_length-1]

            if lid == 0:
                print(tokenizer.decode(xx))
                print(tokens)
                print(tokenizer_cls.decode(tokens))
            assert len(tokens) <= args.max_length
            mean_length += len(tokens)

            builder.add_np_item(np.array(tokens, dtype=dtype))
        
        builder.finalize()

        mean_length /= len(x)
        
        print(f"{split} max_length (before trunc): {max_length_no_trunc}, min_length: {min_length_no_trunc}, mean_length: {mean_length}")
        
        new_y = np.array(new_y)
        plt.plot(np.sort(new_y)[::-1], label="scored")
        baseline = np.ones_like(new_y) / len(new_y)
        plt.plot(baseline, label="baseline")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{split}_scores.png"))
        plt.close()

        with h5py.File(os.path.join(output_dir, f"{split}_scores.hdf5"), "w") as f:
            f.create_dataset("scores", data=new_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare training data for data scorer.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scoring_data"], default="full_data", help="The content to be downloaded.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)

    # args = get_args()
    main(args)
