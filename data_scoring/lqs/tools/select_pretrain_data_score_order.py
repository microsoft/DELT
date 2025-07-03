import os
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np
import Levenshtein

from data_utils import ChunkedDatasetBuilder, DistributedMMapIndexedDataset, best_fitting_dtype
from utils import get_tokenizer
from arguments import add_runtime_args, add_data_args, add_model_args


def add_additional_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data-scorer-tokenizer-path", type=str, default=None)
    parser.add_argument("--data-scorer-model-type", type=str, default=None)
    parser.add_argument("--ds-score-path", type=str, default=None)
    parser.add_argument("--ds-ratio", type=float, default=None)
    parser.add_argument("--ds-gumbel-temperature", type=float, default=None)
    parser.add_argument("--ascend", type=bool, default=False)
    parser.add_argument("--middle_to_sides", type=bool, default=False)
    parser.add_argument("--token-length-select", type=bool, default=False)
    parser.add_argument("--folding-order", type=int, default=3)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_additional_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    args = parser.parse_args()
    return args


def add_gumbel_noise(scores, T):
    scores = scores / T
    print("Scores before noise:", scores[:10])
    u = np.random.uniform(size=np.shape(scores))
    z = -np.log(-np.log(u))
    z = torch.tensor(z, dtype=torch.float32)
    scores = scores + z
    print("Noise:", z[:10])
    print("Scores after noise:", scores[:10])    

    return scores


def sanity_check(args, tokenizer, tokenizer_cls, data):
    print("#### Sanity check ####")
    for _, _, files in os.walk(args.ds_score_path):
        for filename in files:
            if "check_indices" in filename:
                print(f"Checking {filename}")
                check_indices = torch.load(os.path.join(args.ds_score_path, filename), map_location="cpu")
                check_insts = torch.load(os.path.join(args.ds_score_path, filename.replace("indices", "insts")), map_location="cpu", weights_only=False)
                for idx, tokens in zip(tqdm(check_indices), check_insts):
                    s = tokenizer.decode(data[idx].astype(int))[:100]
                    s_cls = tokenizer_cls.decode(tokens)[:100]
                    r = Levenshtein.ratio(s, s_cls)
                    if r < 0.9:
                        print(s)
                        print("\n\n\n")
                        print(s_cls)
                    assert r > 0.9, "The documents from different tokenizer should be the same"
    print("#### Sanity check Pass ####")



def folding_sorted_multi_round(selected_scores, selected_data, layers=3, ascending=True):
    if ascending:   
        print('Ascending ...')
    else:
        print('Descending ...')
    scores_tensor = torch.tensor(selected_scores)
    sorted_indices = torch.argsort(scores_tensor, descending=not ascending)
    
    sorted_data = [selected_data[i] for i in sorted_indices]
    
    rounds = [[] for _ in range(layers)]
    
    for idx, item in enumerate(sorted_data):
        rounds[idx % layers].append(item)
    
    final_result = []
    for round_data in rounds:
        final_result.extend(round_data)
    
    return final_result


def main():
    args = get_args()
    output_dir = os.path.join(args.save, f"{args.data_name}-t{args.ds_gumbel_temperature}-r{args.ds_ratio}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args)
    tokenizer_cls = get_tokenizer(
        args, model_path=args.data_scorer_tokenizer_path, model_type=args.data_scorer_model_type)

    dtype = best_fitting_dtype(tokenizer.vocab_size)

    with open(os.path.join(args.ds_score_path, "state.json")) as f:
        state = json.load(f)

    scores = []
    for sidx in tqdm(range(state["idx"]), desc="Loading Scores"):
        _scores = torch.load(os.path.join(args.ds_score_path, f"scores_{sidx}.pt"), map_location='cpu')
        scores.append(_scores)
    scores = torch.cat(scores, dim=0)

    if args.ds_gumbel_temperature > 0:
        print('Applying gumble noise ...')
        scores = add_gumbel_noise(scores, args.ds_gumbel_temperature)
    else:
        print('DONT apply gumble noise ...')


    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    if args.ds_ratio < 1:
        print('Filtering data by', args.ds_ratio)
        selected_gamma = sorted_scores[:int(args.ds_ratio * len(sorted_scores))]
        selected_indices = sorted_indices[:int(args.ds_ratio * len(sorted_scores))]
    else:
        print('No Filtering ...')
        selected_gamma = sorted_scores
        selected_indices = sorted_indices

    sorted_selected_indices = torch.sort(selected_indices, descending=True).values
    sorted_selected_indices = sorted_selected_indices.tolist() # reverse order

    data = DistributedMMapIndexedDataset(args.data_dir, "data", do_probe=True)

    sanity_check(args, tokenizer, tokenizer_cls, data)

    builder = ChunkedDatasetBuilder(args.base_path, output_dir, dtype, do_shuffle=False)

    selected_num = 0
    tot = min(len(data), len(selected_gamma))

    print("Selected Indices Num:", len(sorted_selected_indices))

    selected_data = []
    selected_scores = []

    idx = sorted_selected_indices.pop()
    for i, d in enumerate(data):
        if i == idx:
            selected_data.append(d.astype(int)) 
            selected_scores.append(scores[i].item())
            if len(sorted_selected_indices) == 0:
                break
            idx = sorted_selected_indices.pop()

    if args.middle_to_sides:
        print('Applying Middle-to-Sides Order ...')
        # Sort the data in ascending order to get a baseline sorted list
        sorted_pairs = sorted(zip(selected_scores, selected_data), key=lambda pair: pair[0])
        
        # Use two pointers to construct the middle-to-sides order
        middle_index = len(sorted_pairs) // 2 
        left_pointer = middle_index - 1      
        right_pointer = middle_index + 1     

        # Start with the middle element
        sorted_selected_data = [sorted_pairs[middle_index][1]]

        # Alternate between left and right pointers to build the list
        while left_pointer >= 0 or right_pointer < len(sorted_pairs):
            if left_pointer >= 0:
                sorted_selected_data.append(sorted_pairs[left_pointer][1])
                left_pointer -= 1
            if right_pointer < len(sorted_pairs):
                sorted_selected_data.append(sorted_pairs[right_pointer][1])
                right_pointer += 1

    elif args.token_length_select:
        print('Selecting by Token Length ...')
        selected_scores = []
        selected_scores = [len(d) for d in selected_data]

        sorted_selected_data = [
            x for _, x in sorted(zip(selected_scores, selected_data), key=lambda pair: pair[0], reverse=False)
        ]
    elif args.folding_order > 1:
        print('Selecting by Folding Order ...')
        sorted_selected_data = folding_sorted_multi_round(selected_scores, selected_data, layers=args.folding_order, ascending=args.ascend)

    else:
        if args.ascend:
            print('Applying Ascend Order ...')
            sorted_selected_data = [
                x for _, x in sorted(zip(selected_scores, selected_data), key=lambda pair: pair[0], reverse=False)
            ]
        else:
            print('Applying Descend Order ...')
            sorted_selected_data = [
                x for _, x in sorted(zip(selected_scores, selected_data), key=lambda pair: pair[0], reverse=True)
            ]

    pbar = tqdm(total=len(sorted_selected_data), desc="Train")
    for d in sorted_selected_data:
        if selected_num == 0:
            print("#### Example Instance ####")
            print(d)
            print(tokenizer.decode(d))
            print("#### Example Instance ####")
        elif selected_num == len(sorted_selected_data)-1:
            print("#### Example Instance ####")
            print(tokenizer.decode(d))
            print("#### Example Instance ####")
        
        builder.add_np_item(d) 
        selected_num += 1
        pbar.update(1)

    builder.finalize()
    print(f"Selected Data Num: {selected_num}")


if __name__ == "__main__":
    main()