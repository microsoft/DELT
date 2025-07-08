import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import torch
import argparse

from utils import load_yaml, init, init_deepspeed_infer, add_args, base_data_suffix, base_model_suffix

from data_scoring.lqs.trainer.trainer import DataScorerTrainer


torch.set_num_threads(16)


def main(args):
    torch.backends.cudnn.enabled = False 
    
    init(args)
    ds_config = init_deepspeed_infer(args)
    
    args.save = os.path.join(
        args.save,
        base_data_suffix(args),
        base_model_suffix(args),
    )
    
    if args.type == "data_scorer":
        trainer = DataScorerTrainer(args, ds_config, args.do_train)
    else:
        raise ValueError(f"Invalid type: {args.type}")    
    
    trainer.inference()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample proxy data for annotation.") 
    parser.add_argument("--local_rank", type=int, help="Local rank for deepspeed.", default=0)
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training", "scorer_data_infer"], default="scorer_data_infer", help="The content to be downloaded.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    
    main(args)
