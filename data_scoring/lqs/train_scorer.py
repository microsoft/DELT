import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import torch
import deepspeed
import argparse

from utils import load_yaml, init, init_deepspeed, add_args
from data_scoring.lqs.annotation.trainer import GammaTrainer
from data_scoring.lqs.trainer.trainer import DataScorerTrainer

torch.set_num_threads(16)


def main(args):
    torch.backends.cudnn.enabled = False
    
    init(args)
    ds_config = init_deepspeed(args)

    
    if args.type == "annotation_data":
        trainer = GammaTrainer(args)
    elif args.type == "scoring_data":
        trainer = DataScorerTrainer(args, ds_config, args.do_train)
    else:
        raise ValueError(f"Invalid type: {args.type}")    
    
    trainer.train()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample proxy data for annotation.") 
    parser.add_argument("--local_rank", type=int, help="Local rank for deepspeed.", default=0)
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scoring_data"], default="full_data", help="The content to be downloaded.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    
    main(args)
