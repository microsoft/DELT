import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import torch
import deepspeed
import argparse

from utils import load_yaml, init, init_deepspeed, add_args, base_data_suffix, base_model_suffix, base_training_hp_suffix
from data_scoring.lqs.annotation.trainer import GammaTrainer
from data_scoring.lqs.trainer.trainer import DataScorerTrainer

torch.set_num_threads(16)

def main(args):
    torch.backends.cudnn.enabled = False
    
    init(args)
    ds_config = init_deepspeed(args)

    if args.type in ["annotation_data"]:
        save_path = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            f"{args.optimizer_name}-" + base_training_hp_suffix(args) + f"-ct{args.compute_ct_interval}" 
        )
        args.save = save_path
    elif args.type in ["scoring_data"]:
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
        )
        if args.do_train:
            args.save = os.path.join(
                args.save,
                base_training_hp_suffix(args),
                f"{args.data_scorer_encoding}" + ("-bias" if args.data_scorer_bias else "") +
                f"-{args.data_scorer_head_type}"
            )

    os.makedirs(args.save, exist_ok=True)
    
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
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data_training"], default="full_data", help="The content to be downloaded.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    
    main(args)
