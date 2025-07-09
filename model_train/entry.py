import os
import sys
import deepspeed

base_path = os.getcwd()
sys.path.insert(0, base_path)

import argparse
from pre_trainer import PreTrainer
# import post_trainer import PostTrainer
from utils import load_yaml, init, init_deepspeed, add_args, base_data_suffix, base_model_suffix, base_training_hp_suffix

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation.")
    parser.add_argument("--local_rank", type=int, help="Local rank for deepspeed.", default=0)
    parser.add_argument("--data_path", type=str, required=True, help="The path of data for training.")
    parser.add_argument("--model_path", type=str, required=True, help="The path of model for training.")
    parser.add_argument("--save", type=str, required=True, help="The path for model saving.")
    parser.add_argument("--method", type=str, choices=["pretrain"], default="pretrain",
                        help="Training type: 'pretrain'. Defaults to 'pretrain'.")
    parser.add_argument("--config_path", type=str, default="./model_train/config/pre_train.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()
    parser = deepspeed.add_config_arguments(parser)
    method_params = load_yaml(args.config_path)

    if args.method in ["pretrain"]:
        args = add_args(args, method_params)
        args.save = os.path.join(
            args.save,
            base_data_suffix(args),
            base_model_suffix(args),
            base_training_hp_suffix(args) + (f"-scr" if args.from_scratch else "")
        )

    if args.method == "pretrain":
        init(args)
        ds_config = init_deepspeed(args)
        PreTrainer(args, ds_config).train()
    # elif args.method == "posttrain":
        #PostTrainer(args, ds_config).train()

    # if args.method == "pretrain2":
        # trainer.train(args.data_path, args.model_path, args.save, method_params)
