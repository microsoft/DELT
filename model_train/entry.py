import argparse
import trainer
from pre_trainer import PreTrainer
#import post_trainer import PostTrainer
from ..utils import load_yaml, init, init_deepspeed, add_args

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation.")
    parser.add_argument("--input_data_path", type=str, required=True, help="The path of data for training.")
    parser.add_argument("--input_model_path", type=str, required=True, help="The path of model for training.")
    parser.add_argument("--output_model_path", type=str, required=True, help="The path for model saving.")
    parser.add_argument("--method", type=str, choices=["pretrain"], default="pretrain",
                        help="Training type: 'pretrain'. Defaults to 'pretrain'.")
    parser.add_argument("--config", type=str, default="./model_train/config/pre_train.yaml", help="Config file for additional parameters (YAML format).")

    args = parser.parse_args()
    method_params = load_yaml(args.config)

    if args.method == "pretrain2":
        trainer.train(args.input_data_path, args.input_model_path, args.output_model_path, method_params)
    if args.method == "pretrain":
        args = add_args(args, method_params)
        init(args)
        ds_config = init_deepspeed(args)
        PreTrainer(args, ds_config).train()
    #if args.method == "posttrain":
        #PostTrainer(args, ds_config).train()
