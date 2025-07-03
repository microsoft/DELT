import time
import os

import random
import numpy as np
import torch
# init_env()
import torch.distributed as dist
import json
from data_scorer.lqs.argments_lqs import get_args

from utils import print_args, initialize
from utils import save_rank

from data_scorer.lqs.annotation import GammaTrainer
from data_scorer.lqs.trainer import DataScorerTrainer


torch.set_num_threads(16)



def init_env(args):
    print('Random Seed: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # if os.environ.get('DETERMINISTIC') is not None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
 
    # be consistent with nanogpt settings
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.cuda.manual_seed_all(args.seed)
    print('Set Random Seed Successful: ', args.seed)

def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)        

    init_env(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.time_stamp = cur_time
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)

        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config["gradient_clipping"] = args.clip_grad
        ds_config["steps_per_print"] = 10000000
        
        if not args.do_train:
            ds_config["zero_optimization"]["stage"] = 0
        
        if not ds_config["fp16"]["enabled"]:
            args.fp32 = True
        
        args.deepspeed_config = None
    else:
        ds_config = None
    
    if args.type == "data_annotation":
        trainer = GammaTrainer(args, device)
    elif args.type == "data_scorer":
        trainer = DataScorerTrainer(args, ds_config, device, args.do_train)
    else:
        raise ValueError(f"Invalid type: {args.type}")    
    
    trainer.train()

    
if __name__ == "__main__":
    main()
