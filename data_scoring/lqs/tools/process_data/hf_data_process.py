import os
import sys

base_path = os.getcwd()
sys.path.insert(0, base_path)

import multiprocessing
import time
import torch
import json
import sys
import numpy as np
import shutil
import argparse
import datasets

from utils import BOS_MODELS, get_tokenizer, load_yaml, add_args
from model_train.data_utils import ChunkedDatasetBuilder, best_fitting_dtype


class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = get_tokenizer(self.args)
        
    def encode(self, line):
        conversations = line[str(self.args.field)]
        conv = ""
        for utt in conversations:
            utt = utt.replace("\n", " ")
            conv += utt + "\n"
        conv = conv.strip()
        
        tokens = Encoder.tokenizer.encode(conv, add_special_tokens=False)
        if self.args.model_type in BOS_MODELS:
            tokens = [Encoder.tokenizer.bos_token_id] + tokens
        tokens = tokens + [Encoder.tokenizer.eos_token_id]
        tokens = tokens[:self.args.max_length]
        
        assert len(tokens) > 1
    
        return line, conv, tokens, len(conv)


def main(args):
        
    output_dir = os.path.join(args.save, args.data_name, f"{args.model_type}-{args.max_length}")
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = get_tokenizer(args)
    dtype = best_fitting_dtype(tokenizer.vocab_size)
    
    dataset = datasets.load_dataset(os.path.join(args.data_path))
    
    for split in dataset:
        builder = ChunkedDatasetBuilder(base_path, output_dir, dtype, split=split, do_shuffle=True)
        
        encoder = Encoder(args)

        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, dataset[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0

        inst_num = 0
        print("#"*10, split, "#"*10)
        
        tokens_lens = []
        
        json_file = open(os.path.join(output_dir, f"{split}.jsonl"), "w")
        
        for lid, (line, conv, tokens, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            
            if lid == 0:
                print("[[conv]]", conv)
                print("[[tokens]]", tokens)
            
            builder.add_np_item(np.array(tokens, dtype=dtype))

            json_file.write(json.dumps({
                "conv": conv,
            }) + "\n")

            tokens_lens.append(len(tokens))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        builder.finalize()

        pool.close()
        json_file.close()
                
        print("Data num", len(tokens_lens))
        print(f"Mean tokens len: {np.mean(tokens_lens)} | Max tokens len: {np.max(tokens_lens)} | Min tokens len: {np.min(tokens_lens)}")


    os.makedirs(os.path.join(output_dir, "dev"), exist_ok=True)
    shutil.copy(os.path.join(output_dir, "train_0.bin"), os.path.join(output_dir, "dev", "data_0.bin"))
    shutil.copy(os.path.join(output_dir, "train_0.idx"), os.path.join(output_dir, "dev", "data_0.idx"))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and tokenize the training data.")
    parser.add_argument("--lqs-process", type=str, required=True, choices=["full_data", "target_data", "proxy_data", "annotation_data", "scorer_data"], default="full_data", help="The content to be downloaded.")
    parser.add_argument("--config-path", type=str, required=True, help="Config path.")

    args = parser.parse_args()
    args = add_args(args, load_yaml(args.config_path), args.lqs_process)
    main(args)