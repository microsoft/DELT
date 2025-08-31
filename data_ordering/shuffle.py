import random

def order(in_data, args):
    random.seed(args.seed)
    out_data = random.sample(in_data, len(in_data))
    return out_data
