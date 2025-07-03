import random

def order(in_data, method_params):
    #random.shuffle(in_data)
    out_data = random.sample(in_data, len(in_data))
    return out_data
