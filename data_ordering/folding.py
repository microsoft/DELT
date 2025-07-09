def order(in_data, args):
    score_field = args.score_field
    layers = args.folding_layer

    # folding order.
    sorted_data = sorted(in_data, key=lambda x: x.get(score_field, float("-inf")), reverse=False)
    
    out_data = list()
    for l in range(layers):
        sub_data = [sorted_data[i] for i in range(len(sorted_data)) if i % layers == l]
        out_data.extend(sub_data)
    return out_data
