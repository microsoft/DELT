def order(in_data, args):
    score_field = args.score_field

    if args.ascending: 
        # ascending order.
        out_data = sorted(in_data, key=lambda x: x[score_field], reverse=False)
    else:
        # descending order.
        out_data = sorted(in_data, key=lambda x: x[score_field], reverse=True)

    return out_data
