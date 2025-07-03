def order(in_data, method_params):
    score_field = method_params["score_field"]

    # ascending order.
    out_data = sorted(in_data, key=lambda x: x.get(score_field, float("-inf")), reverse=False)
    return out_data
