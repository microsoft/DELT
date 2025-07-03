def select(in_data, method_params):
    score_field = method_params["score_field"]
    threshold = method_params["threshold"]

    out_data = list()
    for x in in_data:
        if x.get(score_field, float("-inf")) > threshold:
            out_data.append(line)

    return out_data
