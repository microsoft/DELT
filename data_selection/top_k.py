def select(in_data, method_params):
    in_data = load_jsonl(input_file)
    score_field = method_params["score_field"]
    k = int(max(0, method_params["k"]))

    try:
        out_data = sorted(in_data, key=lambda x: x.get(score_field, float("-inf")), reverse=False)
        if k < len(out_data):
            out_data = out_data[-k:]
    except TypeError as e:
        raise ValueError(f"Error sorting data by field '{score_field}': {e}")

    return out_data
