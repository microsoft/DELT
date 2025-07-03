




def main():
    args = get_args()
    output_dir = os.path.join(args.save, f"{args.data_name}-t{args.ds_gumbel_temperature}-r{args.ds_ratio}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args)
    tokenizer_cls = get_tokenizer(
        args, model_path=args.data_scorer_tokenizer_path, model_type=args.data_scorer_model_type)

    dtype = best_fitting_dtype(tokenizer.vocab_size)

    with open(os.path.join(args.ds_score_path, "state.json")) as f:
        state = json.load(f)

    scores = []
    for sidx in tqdm(range(state["idx"]), desc="Loading Scores"):
        _scores = torch.load(os.path.join(args.ds_score_path, f"scores_{sidx}.pt"), map_location='cpu')
        scores.append(_scores)
    scores = torch.cat(scores, dim=0)

    if args.ds_gumbel_temperature > 0:
        print('Applying gumble noise ...')
        scores = add_gumbel_noise(scores, args.ds_gumbel_temperature)
    else:
        print('DONT apply gumble noise ...')

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    if args.ds_ratio < 1:
        print('Filtering data by', args.ds_ratio)
        selected_gamma = sorted_scores[:int(args.ds_ratio * len(sorted_scores))]
        selected_indices = sorted_indices[:int(args.ds_ratio * len(sorted_scores))]
    else:
        print('No Filtering ...')
        selected_gamma = sorted_scores
        selected_indices = sorted_indices

    sorted_selected_indices = torch.sort(selected_indices, descending=True).values
    sorted_selected_indices = sorted_selected_indices.tolist()  # reverse order

    data = DistributedMMapIndexedDataset(args.data_dir, "data", do_probe=True)

    sanity_check(args, tokenizer, tokenizer_cls, data)

    selected_data = []
    selected_scores = []

    idx = sorted_selected_indices.pop()
    for i, d in enumerate(data):
        if i == idx:
            selected_data.append(d.astype(int))
            selected_scores.append(scores[i].item())
            if len(sorted_selected_indices) == 0:
                break
            idx = sorted_selected_indices.pop()

    # 保存选择后的数据和分数
    torch.save(selected_data, os.path.join(output_dir, "selected_data.pt"))
    torch.save(selected_scores, os.path.join(output_dir, "selected_scores.pt"))
    print("Data selection completed. Results saved to:", output_dir)


if __name__ == "__main__":
    main()