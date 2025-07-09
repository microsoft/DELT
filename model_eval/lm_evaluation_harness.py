import os
import lm_eval

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def eval(model_path, args):
    model_args = {"pretrained": model_path, "add_bos_token": args.add_bos_token}
    results = lm_eval.simple_evaluate( 
        model=args.model_format,
        model_args=model_args,
        tasks=args.tasks,
        device=args.device,
        batch_size=args.batch_size,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        confirm_run_unsafe_code=True,
        # num_fewshot=0,
        # limit=10,
        )
    return results["results"]
