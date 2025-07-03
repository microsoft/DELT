import os
import lm_eval

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def eval(model_path, method_params):
    model_args = {"pretrained": model_path, "add_bos_token": method_params["add_bos_token"]}
    results = lm_eval.simple_evaluate( 
        model=method_params["model_format"],
        model_args=model_args,
        tasks=method_params["tasks"],
        device=method_params["device"],
        batch_size=method_params["batch_size"],
        random_seed=method_params["seed"],
        numpy_random_seed=method_params["seed"],
        torch_random_seed=method_params["seed"],
        fewshot_random_seed=method_params["seed"],
        confirm_run_unsafe_code=True,
        #num_fewshot=0,
        #limit=10,
        )
    return results["results"]
