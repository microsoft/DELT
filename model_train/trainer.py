import os
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from accelerate.utils import DistributedType

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomTrainer(Trainer):
    def compute_loss(self, model_a, inputs_a, return_outputs=False):
        outputs = model_a(**inputs_a, labels=inputs_a["input_ids"])
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def train(input_data_path, model_name, output_model_path, method_params):
    # device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")
    print(f"cpu count: {os.cpu_count()}")

    # tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # model.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    model.to(device)
    model.train()

    # data.
    dataset = load_dataset("json", data_files={"train": [input_data_path]})
    tokenized_datasets = dataset.map(lambda examples: tokenizer(examples[method_params["data_field"]], padding="max_length", truncation=True, max_length=method_params["max_length"]), batched=True, num_proc=os.cpu_count(), load_from_cache_file=False)
    train_dataset = tokenized_datasets["train"]#.select(range(100))
    if method_params["shuffle_data"]:
        train_dataset = train_dataset.shuffle(seed=method_params["seed"])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=method_params["batch_size"])

    training_args = TrainingArguments(
        output_dir=output_model_path, 
        overwrite_output_dir=True,
        learning_rate=method_params["lr"],
        num_train_epochs=method_params["num_epochs"],
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=method_params["batch_size"],
        #save_strategy="epoch",
        save_strategy="steps",
        save_steps=10000,
        logging_steps=100,
        report_to="none",
        fp16=True,
        save_total_limit=10,
        deepspeed=method_params["deepspeed_config"],
        gradient_accumulation_steps=method_params["gradient_accumulation_steps"],
    )
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #tokenizer=tokenizer,
        #data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
