"""
Fine-tune a causal language model with a LoRA adapter (via PEFT).

Usage:
    python scripts/train.py --config configs/lora_config.yaml

Everything tunable lives in the YAML config, not in this file, so you can
run experiments by editing configs/lora_config.yaml (or copying it) instead
of touching code.
"""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Most causal LMs (GPT-2 family especially) ship without a pad token.
    # We reuse the EOS token as padding since we never attend to pad tokens
    # for loss computation anyway.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset, tokenizer, text_field: str, max_length: int):
    def _tokenize(batch):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    tokenizer = build_tokenizer(model_cfg["name"])

    print(f"Loading base model: {model_cfg['name']}")
    load_kwargs = {}
    if model_cfg.get("load_in_4bit"):
        # QLoRA-style loading: the frozen base weights are stored in 4-bit,
        # while the LoRA adapters we train stay in full precision.
        load_kwargs.update(
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **load_kwargs)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading and tokenizing data...")
    raw = load_dataset(
        "json",
        data_files={"train": data_cfg["train_file"], "eval": data_cfg["eval_file"]},
    )
    train_ds = tokenize_dataset(
        raw["train"], tokenizer, data_cfg["text_field"], data_cfg["max_seq_length"]
    )
    eval_ds = tokenize_dataset(
        raw["eval"], tokenizer, data_cfg["text_field"], data_cfg["max_seq_length"]
    )

    # mlm=False means "predict the next token", i.e. standard causal LM
    # training, as opposed to masked-language-modeling (BERT-style).
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args_t = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        seed=train_cfg["seed"],
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=args_t,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    # Saves ONLY the small adapter weights (a few MB), not the full base model.
    model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"\nDone. LoRA adapter saved to: {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
