"""
Load a base model plus a trained LoRA adapter and generate text.

Usage:
    python scripts/inference.py --adapter outputs/lora-run-1 --prompt "What is a LoRA adapter?"
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=None,
                        help="Override the base model name. Defaults to the one stored in the adapter config.")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to the folder produced by scripts/train.py")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # PeftModel.from_pretrained reads the base model name from the adapter's
    # own config unless we override it, so we don't need to know it here.
    from peft import PeftConfig

    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_model_name = args.base_model or peft_config.base_model_name_or_path

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    print(f"Applying LoRA adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    prompt = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n--- Generated ---\n")
    print(text)


if __name__ == "__main__":
    main()
