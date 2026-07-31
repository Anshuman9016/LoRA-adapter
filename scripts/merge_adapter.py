"""
Merge a trained LoRA adapter into the base model to produce one standalone
model directory — useful when you want to deploy without depending on PEFT
at inference time, or upload a single merged model to the Hub.

Usage:
    python scripts/merge_adapter.py --adapter outputs/lora-run-1 --output outputs/merged-model
"""

import argparse

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    model = PeftModel.from_pretrained(base_model, args.adapter)

    # This bakes the LoRA low-rank update directly into the original weight
    # matrices (W' = W + B @ A), after which the adapter is no longer needed.
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Merged model saved to: {args.output}")


if __name__ == "__main__":
    main()
