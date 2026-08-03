# Hands-On Model Fine-Tuning with PyTorch and LoRA/PEFT

A complete, runnable, beginner-friendly project for fine-tuning a language
model efficiently using **LoRA** (Low-Rank Adaptation) through Hugging
Face's **PEFT** library — built so you can go from "what is fine-tuning?"
to a working fine-tuned model in one sitting.

**Assumed background: none.** If you don't know what a "parameter" or
"tokenizer" is, start with [docs/01_concepts.md](docs/01_concepts.md) —
everything is explained from scratch before you write or run any code.

## What you'll end up with

- A working understanding of what LoRA/PEFT actually do and why they exist
- A trained LoRA adapter (a small file, a few MB) fine-tuned on sample data
- The ability to swap in your own data and fine-tune your own model
- Scripts to run inference with your fine-tuned model, and to merge the
  adapter into a standalone model

## Project structure

```
ML PROJECT/
├── README.md                  <- you are here
├── requirements.txt            <- Python packages needed
├── configs/
│   └── lora_config.yaml        <- every training setting lives here
├── data/
│   ├── train.jsonl              <- sample training examples
│   └── eval.jsonl               <- sample held-out evaluation examples
├── scripts/
│   ├── train.py                 <- runs LoRA fine-tuning
│   ├── inference.py              <- generates text with your trained adapter
│   └── merge_adapter.py           <- merges adapter into a standalone model
├── docs/
│   ├── 01_concepts.md            <- start here: what is LoRA/PEFT, from zero
│   ├── 02_setup.md                <- installing Python packages, checking for a GPU
│   ├── 03_training_guide.md        <- step-by-step: run your first training job
│   └── 04_troubleshooting.md        <- common errors and fixes
└── outputs/                    <- created automatically; trained adapters land here
```

## Learning path (read in this order)

1. **[docs/01_concepts.md](docs/01_concepts.md)** — What is a language
   model? What is fine-tuning? What problem does LoRA solve? What is rank?
   What is PEFT? Read this first even if you're impatient to run code — the
   rest will make far more sense.
2. **[docs/02_setup.md](docs/02_setup.md)** — Install Python, create a
   virtual environment, install dependencies, check whether you have a GPU.
3. **[docs/03_training_guide.md](docs/03_training_guide.md)** — Walks
   through the sample data format, the config file, running training,
   running inference, and merging the adapter — explaining what each step
   and each printed number means.
4. **[docs/04_troubleshooting.md](docs/04_troubleshooting.md)** — Reach for
   this when something errors or the output looks wrong.

## Quickstart (once you've read the docs above)

```bash
# 1. Set up environment (see docs/02_setup.md for details)
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; see docs/02_setup.md for other shells
pip install -r requirements.txt

# 2. Train a LoRA adapter on the included sample data
python scripts/train.py --config configs/lora_config.yaml

# 3. Try it out
python scripts/inference.py --adapter outputs/lora-run-1 --prompt "What is a LoRA adapter?"

# 4. (Optional) merge into one standalone model
python scripts/merge_adapter.py --adapter outputs/lora-run-1 --output outputs/merged-model
```

The default config uses `distilgpt2`, a small model, so this runs in a few
minutes even without a GPU — it's meant as a working proof-of-concept you
can then scale up (bigger model, your own data, more epochs) once you trust
the pipeline.

## Using your own data

Replace `data/train.jsonl` / `data/eval.jsonl` with your own examples,
keeping one JSON object per line with a `"text"` field. See
[docs/03_training_guide.md](docs/03_training_guide.md#step-2--use-your-own-data-optional)
for format details and tips on dataset size.

## Using a bigger / different model

Edit `configs/lora_config.yaml`:
- Change `model.name` to any causal LM on the Hugging Face Hub (e.g.
  `meta-llama/Llama-3.2-1B`, `microsoft/phi-2`, `mistralai/Mistral-7B-v0.1`).
- Update `lora.target_modules` to match that architecture — see the lookup
  table in [docs/03_training_guide.md](docs/03_training_guide.md#finding-the-right-target_modules-for-your-model).
- If the model is large and you have a CUDA GPU, set `model.load_in_4bit: true`
  to enable QLoRA (4-bit quantized base model + LoRA adapters) — see
  [docs/01_concepts.md](docs/01_concepts.md#5-what-is-qlora--4-bit-loading).

## Why LoRA/PEFT instead of full fine-tuning?

Full fine-tuning updates every weight in the model — for a multi-billion
parameter model that means huge GPU memory requirements and multi-gigabyte
checkpoints. LoRA freezes the original weights and trains only small
"adapter" matrices alongside them (often <1% of total parameters),
producing a tiny, portable file and letting you fine-tune much larger
models on much smaller hardware. Full explanation with diagrams-in-text in
[docs/01_concepts.md](docs/01_concepts.md).
