# 02 — Setup

## 1. Install Python

You need Python 3.9–3.12. Check what you have:

```bash
python --version
```

If you don't have Python, install it from [python.org](https://www.python.org/downloads/).

## 2. Create a virtual environment

A virtual environment keeps this project's packages separate from anything
else on your machine, so nothing conflicts.

From inside the `ML PROJECT` folder:

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (Git Bash):** `source .venv/Scripts/activate`
- **macOS/Linux:** `source .venv/bin/activate`

You'll know it worked because your terminal prompt will show `(.venv)` at
the start of the line.

## 3. Install the dependencies

```bash
pip install -r requirements.txt
```

This installs:

- **torch** — PyTorch, the deep learning framework everything runs on
- **transformers** — Hugging Face's library for loading pretrained models
- **peft** — the LoRA / adapter library
- **datasets** — for loading and processing training data
- **accelerate** — handles running training on CPU/GPU/multi-GPU transparently
- **trl** — helper utilities for LLM fine-tuning (used for some data collators)
- **bitsandbytes** — enables 4-bit quantization (QLoRA); GPU/CUDA only

If you don't have an NVIDIA GPU, that's fine — the default config in this
project uses a tiny model (`distilgpt2`) that can train on a CPU in a few
minutes. `bitsandbytes` may fail to install on Mac/CPU-only machines; that's
expected and won't stop the rest of the project from working since
`load_in_4bit` is off by default.

## 4. (Optional but recommended) Check if you have a GPU

```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

- If `True`: training will be much faster, and you can try larger models.
- If `False`: you're on CPU. Stick with small models like `distilgpt2` or
  `gpt2` for your first runs.

## 5. Verify the install

```bash
python -c "import transformers, peft, datasets; print('All good:', transformers.__version__, peft.__version__)"
```

If that prints version numbers with no errors, you're ready.

Next: [03_training_guide.md](03_training_guide.md) to run your first
fine-tuning job.
