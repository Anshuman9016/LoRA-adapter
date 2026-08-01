# 04 — Troubleshooting

## "CUDA out of memory"

Your GPU ran out of VRAM. In order of preference:

1. Lower `training.per_device_train_batch_size` in the config (try 2, then 1).
2. Raise `training.gradient_accumulation_steps` to compensate (keeps the
   *effective* batch size similar while using less memory per step —
   effective batch size = `per_device_train_batch_size * gradient_accumulation_steps`).
3. Lower `data.max_seq_length` (shorter sequences use less memory).
4. Set `model.load_in_4bit: true` in the config (requires `bitsandbytes`
   and a CUDA GPU) — this is QLoRA.
5. Use a smaller `model.name`.

## Training runs but loss doesn't go down

- Check `learning_rate` — LoRA often wants a *higher* learning rate than
  full fine-tuning (1e-4 to 3e-4 is a reasonable range). Too low and
  nothing visibly happens in a few epochs.
- Check your data — make sure `data.text_field` actually matches the key
  name used in your `.jsonl` files.
- Make sure `lora.target_modules` actually matches real layer names in your
  model (see the table in [03_training_guide.md](03_training_guide.md)) —
  if the names are wrong, PEFT may silently attach adapters to nothing
  useful, or raise an error.

## `bitsandbytes` fails to install or import

This library depends on CUDA and doesn't support every platform (notably
Apple Silicon / CPU-only Windows). This is fine as long as
`model.load_in_4bit` stays `false` in your config — it's only needed for
QLoRA-style 4-bit loading.

## "This model's tokenizer does not have a pad token"

Already handled in `scripts/train.py` and `scripts/inference.py` — the EOS
token is reused as the pad token if none exists. If you see this error
elsewhere (e.g. writing your own script), add:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## Generated text looks like gibberish or ignores the instruction

- You likely need more training data or more epochs — a handful of
  examples for 1-2 epochs on a small model is a proof-of-concept, not a
  polished model.
- Double check the prompt format at inference time matches the format used
  in training exactly (same `### Instruction:` / `### Response:` markers,
  same whitespace).
- Try lowering `temperature` in `scripts/inference.py` (e.g. to 0.3) for
  more focused, less random output while you're debugging whether the model
  learned the task at all.

## Training is extremely slow on CPU

Expected for anything beyond tiny models like `distilgpt2`/`gpt2`. Options:
- Reduce dataset size and `num_train_epochs` for quick experiments.
- Use a cloud GPU notebook (e.g. Google Colab, which offers free GPU time)
  if you don't have a local NVIDIA GPU.

## `ImportError` for `peft`, `transformers`, `datasets`, etc.

Make sure your virtual environment is activated (`(.venv)` should show in
your terminal prompt — see [02_setup.md](02_setup.md)) and that you ran
`pip install -r requirements.txt` inside it.

## Where to ask for more help

- Hugging Face PEFT docs: https://huggingface.co/docs/peft
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers
- Hugging Face forums: https://discuss.huggingface.co
