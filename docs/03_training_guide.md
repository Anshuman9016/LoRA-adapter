# 03 — Training Guide (Step by Step)

## Step 1 — Look at the data format

Open [`data/train.jsonl`](../data/train.jsonl). Each line is one JSON
object with a `"text"` field — this is called **JSON Lines** (`.jsonl`):
one valid JSON object per line, instead of one big JSON array. It's the
standard format for training datasets because you can stream it line by
line without loading the whole file into memory.

The included sample data uses an instruction/response format:

```json
{"text": "### Instruction:\nWhat is a LoRA adapter?\n\n### Response:\nA LoRA adapter is..."}
```

This teaches the model: "when you see `### Instruction:` followed by a
question, continue with `### Response:` followed by a good answer." This
is a simple but real instruction-tuning format.

## Step 2 — Use your own data (optional)

Replace the contents of `data/train.jsonl` and `data/eval.jsonl` with your
own examples, keeping the same one-JSON-object-per-line structure. Rules of
thumb:

- More examples generally help, but even 50–200 good examples can produce
  a noticeable style/behavior shift with LoRA.
- Keep a small held-out `eval.jsonl` (different examples than training) so
  you can tell if the model is actually learning or just memorizing.
- Consistent formatting matters a lot — pick one instruction/response
  template and stick to it across all examples.

## Step 3 — Understand the config

Open [`configs/lora_config.yaml`](../configs/lora_config.yaml). The
important knobs for a first-timer:

| Setting | What it does | When to change it |
|---|---|---|
| `model.name` | Which pretrained model to start from | Bigger model = better quality, more memory needed |
| `lora.r` | LoRA rank — adapter capacity | Raise to 16–32 if the model isn't learning enough; lower if overfitting |
| `lora.target_modules` | Which layers get adapters | Must match the model architecture (see below) |
| `training.num_train_epochs` | How many passes over the data | More epochs = more learning, but risk of overfitting on small data |
| `training.learning_rate` | Step size for updates | LoRA typically uses higher LR (1e-4 to 3e-4) than full fine-tuning |
| `training.per_device_train_batch_size` | Examples per step | Lower this first if you hit an out-of-memory error |

### Finding the right `target_modules` for your model

LoRA needs to know which weight matrices inside the transformer to attach
adapters to. This varies by model family:

| Model family | `target_modules` |
|---|---|
| GPT-2 / DistilGPT-2 | `["c_attn"]` |
| Llama / Llama 2 / Llama 3 | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| Mistral | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| Phi-2 / Phi-3 | `["q_proj", "k_proj", "v_proj", "dense"]` |
| Falcon | `["query_key_value"]` |

If unsure, run this to list a model's linear layer names and match them
against known patterns:

```bash
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('YOUR_MODEL_NAME')
print(set(n.split('.')[-1] for n, mod in m.named_modules() if 'Linear' in type(mod).__name__))
"
```

## Step 4 — Run training

```bash
python scripts/train.py --config configs/lora_config.yaml
```

What you'll see:

1. `Loading base model: distilgpt2` — downloading/loading the pretrained model.
2. `trainable params: ... || all params: ... || trainable%: ...` — this is
   PEFT confirming how few parameters are actually being trained. For
   `distilgpt2` + rank 8, expect well under 1%.
3. A training loop printing `loss` every `logging_steps` — this number
   should generally trend downward. It won't be perfectly smooth.
4. An eval pass at the end of each epoch showing `eval_loss`.
5. `Done. LoRA adapter saved to: outputs/lora-run-1`

This first run (tiny model, tiny dataset) should take a few minutes even on
CPU.

### Watching training in TensorBoard (optional)

```bash
tensorboard --logdir outputs/lora-run-1
```

Open the printed `localhost` URL in a browser to see loss curves over time.

## Step 5 — Generate text with your fine-tuned model

```bash
python scripts/inference.py --adapter outputs/lora-run-1 --prompt "What is the difference between LoRA and full fine-tuning?"
```

This loads the original base model, applies your trained adapter on top,
and generates a response. Compare it to what the un-tuned base model would
say (you can test this by pointing `--adapter` elsewhere, or just knowing
that the base model would ramble instead of following the instruction
format).

## Step 6 — (Optional) merge the adapter into a standalone model

```bash
python scripts/merge_adapter.py --adapter outputs/lora-run-1 --output outputs/merged-model
```

This produces a folder you can load with plain `transformers`
(`AutoModelForCausalLM.from_pretrained("outputs/merged-model")`) with no
PEFT dependency at inference time — useful for deployment.

## Step 7 — Iterate

Things worth trying next, in order of "cheapest experiment first":

1. Add more of your own training examples.
2. Increase `num_train_epochs` from 3 to 5–10 if the loss is still dropping
   steadily at the end of training.
3. Try a bigger `lora.r` (e.g. 16) if outputs still look too generic.
4. Try a bigger base model once you're comfortable with the pipeline (see
   [04_troubleshooting.md](04_troubleshooting.md) for memory issues).
