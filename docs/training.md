# Training and evaluation

[Home](../README.md) · [中文首页](../README.zh-CN.md) · [Data format](data.md) · [Method](method.md)

Use the pretrained weights linked from the homepage if you only need inference. This guide is for training or inspecting the method. Run commands from the repository root after installation.

## 1. Prepare your data

Place your training pool in `private_data/train_source.jsonl`, using the [data format](data.md), then run:

```bash
python -m adaguard.data --input private_data/train_source.jsonl --output private_data/prepared
```

This writes `train.jsonl`, `dev.jsonl` and a manifest containing counts, the random seed and the source-file hash. Related examples stay together. Development allocation is based on the number of independent groups, not an exact percentage of rows.

For a format-only exercise, use the four synthetic records:

```bash
python -m adaguard.data --input data/examples.jsonl --output outputs/example_split
```

The destination must not already exist. Keep the held-out test set separate: never send it to the split command or either trainer. The trainers do not automatically evaluate the development split or select a checkpoint.

## 2. Supervised fine-tuning (SFT)

Place a compatible **base model** in `checkpoints/base`. This directory must contain standard Transformers configuration, Safetensors weights and a fast Qwen-compatible ChatML tokenizer. A base model is a training initialization, not an evaluated AdaGuard checkpoint.

```bash
python -m adaguard.train_sft --model checkpoints/base --train private_data/prepared/train.jsonl --output outputs/sft --device cuda --dtype bfloat16
```

With the default one epoch, this saves `outputs/sft/epoch-1`. The loop trains all model parameters with AdamW and uses one example per optimizer step. Prompt tokens receive no loss; the label span receives weight 4 and other answer tokens receive weight 1. Inputs that exceed the context budget raise an error instead of being silently truncated.

Choose the learning rate, context budget and duration for your experiment. Use `python -m adaguard.train_sft --help` to inspect the available options.

## 3. SafePO training

Start from the supervised checkpoint:

```bash
python -m adaguard.train_safepo --model outputs/sft/epoch-1 --train private_data/prepared/train.jsonl --output outputs/safepo --device cuda --dtype bfloat16 --steps 100 --prompts-per-step 1
```

Each prompt produces eight sampled answers. A trainable guard, a frozen supervised reference and a separate value model are all held on the selected device. This needs substantially more memory than inference, especially for 4B and 8B models. There is no FSDP, inference-engine offloading or multi-GPU scheduling in this reference trainer.

By default, checkpoints are saved every ten steps and at the final step, together with metrics and local training state. There is **no automatic resume or checkpoint selection**. A new run needs a new output directory; saved optimizer state alone is not a resume command.

For a short execution check, use `--steps 1 --max-new-tokens 16`. This is only a smoke test; a short answer budget often produces invalid answers and does not test model quality. See [method details](method.md) for the full reward and loss definitions.

## 4. Evaluate a checkpoint

Generate predictions on the same labeled test file that you pass to the evaluator. Reference annotations are not included in the inference prompt.

```bash
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python -m adaguard.infer --model checkpoints/AdaGuard-0.6B --input private_data/test.jsonl --output outputs/test_predictions.jsonl --device cuda --dtype bfloat16
python -m adaguard.evaluate --data private_data/test.jsonl --predictions outputs/test_predictions.jsonl --output outputs/metrics.json
```

Replace the model path with your own trained checkpoint if appropriate. Both output files must be new. Predictions must match the reference records **one-for-one and in the same order**; the evaluator checks the count, but cannot establish alignment by record ID.

| Metric | Interpretation |
| --- | --- |
| `coverage` | Fraction of samples with valid outputs |
| `accuracy_all_invalid_incorrect` | Binary accuracy over all samples; invalid outputs count as errors |
| `set_exact_all_invalid_incorrect` | Fraction with exactly the correct rule set; invalid outputs count as errors |
| `successful_only` | Accuracy, precision, recall and F1 calculated only on valid outputs |

Unsafe is the positive class. An undefined metric is `null`, not zero. Report coverage alongside successful-only metrics so invalid answers are not hidden. The code does not turn generated labels into a calibrated probability.

## Reproducibility boundary

The commands demonstrate the local implementation. The repository does not bundle the full AdaptiveSafety dataset or the original distributed experiment configuration. Passing tests, completing a training loop or evaluating the four synthetic examples does **not** reproduce the paper's benchmark scores.
