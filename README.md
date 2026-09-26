# AdaGuard

Anonymous source release for policy-conditioned safety assessment.

- **AdaptiveSafety** is the dataset.
- **SafePO** is the reinforcement learning algorithm.
- **AdaGuard** is the guard model family.

Given a policy and an interaction record, the guard generates one analysis and an ordered list of violated rule identifiers. `NR` represents an empty violation set. Rule identifiers are local to each policy.

## Included

| File | Purpose |
| --- | --- |
| `adaguard/data.py` | Strict data validation, prompt construction, grouped train/development split |
| `adaguard/prompts.py` | Policy/evidence instructions |
| `adaguard/train_sft.py` | Full-parameter supervised training; label span weight 4 |
| `adaguard/safepo.py` | Group-relative advantages, prefix-value modulation, fixed region budgets and losses |
| `adaguard/reward.py` | Exact, set and sequence reward with format and length handling |
| `adaguard/train_safepo.py` | Local SafePO training entry point |
| `adaguard/infer.py` | Python inference interface and JSONL command |
| `adaguard/evaluate.py` | Local binary and rule-set evaluation |
| `tools/audit_release.py` | Text-only source and credential checks |

The training entry points are single-device reference implementations. They expose the training computation without the original distributed orchestration, server paths or experiment history. They are not a claim of reproducing the paper's scores with default command-line settings. See `docs/method.md` for the implementation boundary.

## Install

Use Python 3.10 or newer. Install a PyTorch build appropriate for your hardware, then:

```bash
pip install -e '.[test]'
```

Training and inference accept **existing local model directories** containing standard Transformers configuration, a fast Qwen-compatible ChatML tokenizer, and Safetensors weights. The code disables Hub access and telemetry, uses `local_files_only=True`, and does not execute checkpoint-provided Python code. Model acquisition is outside this repository.

For trained AdaGuard weights, use `Guard` below. A base checkpoint is an initialization for training, not an evaluated AdaGuard model.

## Data

Only four hand-written synthetic examples are included. They demonstrate the schema and are not AdaptiveSafety benchmark data. No original trajectories, generated annotations or model weights are redistributed in this source release.

Put separately reviewed training data in `private_data/train_source.jsonl`, following `docs/data.md`. Derive development data from the training pool:

```bash
python -m adaguard.data --input private_data/train_source.jsonl --output private_data/prepared
```

For a local schema check, substitute `data/examples.jsonl`. Never pass the held-out test set to this split command or to either trainer.

## Supervised training

```bash
python -m adaguard.train_sft \
  --model checkpoints/base \
  --train private_data/prepared/train.jsonl \
  --output outputs/sft --device cuda --dtype bfloat16
```

This saves `outputs/sft/epoch-1`. The reference loop uses one example per optimizer step, with full-parameter AdamW. Choose learning rate and training duration for the intended experiment; defaults are illustrative. Prompt tokens have zero loss weight; the complete label span has weight 4; other response tokens have weight 1. Inputs that exceed the context budget fail explicitly.

## SafePO training

```bash
python -m adaguard.train_safepo \
  --model outputs/sft/epoch-1 \
  --train private_data/prepared/train.jsonl \
  --output outputs/safepo --device cuda --dtype bfloat16 \
  --steps 100 --prompts-per-step 1
```

Each prompt produces eight sampled responses. The actor, frozen supervised reference and independent value backbone reside on the selected device. Full-parameter optimization requires substantial memory, especially for 4B/8B models. This release does not include FSDP, inference-engine offloading or multi-GPU scheduling.

Checkpoints and numerical metrics are saved periodically. There is no automatic checkpoint selection and no automatic resume. Training state is saved for inspection or an explicitly implemented continuation; rerunning a command requires a new output directory. For a quick execution check, use `--steps 1 --max-new-tokens 16`; short generation budgets are smoke tests, not the paper protocol.

## Inference

```python
from adaguard.infer import Guard

guard = Guard('checkpoints/adaguard', device='cuda', dtype='bfloat16')
result = guard.predict(
    policy=[{'id': 'R1', 'text': 'Do not attempt to delete protected files.'}],
    content=[[{'role': 'user', 'content': 'Please handle the protected draft.'},
              {'role': 'agent', 'thought': '',
               'action': 'delete_file("protected_draft")'}]],
)
print(result)
```

The result contains `status`, `analysis`, `violated_ids`, `unsafe`, raw response and output-token count. Malformed, unfinished or out-of-order outputs are `INVALID`; their safety fields are `None`.

```bash
python -m adaguard.infer --model checkpoints/adaguard \
  --input data/inference.jsonl --output outputs/predictions.jsonl \
  --device cuda --dtype bfloat16
python -m adaguard.evaluate --data private_data/test.jsonl \
  --predictions outputs/test_predictions.jsonl --output outputs/metrics.json
```

Evaluation requires one prediction per reference record in the same order. Reports distinguish all-sample accuracy/exact match (invalid outputs count as incorrect) from successful-only precision, recall and F1. No aggregate probability is invented from generated labels.

## Verification and anonymous packaging

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests
python tools/audit_release.py
```

The audit prints locations and rule names, never matched secrets. It rejects unexpected binary files, hidden files, symlinks, network client imports, endpoints and several credential patterns. It is a check, not proof that arbitrary text contains no identifying content. Keep local datasets, checkpoints, logs, build artifacts and version-control metadata out of the submitted folder. Synthetic examples and source files have been reviewed separately.

See `docs/validation.md` for what was actually tested. See `LICENSE` for the source license. Upstream models and dependencies retain their own licenses; this source license does not grant rights to redistribute them.
