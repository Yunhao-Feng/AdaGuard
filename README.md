# AdaGuard

**Check an AI agent's behavior against rules you provide.**

[English](README.md) | [简体中文](README.zh-CN.md)

[![arXiv paper](https://img.shields.io/badge/arXiv-2609.34241-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.34241)
[![Hugging Face paper](https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/papers/2609.34241)

[![AdaGuard 0.6B model](https://img.shields.io/badge/AdaGuard-0.6B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Yunhao-Feng/AdaGuard-0.6B)
[![AdaGuard 4B model](https://img.shields.io/badge/AdaGuard-4B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Yunhao-Feng/AdaGuard-4B)
[![AdaGuard 8B model](https://img.shields.io/badge/AdaGuard-8B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Yunhao-Feng/AdaGuard-8B)

AdaGuard takes two inputs: **your rules** and **a record of what an agent did**. It returns an explanation and the IDs of the rules that were violated. You can change the rules for each request, without changing a fixed list of safety categories.

For example, if rule `R1` says “Do not attempt to delete protected files,” an agent action such as `delete_file("protected_draft")` should be flagged as violating `R1`. Reading an unrestricted note would not violate that rule. These are input examples, not commands executed by AdaGuard.

## Resources

| Resource | Link |
| --- | --- |
| Paper | [AdaGuard: An Adaptive Guard Model with User-defined Policies](https://arxiv.org/abs/2609.34241) |
| Paper discussion | [Hugging Face Papers](https://huggingface.co/papers/2609.34241) |
| AdaGuard-0.6B weights | [Yunhao-Feng/AdaGuard-0.6B](https://huggingface.co/Yunhao-Feng/AdaGuard-0.6B) |
| AdaGuard-4B weights | [Yunhao-Feng/AdaGuard-4B](https://huggingface.co/Yunhao-Feng/AdaGuard-4B) |
| AdaGuard-8B weights | [Yunhao-Feng/AdaGuard-8B](https://huggingface.co/Yunhao-Feng/AdaGuard-8B) |
| Citation files | [BibTeX](CITATION.bib) · [GitHub citation metadata](CITATION.cff) |

All three sizes use the same interface. Larger checkpoints need more memory; this repository does not provide measured hardware requirements or guarantee that a given model fits your device.

## Quick start

### 1. Install

Use Python 3.10 or newer and install a PyTorch build suitable for your hardware. From the repository root:

```bash
python -m pip install -e ".[test]"
```

### 2. Download a model

Download a checkpoint before running inference. For example:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Yunhao-Feng/AdaGuard-0.6B', local_dir='checkpoints/AdaGuard-0.6B')"
```

For the other sizes, replace both occurrences of `0.6B` with `4B` or `8B`. Run the download in a separate process before importing `adaguard.model`: the model runtime defaults to offline mode.

Training and inference load **local directories**, not Hub IDs. The runtime uses `local_files_only=True`, Safetensors weights and `trust_remote_code=False`; it requires a fast Qwen-compatible ChatML tokenizer. It does not automatically download a missing checkpoint.

### 3. Check an interaction

```python
from adaguard.infer import Guard

guard = Guard("checkpoints/AdaGuard-0.6B", device="cuda", dtype="bfloat16")
result = guard.predict(
    policy=[{"id": "R1", "text": "Do not attempt to delete protected files."}],
    content=[[
        {"role": "user", "content": "Please handle the protected draft."},
        {"role": "agent", "thought": "", "action": 'delete_file("protected_draft")'},
    ]],
)
print(result)
```

Use `device="cpu", dtype="float32"` if needed; CPU inference may be slow. Leave `thought` empty when no agent reasoning was recorded.

### 4. Read the result

| Field | Meaning |
| --- | --- |
| `status` | `OK` for a complete, correctly formatted answer; otherwise `INVALID`. This is a format check, not a guarantee of correctness. |
| `analysis` | The model's explanation. |
| `violated_ids` | Rule IDs in the order supplied by your policy; `[]` means no supplied rule was found to be violated. |
| `unsafe` | `True` if any supplied rule is predicted to be violated; `False` otherwise. |
| `raw_response` | The generated text, for debugging. |
| `output_tokens` | Number of generated tokens. |

For an `INVALID` answer, `analysis`, `violated_ids` and `unsafe` are `None`. **Do not treat an invalid answer as safe.** The raw text uses `NR` for no violation; the Python interface returns an empty list instead.

### Batch inference

The included input contains one synthetic example. Create the output directory first:

```bash
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python -m adaguard.infer --model checkpoints/AdaGuard-0.6B --input data/inference.jsonl --output outputs/predictions.jsonl --device cuda --dtype bfloat16
```

Output files must not already exist. Choose a new filename for another run. See the [data guide](docs/data.md) to prepare your own inputs.

## What is in this repository?

| Name | In plain language |
| --- | --- |
| **AdaGuard** | The guard model family that checks behavior against your policy. |
| **AdaptiveSafety** | The dataset introduced in the paper. The full dataset is **not included** here. |
| **SafePO** | The reinforcement-learning method used after supervised training to improve rule-violation predictions. |

The repository includes inference, evaluation, single-device training reference code and four hand-written training examples. The examples demonstrate the format; they are **not benchmark data**. Model weights are hosted separately at the links above.

| Path | Purpose |
| --- | --- |
| [adaguard/infer.py](adaguard/infer.py) | Python interface and batch inference |
| [adaguard/data.py](adaguard/data.py) | Input validation, prompt construction and grouped data splitting |
| [adaguard/train_sft.py](adaguard/train_sft.py) | Supervised fine-tuning |
| [adaguard/train_safepo.py](adaguard/train_safepo.py) | SafePO training entry point |
| [adaguard/safepo.py](adaguard/safepo.py) · [adaguard/reward.py](adaguard/reward.py) | Training losses, token weights and rewards |
| [adaguard/evaluate.py](adaguard/evaluate.py) | Binary and rule-set metrics |
| [tests](tests/) | Unit tests and tiny-model integration tests |
| [tools/audit_release.py](tools/audit_release.py) | Source-release checks for credentials and unexpected files or links |

## Training, evaluation and checks

- [Training and evaluation](docs/training.md): prepare data, run supervised fine-tuning, run SafePO and evaluate predictions.
- [Data format](docs/data.md): required fields, rule IDs and how related examples stay in the same split.
- [Method details](docs/method.md): exact rewards, losses and implementation limits.
- [Validation](docs/validation.md): what the tests cover and what they do not establish.

The trainers expose the core computation on a **single device**. They do not include the original distributed experiment setup, automatic checkpoint selection or automatic resume. Default settings and synthetic examples are not a recipe for reproducing the paper's scores.

Run the checks from the repository root:

```bash
python -B -m pytest -p no:cacheprovider tests
python -B tools/audit_release.py
```

The audit allows the project's public resource links while checking other endpoints and common credential patterns. In a Git checkout it scans tracked and non-ignored untracked files, not Git history or ignored local artifacts. It is a best-effort check, not proof that a release is free of sensitive information.

## Limitations and license

AdaGuard makes model predictions, not safety guarantees. Its judgments depend on the rules and evidence you supply. Use application-level validation and appropriate oversight before acting on predictions.

Source code is covered by the [MIT license](LICENSE). Model checkpoints, datasets and upstream dependencies retain their own licenses; consult their resource pages before use or redistribution.

## Citation

If you use AdaGuard in your work, please cite the [arXiv paper](https://arxiv.org/abs/2609.34241). The same entry is available in [CITATION.bib](CITATION.bib).

```bibtex
@misc{feng2026adaguard,
  title         = {{AdaGuard}: An Adaptive Guard Model with User-defined Policies},
  author        = {Yunhao Feng and Yifan Ding and Yuxiang Xie and Zheng Li and Mingrui Lao and Zeyuan Wang and Yanming Guo},
  year          = {2026},
  eprint        = {2609.34241},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2609.34241},
  url           = {https://arxiv.org/abs/2609.34241}
}
```
