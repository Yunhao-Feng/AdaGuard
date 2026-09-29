# Validation and release checks

[Home](../README.md) · [中文首页](../README.zh-CN.md)

## Run the checks

Install the test extra from the repository root, then run:

```bash
python -m pip install -e ".[test]"
python -B -m pytest -p no:cacheprovider tests
python -B tools/audit_release.py
```

The tests do not download pretrained weights. Runtime integration tests construct a tiny, randomly initialized Qwen model locally and run it on CPU. PyTorch, Transformers and the other project dependencies must already be installed.

## Recorded local check

On 2026-09-29, the commands above completed on Windows with Python 3.12.14, PyTorch 2.14.0+cpu, Transformers 4.57.6 and pytest 9.1.1:

- **29 tests passed**, including the three tiny-model runtime tests and README badge checks.
- The source audit reported **0 findings**.
- Package TOML and citation YAML parsed successfully; the two README BibTeX blocks matched `CITATION.bib`.
- The bundled data validated and split into three training examples and one development example.

This records that local check only. The published model weights were not downloaded or evaluated as part of it.

## What is covered?

| Area | Checks |
| --- | --- |
| Data | Required fields, rule IDs, prompt isolation and grouped splitting |
| Rewards | Exact/partial matches, ordering, invalid responses and length handling |
| SafePO | Token-weight budgets, zero-advantage groups and finite gradients |
| Runtime | Local model loading, token regions, context limits, SFT checkpoint creation and a controlled SafePO update |
| Evaluation | Invalid outputs are not counted as safe; undefined metrics remain explicit |
| Public metadata | Resource links, clickable README badges, local documentation links and matching BibTeX entries |
| Release audit | Approved links are accepted; unexpected endpoints and credential patterns remain flagged |

The controlled SafePO test injects known responses to exercise the update. It is not a model-quality evaluation. None of these tests establish published-model accuracy, hardware requirements, successful model downloads or reproduction of the paper's results.

## Source audit scope

In a Git checkout, the audit examines tracked files plus non-ignored untracked files. Git must be available; enumeration errors fail the audit. Ignored weights, local data, outputs and caches are outside this check, but an ignored file that is already tracked is still checked. Git internals and commit history are not scanned.

In an exported directory without `.git`, the audit examines the directory contents. Export only source files and documentation, not local models, caches or private data.

The audit checks unexpected file types, hidden files, symlinks, common credential patterns, machine-local paths, network-client imports and unapproved web endpoints. Exact project resource URLs are permitted in documentation and citation/package metadata; the five approved badge image URLs are permitted only in Markdown. An approved host is **not** a blanket exemption for arbitrary URLs, query strings or credentials. The scanner's own detection signatures are excluded from content scanning.

Findings contain file locations and rule names, never the matched credential text. This is a best-effort source check: review changes yourself, and use a separate history-aware secret scanner when auditing past commits.
