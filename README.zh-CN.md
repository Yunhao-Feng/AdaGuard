# AdaGuard

**给出你的规则，让 AdaGuard 检查 AI 智能体的行为是否违规。**

[English](README.md) | [简体中文](README.zh-CN.md)

[![arXiv 论文](https://img.shields.io/badge/arXiv-2609.34241-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.34241)
[![Hugging Face 论文页](https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/papers/2609.34241)

[![AdaGuard 0.6B 模型权重](https://img.shields.io/badge/AdaGuard-0.6B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Yunhao-Feng/AdaGuard-0.6B)
[![AdaGuard 4B 模型权重](https://img.shields.io/badge/AdaGuard-4B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Yunhao-Feng/AdaGuard-4B)
[![AdaGuard 8B 模型权重](https://img.shields.io/badge/AdaGuard-8B-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E)](https://huggingface.co/Yunhao-Feng/AdaGuard-8B)

AdaGuard 的输入是**一组规则**和**智能体的交互记录**，输出是判断说明和被违反的规则编号。你可以在每次调用时更换规则，不必局限于一套固定的风险分类。

例如，规则 `R1` 是“禁止尝试删除受保护的文件”。记录中的 `delete_file("protected_draft")` 应被判为违反 `R1`，而读取一份不受限制的笔记不违反这条规则。这些只是待检查的文本，AdaGuard 不会执行其中的操作。

## 论文与模型

| 资源 | 链接 |
| --- | --- |
| 论文 | [AdaGuard: An Adaptive Guard Model with User-defined Policies](https://arxiv.org/abs/2609.34241) |
| 论文讨论页 | [Hugging Face Papers](https://huggingface.co/papers/2609.34241) |
| AdaGuard-0.6B 模型权重 | [Yunhao-Feng/AdaGuard-0.6B](https://huggingface.co/Yunhao-Feng/AdaGuard-0.6B) |
| AdaGuard-4B 模型权重 | [Yunhao-Feng/AdaGuard-4B](https://huggingface.co/Yunhao-Feng/AdaGuard-4B) |
| AdaGuard-8B 模型权重 | [Yunhao-Feng/AdaGuard-8B](https://huggingface.co/Yunhao-Feng/AdaGuard-8B) |
| 引用文件 | [BibTeX](CITATION.bib) · [GitHub 引用元数据](CITATION.cff) |

三个尺寸使用相同的调用接口。较大的模型需要更多内存或显存；本仓库没有提供实测硬件需求，也不保证某个尺寸一定能在你的设备上运行。

## 快速开始

### 1. 安装

使用 Python 3.10 或更高版本，先安装适合本机硬件的 PyTorch，再在仓库根目录运行：

```bash
python -m pip install -e ".[test]"
```

### 2. 下载权重

以 0.6B 为例：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Yunhao-Feng/AdaGuard-0.6B', local_dir='checkpoints/AdaGuard-0.6B')"
```

下载 4B 或 8B 时，将命令中的两处 `0.6B` 都替换成对应尺寸。请在导入 `adaguard.model` **之前，用独立进程完成下载**，因为模型运行模块默认使用离线模式。

训练和推理接收的是**本地模型目录**，不是 Hugging Face 仓库 ID。加载使用 `local_files_only=True`、Safetensors 权重和 `trust_remote_code=False`，并要求 Qwen 兼容的快速 ChatML tokenizer；缺失的权重不会被自动下载。

### 3. 检查一段交互

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

没有合适的 GPU 时，可使用 `device="cpu", dtype="float32"`，但运行可能较慢。没有记录智能体思考内容时，将 `thought` 留空，不要补造。

### 4. 理解输出

| 字段 | 含义 |
| --- | --- |
| `status` | `OK` 表示回答完整且格式合法；`INVALID` 表示输出不可用于判断。格式合法不等于判断一定正确。 |
| `analysis` | 模型给出的判断说明。 |
| `violated_ids` | 按输入规则顺序排列的违规编号；`[]` 表示未发现违反所给规则。 |
| `unsafe` | 预测有违规时为 `True`，否则为 `False`。 |
| `raw_response` | 模型生成的原始文本，供排查问题使用。 |
| `output_tokens` | 生成的 token 数量。 |

当 `status` 为 `INVALID` 时，`analysis`、`violated_ids` 和 `unsafe` 都为 `None`。**不要把无效输出当作安全结论。** 原始文本中的 `NR` 表示没有违规，Python 接口会将其转成空列表。

### 批量推理

仓库自带一个合成输入示例。先创建输出目录，再运行：

```bash
python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
python -m adaguard.infer --model checkpoints/AdaGuard-0.6B --input data/inference.jsonl --output outputs/predictions.jsonl --device cuda --dtype bfloat16
```

输出文件必须尚不存在；重复运行时请换一个文件名。自定义输入格式见[数据说明](docs/data.md)。

## 三个名称分别是什么？

| 名称 | 含义 |
| --- | --- |
| **AdaGuard** | 根据用户规则检查智能体行为的模型系列。 |
| **AdaptiveSafety** | 论文提出的数据集；本仓库**不包含完整数据集**。 |
| **SafePO** | 在监督微调之后使用的强化学习方法，用来改进违规规则识别。 |

仓库包含推理、评估、单设备训练参考实现，以及四条手写训练示例。示例只用于说明数据格式，**不是评测基准**。模型权重通过上方链接单独提供。

## 训练与开发

- [训练与评估](docs/training.md)：数据划分、监督微调、SafePO 和指标计算。
- [数据说明](docs/data.md)：字段含义、规则编号要求和分组划分。
- [方法细节](docs/method.md)：奖励、损失、token 权重与实现边界。
- [验证说明](docs/validation.md)：测试覆盖范围及其限制。
- [英文首页的目录索引](README.md#what-is-in-this-repository)：主要源码文件。

训练入口是**单设备参考实现**，没有包含原始分布式实验配置、自动选取最佳检查点或自动断点续训。默认参数和合成示例不代表能够复现论文分数。

在仓库根目录执行检查：

```bash
python -B -m pytest -p no:cacheprovider tests
python -B tools/audit_release.py
```

审计允许项目的公开资源链接，同时检查其他端点、常见凭据模式及异常文件。在 Git 仓库中，它扫描已跟踪文件和未被忽略的新文件，不扫描 Git 历史或被忽略的本地产物。检查通过不代表绝对没有敏感信息。

## 使用限制与许可

AdaGuard 给出的是模型预测，不是安全保证。判断取决于你提供的规则和行为证据；实际部署仍需结合应用层验证和适当的人工监督。

源码采用 [MIT 许可证](LICENSE)。模型权重、数据集和上游依赖遵循各自的许可，使用或再分发前请查看对应资源页。

## 引用

如果你的工作使用了 AdaGuard，请引用 [arXiv 论文](https://arxiv.org/abs/2609.34241)。也可直接使用 [CITATION.bib](CITATION.bib)。

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
