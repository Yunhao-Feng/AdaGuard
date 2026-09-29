# How AdaGuard is trained

[Home](../README.md) · [中文首页](../README.zh-CN.md) · [Training commands](training.md)

## The short version

1. **Learn from examples.** Supervised fine-tuning teaches the model to explain a judgment and list the violated rules.
2. **Compare several answers.** SafePO samples eight answers for the same input and rewards accurate rule predictions.
3. **Balance explanation and verdict.** The two parts receive separate token-weight budgets, so a long explanation does not overwhelm the final rule labels.

The sections below describe the computation implemented in this repository. `P` means the predicted sequence of rule IDs; `G` means the reference sequence. An “advantage” measures how an answer's reward compares with the other answers for that input. A “prefix value” estimates final reward from the text generated so far.

## Supervised initialization

The prompt follows the local ChatML template. The complete `<label>...</label>` span receives weight 4, other response tokens receive weight 1, and prompt positions receive weight 0. The loss is weighted token cross-entropy divided by total active weight. The tokenizer, template and model are saved together.

## Structured reward

For a parse-valid prediction sequence P and reference G, set F1 uses set overlap; sequence F1 uses the longest common subsequence. Both scores are 1 when P and G are empty and 0 when exactly one is empty.

- Invalid or unfinished response: -1.
- Exact sequence match: 1.
- Same set, different order: 0.8 + 0.15 × sequence F1.
- Otherwise: -0.5 + set F1 + 0.25 × sequence F1.

Only valid non-exact responses receive a length penalty, increasing from 0 at 384 response tokens to 0.05 at 640. There is no separate binary reward and no independent factual check of the explanation. Order errors retain partial reward during training but are marked invalid by the strict inference interface.

## SafePO

Eight responses are sampled per prompt with an untempered, unfiltered sampling distribution. Response-level advantages are centered by the group's mean reward and divided by its population standard deviation plus 1e-6. Equal-reward groups have exactly zero advantage.

The analysis body receives total weight 1/3; structure, verdict and EOS receive total weight 2/3. An independent value backbone initialized from the supervised model predicts final reward from each causal prefix, using a zero-initialized scalar head and the bound `(2/pi) * atan(z)`.

Adjacent pre-update value differences modulate token weights by `1 + (kappa/2) * tanh(sign(advantage) * delta_value)`. The terminal difference is zero. Weights are normalized separately within each region and detached before actor updates. Unreliable parsing/token alignment falls back to uniform weights for both task and fixed-weight terms.

Kappa starts at zero. Pre-update value MSE is compared with a leave-one-out reward predictor using exponential moving averages (decay 0.95). The resulting clipped coefficient affects the next batch only. The leave-one-out predictor is a modulation-quality reference, not the actor advantage baseline.

The value loss uses final reward targets and value clipping at 0.2. The actor uses a clipped surrogate (0.1) plus fixed-region k3 KL to the frozen supervised reference (coefficient 0.005). The value-derived weights affect only the task term. We compute all old log probabilities, values, advantages and weights before updating either model, then perform one value update and one actor update.

## Implementation limits

These are full-parameter, single-device reference trainers. They do not include distributed orchestration, automatic checkpoint selection or automatic resume. SafePO keeps the actor, reference and value backbone on the same device. The four bundled examples verify the format, not the paper's benchmark results.

See the [paper](https://arxiv.org/abs/2609.34241) for the research description and the [validation guide](validation.md) for test coverage. The reward checks rule labels and output structure; it does not independently verify the explanation's factual accuracy.
