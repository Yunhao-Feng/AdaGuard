"""SafePO numerical core. All actor weights are detached."""
from dataclasses import asdict, dataclass
import math

import torch


RECIPE = "safepo-v1"
REWARD_RECIPE = "set_order_v1"
ADVANTAGE_NAME = "safepo"
POLICY_LOSS_NAME = "safepo_ppo"


def reward_contract():
    return {"recipe": REWARD_RECIPE, "invalid": -1., "exact": 1.,
            "same_set_intercept": .8, "same_set_sequence": .15,
            "partial_intercept": -.5, "partial_set": 1., "partial_sequence": .25,
            "length_penalty": .05, "group_size": 8, "group_epsilon": 1e-6,
            "soft_response_length": 384, "max_response_length": 640,
            "value_bound": "2/pi*atan", "value_clip": .2, "quality_decay": .95,
            "modulation_amplitude": .5, "actor_clip": .1, "kl_coef": .005}


def bounded_value_hook(_module, _inputs, output):
    return (2.0 / math.pi) * torch.atan(output.float())


@dataclass
class ValueQuality:
    value_mse: float | None = None
    group_mse: float | None = None
    kappa: float = 0.0
    batches: int = 0

    def observe(self, value_mse: float, group_mse: float):
        if not all(math.isfinite(x) and x >= 0 for x in (value_mse, group_mse)):
            raise ValueError("nonfinite or negative critic quality MSE")
        self.value_mse = value_mse if self.value_mse is None else .95 * self.value_mse + .05 * value_mse
        self.group_mse = group_mse if self.group_mse is None else .95 * self.group_mse + .05 * group_mse
        self.kappa = (max(0., min(1., 1. - self.value_mse / (self.group_mse + 1e-8)))
                      if self.group_mse > 1e-8 else 0.)
        self.batches += 1

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        value = cls(**state)
        if (not isinstance(value.batches, int) or value.batches < 0 or
                not math.isfinite(value.kappa) or not 0 <= value.kappa <= 1):
            raise ValueError("invalid value quality state")
        for x in (value.value_mse, value.group_mse):
            if x is not None and (not math.isfinite(x) or x < 0):
                raise ValueError("invalid quality EMA")
        if (value.batches == 0) != (value.value_mse is None and value.group_mse is None):
            raise ValueError("incomplete quality EMA state")
        if value.batches and (value.value_mse is None or value.group_mse is None):
            raise ValueError("incomplete quality EMA state")
        return value


@torch.no_grad()
def compute_safepo(rewards, response_mask, prompt_ids, analysis_mask, structured,
                       *, values=None, quality=None, verdict_weight=2.0):
    """Return weighted advantages, MC targets, fixed region weights and diagnostics.

    ``values[:,t]`` is already bounded V(prefix before response token t).
    Caller must pass the whole rollout batch before microbatch/DP splitting.
    """
    mask = response_mask.bool()
    if mask.ndim != 2 or not mask.any(-1).all() or verdict_weight not in (1., 2.):
        raise ValueError("invalid response mask or region mass")
    n, width = mask.shape
    rewards = torch.as_tensor(rewards, device=mask.device, dtype=torch.float32)
    analysis = torch.as_tensor(analysis_mask, device=mask.device, dtype=torch.bool) & mask
    structured = torch.as_tensor(structured, device=mask.device, dtype=torch.bool)
    if rewards.shape != (n,) or analysis.shape != mask.shape or structured.shape != (n,):
        raise ValueError("misaligned reward/region metadata")
    if len(prompt_ids) != n or not torch.isfinite(rewards).all() or (rewards.abs() > 1.000001).any():
        raise ValueError("invalid rewards or prompt groups")
    groups = {}
    for i, uid in enumerate(prompt_ids):
        if not uid:
            raise ValueError("empty prompt UID")
        groups.setdefault(str(uid), []).append(i)
    scalar, loo = torch.zeros_like(rewards), torch.zeros_like(rewards)
    equal = 0
    for rows in groups.values():
        if len(rows) != 8:
            raise ValueError(f"SafePO requires 8 responses per prompt, got {len(rows)}")
        r = rewards[rows]
        same = bool((r == r[0]).all())
        # Decimal rewards (e.g. .9) can acquire a nonzero residual in FP32
        # reduction. All-equal groups must be EXACTLY zero, not amplified by eps.
        if same:
            scalar[rows] = 0.
            loo[rows] = r
        else:
            scalar[rows] = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
            loo[rows] = (r.sum() - r) / 7
        equal += int(same)
    verdict = mask & ~analysis
    structured = structured & analysis.any(-1) & verdict.any(-1)
    fixed = mask.float() / mask.sum(-1, keepdim=True)
    regional = (analysis.float() / analysis.sum(-1, keepdim=True).clamp_min(1)
                + verdict_weight * verdict.float() / verdict.sum(-1, keepdim=True).clamp_min(1)) / (1 + verdict_weight)
    fixed = torch.where(structured[:, None], regional, fixed)
    returns = rewards[:, None] * mask
    coefficients = torch.ones_like(fixed)
    used_kappa = quality.kappa if quality is not None and values is not None else 0.
    stats = {"kappa_used": used_kappa, "all_equal_group_fraction": equal / len(groups),
             "all_wrong_group_fraction": sum(bool((rewards[rows] < 1).all()) for rows in groups.values()) / len(groups),
             "sequence_adv_abs": scalar.abs().mean().item(),
             "structured_region_rate": structured.float().mean().item()}
    if values is not None:
        values = values.detach().float()
        if values.shape != mask.shape or not torch.isfinite(values[mask]).all() or (values[mask].abs() > 1.000001).any():
            raise ValueError("critic values must be finite, aligned, and bounded")
        delta = torch.zeros_like(values)
        # Explicit active indices: do not use padded values as terminal feedback.
        for i in range(n):
            active = mask[i].nonzero().flatten()
            delta[i, active[:-1]] = values[i, active[1:]] - values[i, active[:-1]]
        coefficients += .5 * used_kappa * torch.tanh(scalar.sign()[:, None] * delta)
        coefficients = torch.where(structured[:, None], coefficients, torch.ones_like(coefficients))
        ev = (((values - returns).square() * fixed).sum(-1)).mean().item()
        eg = (rewards - loo).square().mean().item()
        stats.update(value_mse=ev, group_mse=eg)
        if quality is None:
            raise ValueError("critic mode needs checkpointable quality state")
        quality.observe(ev, eg)  # affects the NEXT call, never coefficients above
    weights = fixed.clone()
    for region, mass in ((analysis, 1 / (1 + verdict_weight)),
                         (verdict, verdict_weight / (1 + verdict_weight))):
        local = coefficients * region
        local = mass * local / local.sum(-1, keepdim=True).clamp_min(1e-8)
        weights = torch.where(structured[:, None] & region, local, weights)
    advantages = scalar[:, None] * weights * width
    stats.update(kappa_next=quality.kappa if quality else 0.,
                 region_weight_error=(weights.sum(-1) - 1).abs().max().item(),
                 advantage_mass_error=(advantages.sum(-1) / width - scalar).abs().max().item())
    return advantages, returns, fixed, stats


def actor_loss(log_probs, old_log_probs, ref_log_probs, weighted_advantages,
               fixed_weights, clip=.1, beta=.005):
    """One response's clipped task loss plus fixed-weight k3 KL."""
    ratio = (log_probs - old_log_probs.detach()).exp()
    advantage = weighted_advantages.detach()
    task = -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).sum()
    h = ref_log_probs.detach() - log_probs
    kl = ((h.exp() - h - 1) * fixed_weights.detach()).sum()
    return task + beta * kl


def value_loss(values, old_values, reward, fixed_weights, clip=.2):
    clipped = values.clamp(old_values.detach()-clip, old_values.detach()+clip)
    return .5 * (torch.maximum((values-reward).square(), (clipped-reward).square()) * fixed_weights.detach()).sum()
