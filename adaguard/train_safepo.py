"""Single-device SafePO reference trainer, with one actor epoch per rollout."""
import argparse
import json
import random
from pathlib import Path
from .data import load
from .reward import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='local supervised initialization')
    p.add_argument('--train', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--prompts-per-step', type=int, default=1)
    p.add_argument('--actor-lr', type=float, default=2e-7)
    p.add_argument('--value-lr', type=float, default=1e-6)
    p.add_argument('--max-length', type=int, default=16000)
    p.add_argument('--max-new-tokens', type=int, default=640, help='640 for the paper reward budget; smaller values are for smoke tests')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.prompts_per_step, args.save_every) < 1 or min(args.actor_lr,args.value_lr) <= 0:
        p.error('steps, batch size, save interval and learning rates must be positive')
    import torch
    from .model import load_model, generate, logprobs, PrefixValue, save_actor, prompt_ids, check_length
    from .safepo import ValueQuality, compute_safepo, actor_loss, value_loss
    if not 1 <= args.max_new_tokens <= 640:
        p.error('max new tokens must be in 1..640')
    rows = load(args.train)
    if args.prompts_per_step > len(rows):
        p.error('prompts per step exceeds dataset size')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    actor, tokenizer = load_model(args.model, args.device, args.dtype)
    reference, _ = load_model(args.model, args.device, args.dtype)
    reference.requires_grad_(False)
    base, _ = load_model(args.model, args.device, args.dtype)
    value = PrefixValue(base)
    del base
    value.eval()
    for row in rows:
        check_length(actor, len(prompt_ids(tokenizer,row))+args.max_new_tokens, args.max_length)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    value_opt = torch.optim.AdamW(value.parameters(), lr=args.value_lr)
    quality = ValueQuality()
    rng = random.Random(args.seed)
    with (output/'metrics.jsonl').open('w') as log:
        for step in range(1, args.steps+1):
            rollouts = []
            for group, row in enumerate(rng.sample(rows, args.prompts_per_step)):
                for _ in range(8):
                    prompt, response, (text, region, structured, ended) = generate(
                        actor, tokenizer, row, sample=True, max_length=args.max_length, max_new_tokens=args.max_new_tokens)
                    ids = prompt + response
                    with torch.no_grad():
                        old = logprobs(actor, ids, len(prompt)).detach()
                        ref = logprobs(reference, ids, len(prompt)).detach()
                        values = value(ids, len(prompt)).detach()
                    reward = score(text, row['violated_ids'], [r['id'] for r in row['policy']],
                                   tokens=len(response), unfinished=not ended)
                    rollouts.append((ids,len(prompt),old,ref,values,reward,region,structured,str(group)))
            width = max(len(r[2]) for r in rollouts)
            count = len(rollouts)
            mask = torch.zeros((count,width),dtype=torch.bool,device=args.device)
            analysis = torch.zeros_like(mask)
            old_values = torch.zeros((count,width),device=args.device)
            for i,r in enumerate(rollouts):
                length=len(r[2]); mask[i,:length]=True
                analysis[i,:length]=torch.tensor(r[6],device=args.device)
                old_values[i,:length]=r[4]
            # All weights and advantages are computed before either optimizer step.
            advantages, _, fixed, stats = compute_safepo(
                [r[5] for r in rollouts],mask,[r[8] for r in rollouts],analysis,
                [r[7] for r in rollouts],values=old_values,quality=quality)
            advantages = advantages / width  # Remove the backend's padded-width convention.
            value_opt.zero_grad(set_to_none=True)
            value_total=0.
            for i,r in enumerate(rollouts):
                loss=value_loss(value(r[0],r[1]),r[4],r[5],fixed[i,:len(r[2])])/count
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite value loss')
                loss.backward(); value_total+=loss.item()
            torch.nn.utils.clip_grad_norm_(value.parameters(),1.,error_if_nonfinite=True)
            value_opt.step()
            actor_opt.zero_grad(set_to_none=True)
            actor_total=0.
            for i,r in enumerate(rollouts):
                length=len(r[2])
                loss=actor_loss(logprobs(actor,r[0],r[1]),r[2],r[3],advantages[i,:length],fixed[i,:length])/count
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite actor loss')
                loss.backward(); actor_total+=loss.item()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
            actor_opt.step()
            record={'step':step,'reward':sum(r[5] for r in rollouts)/count,
                    'actor_loss':actor_total,'value_loss':value_total,**stats}
            log.write(json.dumps(record)+'\n');log.flush()
            print(json.dumps(record),flush=True)
            if step % args.save_every == 0 or step == args.steps:
                dest=output/f'step-{step}'
                save_actor(actor,tokenizer,dest)
                (dest/'quality.json').write_text(json.dumps(quality.state_dict(),indent=2)+'\n')
                # Local training state; not part of the public source distribution.
                torch.save({'value':value.state_dict(),'actor_optimizer':actor_opt.state_dict(),
                            'value_optimizer':value_opt.state_dict(),'step':step},dest/'training_state.pt')


if __name__ == '__main__':
    main()
