"""Minimal full-parameter SFT with label-span weight 4 and prompt weight 0."""
import argparse
import json
import random
from pathlib import Path
from .data import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--max-length', type=int, default=16000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        p.error('epochs and learning rate must be positive')
    import torch
    from .model import load_model, sft_example, check_length, save_actor
    rows = load(args.train)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    examples = [sft_example(tokenizer, row) for row in rows]
    for ids, _ in examples:
        check_length(model, len(ids), args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    with (output/'metrics.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            rng.shuffle(examples)
            for step, (ids, weights) in enumerate(examples):
                optimizer.zero_grad(set_to_none=True)
                inputs = torch.tensor([ids], device=args.device)
                logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0,:-1].float()
                loss_tokens = torch.nn.functional.cross_entropy(logits, inputs[0,1:], reduction='none')
                w = torch.tensor(weights[1:], device=args.device)
                loss = (loss_tokens*w).sum()/w.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite SFT loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                record = {'epoch':epoch+1, 'step':step+1, 'loss':loss.item()}
                log.write(json.dumps(record)+'\n'); log.flush()
                print(json.dumps(record), flush=True)
            save_actor(model, tokenizer, output/f'epoch-{epoch+1}')


if __name__ == '__main__':
    main()
