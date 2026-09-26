"""Local inference: python -m adaguard.infer --help."""
import argparse
import json
from pathlib import Path
from .data import validate
from .formatting import parse_guard_response


class Guard:
    def __init__(self, model_path, device='cpu', dtype='float32'):
        from .model import load_model
        self.model, self.tokenizer = load_model(model_path, device, dtype)

    def predict(self, policy, content, max_new_tokens=640, max_length=16000):
        from .model import generate
        row = validate({'policy': policy, 'content': content}, target=False)
        _, response, (text, _, _, ended) = generate(self.model, self.tokenizer, row,
            max_new_tokens=max_new_tokens, max_length=max_length)
        parsed = parse_guard_response(text)
        ids = [r['id'] for r in policy]
        valid = ended and parsed.valid and parsed.labels == tuple(x for x in ids if x in parsed.labels)
        return {'status': 'OK' if valid else 'INVALID', 'raw_response': text,
                'analysis': parsed.analysis if valid else None,
                'violated_ids': list(parsed.labels) if valid else None,
                'unsafe': bool(parsed.labels) if valid else None, 'output_tokens': len(response)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True, help='JSONL with policy and content')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
    args = p.parse_args()
    guard = Guard(args.model, args.device, args.dtype)
    with Path(args.input).open() as source, Path(args.output).open('x') as dest:
        for line in source:
            if not line.strip():
                continue
            row = validate(json.loads(line), target=False)
            dest.write(json.dumps(guard.predict(row['policy'], row['content']), ensure_ascii=False)+'\n')
            dest.flush()


if __name__ == '__main__':
    main()
