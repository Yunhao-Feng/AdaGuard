"""Validate local records and split related examples as indivisible groups."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from .formatting import parse_guard_response, LABEL_PATTERN
from .prompts import SYSTEM_INSTRUCTION, USER_INSTRUCTION


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def validate(row, target=True):
    required = {'policy', 'content'} | ({'analysis', 'violated_ids', 'group_id'} if target else set())
    allowed = required | {'analysis', 'violated_ids', 'group_id'}
    if not isinstance(row, dict) or not required <= row.keys() or row.keys() - allowed:
        raise ValueError('record has missing or unexpected fields')
    policy = row['policy']
    if not isinstance(policy, list) or not 1 <= len(policy) <= 100:
        raise ValueError('policy must contain 1..100 rules')
    ids = []
    for rule in policy:
        if not isinstance(rule, dict) or set(rule) != {'id', 'text'}:
            raise ValueError('each rule requires id and text only')
        if not isinstance(rule['id'], str) or not re.fullmatch(LABEL_PATTERN, rule['id']) or rule['id'].upper() == 'NR':
            raise ValueError('invalid or reserved rule identifier')
        if not isinstance(rule['text'], str) or not rule['text'].strip():
            raise ValueError('empty rule text')
        ids.append(rule['id'])
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate rule identifier')
    if not isinstance(row['content'], list) or not row['content']:
        raise ValueError('content must be a nonempty list of event segments')
    for segment in row['content']:
        if not isinstance(segment, list) or not segment:
            raise ValueError('empty event segment')
        for event in segment:
            if not isinstance(event, dict):
                raise ValueError('invalid event')
            fields = {'role', 'thought', 'action'} if event.get('role') == 'agent' else {'role', 'content'}
            if event.get('role') not in {'user', 'agent', 'environment'} or set(event) != fields:
                raise ValueError('invalid event role or fields')
            if any(not isinstance(v, str) for v in event.values()):
                raise ValueError('event fields must be strings')
            if not any(event[k].strip() for k in fields - {'role'}):
                raise ValueError('empty event')
    if target:
        if not isinstance(row['group_id'], str) or not row['group_id'].strip():
            raise ValueError('group_id is required for related variants')
        gold = row['violated_ids']
        if not isinstance(gold, list) or any(not isinstance(x, str) for x in gold):
            raise ValueError('violated_ids must be a list of strings')
        if len(gold) != len(set(gold)) or gold != [x for x in ids if x in gold]:
            raise ValueError('gold IDs must be unique, in policy, and in policy order')
        if not isinstance(row['analysis'], str):
            raise ValueError('analysis must be text')
        if not parse_guard_response(answer(row)).valid:
            raise ValueError('invalid reference analysis or verdict')
    return row


def answer(row):
    return '<analysis>\n' + row['analysis'] + '\n</analysis>\n<label>' + (','.join(row['violated_ids']) or 'NR') + '</label>'


def messages(row):
    validate(row, target=False)
    policy = {'no_risk_id': 'NR', 'no_risk_condition': 'No supplied rule is violated.',
              'single_label': False, 'risk_rules': [
                  {'id': r['id'], 'risk_category': r['id'], 'risk_description': r['text']} for r in row['policy']]}
    # Neutralize literal delimiters and model special tokens inside input text.
    encode = lambda obj: canonical(obj).replace('<', '\\u003c').replace('>', '\\u003e')
    return [{'role': 'system', 'content': SYSTEM_INSTRUCTION.format(policy_json=encode(policy))},
            {'role': 'user', 'content': USER_INSTRUCTION.format(content_json=encode({'content': row['content']}))}]


def load(path):
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(validate(json.loads(line)))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f'invalid record at line {n}: {exc}') from None
    if not rows:
        raise ValueError('empty dataset')
    return rows


def split(rows, dev_fraction=.1, seed=42):
    if not 0 < dev_fraction < 1:
        raise ValueError('dev fraction must be between zero and one')
    # Connect both declared counterfactual families and identical content.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in [('group', row['group_id']), ('content', canonical(row['content']))]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    values = list(groups.values())
    if len(values) < 2:
        raise ValueError('need at least two independent groups')
    random.Random(seed).shuffle(values)
    count = min(len(values) - 1, max(1, round(len(values) * dev_fraction)))
    return ([r for g in values[count:] for r in g], [r for g in values[:count] for r in g])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dev-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = load(args.input)
    train, dev = split(rows, args.dev_fraction, args.seed)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in [('train', train), ('dev', dev)]:
        (destination / f'{name}.jsonl').write_text(''.join(canonical(r)+'\n' for r in data))
    manifest = {'train': len(train), 'dev': len(dev), 'seed': args.seed,
                'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest()}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
