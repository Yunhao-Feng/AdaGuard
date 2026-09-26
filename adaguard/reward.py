"""Set/order reward used by SafePO. NR represents an empty set."""
from .formatting import parse_guard_response


def lcs(a, b):
    row = [0] * (len(b) + 1)
    for x in a:
        old = row[:]
        for j, y in enumerate(b, 1):
            row[j] = old[j-1] + 1 if x == y else max(old[j], row[j-1])
    return row[-1]


def score(text, gold, policy_ids, tokens=0, unfinished=False):
    gold, policy_ids = tuple(gold), tuple(policy_ids)
    if len(set(policy_ids)) != len(policy_ids) or gold != tuple(x for x in policy_ids if x in gold):
        raise ValueError('invalid reference labels or policy IDs')
    if tokens < 0:
        raise ValueError('negative response length')
    parsed = parse_guard_response(text)
    if unfinished or not parsed.valid or not set(parsed.labels) <= set(policy_ids):
        return -1.0
    predicted = parsed.labels
    if predicted == gold:
        return 1.0
    p, g = set(predicted), set(gold)
    denom = len(p) + len(g)
    fset = 2 * len(p & g) / denom if denom else 1.0
    fseq = 2 * lcs(predicted, gold) / denom if denom else 1.0
    base = .8 + .15 * fseq if p == g else -.5 + fset + .25 * fseq
    return base - .05 * min(1., max(0., (tokens - 384) / (640 - 384)))
