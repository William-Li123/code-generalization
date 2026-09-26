"""Build single-category, budget-matched and selected-category SFT datasets.

Selection is independent of source row order. Recipes specify categories,
not model-specific rankings inferred from held-out test results.
"""
import argparse
from collections import Counter
import hashlib
import json
import re

from .common import external, load, read_rows, save, sha256


def rank(seed, purpose, category, problem_id):
    return hashlib.sha256(f'{seed}|{purpose}|{category}|{problem_id}'.encode()).hexdigest()


def select_ids(ids, n, seed, category, purpose='single_pool'):
    ids = [str(x) for x in ids]
    if len(ids) != len(set(ids)) or not 0 <= n <= len(ids):
        raise ValueError('Duplicate IDs or invalid sampling budget')
    return sorted(sorted(ids, key=lambda x: (rank(seed, purpose, category, x), x))[:n])


def allocation(counts, n, seed, proportional=False):
    if not counts or n < 1 or n > sum(counts.values()) or min(counts.values()) < 1:
        raise ValueError('Invalid category capacities or total budget')
    weights = counts if proportional else dict.fromkeys(counts, 1)
    total = sum(weights.values())
    result = {c: n * w // total for c, w in weights.items()}
    order = sorted(weights, key=lambda c: (-(n * weights[c] % total), rank(seed, 'allocation', c, '')))
    for c in order[:n - sum(result.values())]:
        result[c] += 1
    if any(result[c] > counts[c] for c in counts):
        raise ValueError('Balanced allocation exceeds a category capacity; reduce the budget')
    return result


def build_arms(rows, recipe, seed):
    required = ('problem_id', 'primary_category', 'prompt', 'response')
    if not rows or any(any(k not in r or r[k] is None for k in required) for r in rows):
        raise ValueError(f'Every training row requires {required}')
    if any(not isinstance(r[k], str) or not r[k].strip() for r in rows for k in ('prompt', 'response')):
        raise ValueError('Prompts and responses must be nonempty strings')
    by_id = {str(r['problem_id']): r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Training problem IDs must be unique')
    pools = {}
    for pid, row in by_id.items():
        cat = str(row['primary_category'])
        if not cat:
            raise ValueError('Empty category')
        pools.setdefault(cat, []).append(pid)
    smallest = min(map(len, pools.values()))
    arms = {}
    for spec in recipe:
        name, mode = spec['name'], spec['mode']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in arms or name == 'base':
            raise ValueError('Arm names must be unique safe names other than base')
        cats = spec.get('categories', sorted(pools))
        if not cats or len(set(cats)) != len(cats) or not set(cats) <= set(pools):
            raise ValueError('Unknown or repeated category')
        if mode == 'full':
            chosen = sorted(by_id)
        elif mode == 'random':
            chosen = select_ids(by_id, int(spec['total']), seed, 'all', name)
        elif mode == 'leave_out':
            candidates = [p for c, ids in pools.items() if c not in cats for p in ids]
            # Default matches the smallest leave-one-category-out pool.
            n = int(spec.get('total', len(rows) - max(map(len, pools.values()))))
            chosen = select_ids(candidates, n, seed, ','.join(sorted(cats)), name)
        elif mode in ('single', 'mixture'):
            if mode == 'single' and len(cats) != 1:
                raise ValueError('A single-category arm requires exactly one category')
            if 'total' in spec and 'per_category' in spec:
                raise ValueError('Choose total or per_category, not both')
            counts = {c: len(pools[c]) for c in cats}
            if 'total' in spec:
                take = allocation(counts, int(spec['total']), seed, spec.get('proportional', False))
            else:
                value = spec.get('per_category', 'minimum')
                take = {c: len(pools[c]) if value == 'all' else smallest if value == 'minimum' else int(value) for c in cats}
            # Shared per-category ranking makes matched single arms nested in mixtures.
            chosen = sorted(p for c, n in take.items() for p in select_ids(pools[c], n, seed, c))
        else:
            raise ValueError(f'Unsupported mode: {mode}')
        if not chosen:
            raise ValueError('Empty arm')
        arms[name] = [dict(by_id[p], problem_id=p) for p in chosen]
    if not arms:
        raise ValueError('Recipe must contain at least one arm')
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--recipe', required=True)
    p.add_argument('--validation')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    source = external(args.source)
    out = external(args.output_dir)
    rows = read_rows(source)
    validation = read_rows(external(args.validation)) if args.validation else []
    if args.validation:
        train_ids = {str(r['problem_id']) for r in rows}
        val_ids = [str(r['problem_id']) for r in validation]
        if not val_ids or len(set(val_ids)) != len(val_ids) or train_ids.intersection(val_ids):
            raise ValueError('Validation must be nonempty, unique and disjoint from training')
    recipe = load(args.recipe)
    arms = build_arms(rows, recipe, args.seed)
    # Never replace existing prepared datasets, even after an interrupted build.
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(source_sha256=sha256(source), sampling_seed=args.seed,
                    validation_overlap_checked=bool(args.validation), arms={})
    for name, selected in arms.items():
        dest = out / f'{name}.jsonl'
        with dest.open('x', encoding='utf-8') as handle:
            for row in selected:
                # Training needs only these fields; preserve labels and IDs for checks.
                record = {k: str(row[k]) for k in ('problem_id', 'primary_category', 'prompt', 'response')}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        manifest['arms'][name] = dict(path=dest.name, rows=len(selected), sha256=sha256(dest),
            categories=dict(Counter(str(r['primary_category']) for r in selected)))
    save(out / 'manifest.json', manifest)
    print(json.dumps({name: len(value) for name, value in arms.items()}, indent=2))


if __name__ == '__main__':
    main()
