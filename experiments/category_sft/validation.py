"""Check saved output completeness without changing or recomputing predictions."""
import math

from .common import load, read_rows, sha256

TASKS = {
    'arc_challenge': ('main', 'main_test/arc_challenge.jsonl', 'arc_challenge'),
    'scienceqa': ('diagnostic', 'diagnostic_test/scienceqa.jsonl', 'scienceqa'),
    'legalbench': ('main', 'main_test/legalbench_rule_application_test.jsonl', 'legalbench'),
    'gsm8k': ('diagnostic', 'diagnostic_test/gsm8k.jsonl', 'gsm8k'),
    'math500_medium': ('main', 'main_test/math500_medium.jsonl', 'math500_medium'),
    'math500_high_level': ('hard', 'hard_test/math500_high_level.jsonl', 'math500_high_level'),
    'finqa': ('main', 'main_test/finqa_official_test.jsonl', 'finqa_official_test'),
    'medcalc': ('main', 'main_test/medcalc_bench_verified_test.jsonl', 'medcalc_bench_verified_test'),
    'humaneval': ('main', 'main_test/humaneval.jsonl', 'humaneval'),
    'mbpp_plus': ('main', 'main_test/mbpp_plus.jsonl', 'mbpp_plus'),
    'mbpp_simple': ('diagnostic', 'diagnostic_test/mbpp_simple.jsonl', 'mbpp_simple'),
}


def result_section(metrics, task):
    matches = [(g, metrics[g][task]) for g in ('main', 'hard', 'diagnostic')
               if isinstance(metrics.get(g), dict) and task in metrics[g]]
    if len(matches) != 1 or matches[0][0] != TASKS[task][0] or not isinstance(matches[0][1], dict):
        raise ValueError(f'Missing, duplicate or misplaced result: {task}')
    return matches[0][1]


def validate_eval(out, data_root, tasks):
    metrics = load(out / 'metrics.json')
    if not metrics.get('completed_at') or set(metrics.get('scoreable_dataset_names', [])) != set(tasks):
        raise ValueError('Incomplete evaluation suite')
    files = [out / 'metrics.json']
    for task in tasks:
        _, source, basename = TASKS[task]
        truth = read_rows(data_root / source)
        path = out / f'{basename}_results.jsonl'
        pred = read_rows(path)
        ids = [r.get('id') for r in pred]
        if not ids or None in ids or len(set(ids)) != len(ids) or ids != [r['id'] for r in truth]:
            raise ValueError(f'Incorrect IDs or order: {task}')
        for p, t in zip(pred, truth):
            if task == 'finqa' and not str(t.get('target', '')).strip():
                if p.get('passed') is not None or p.get('status') != 'skipped_empty_target':
                    raise ValueError('Incorrect unscored FinQA row')
            elif type(p.get('passed')) is not bool:
                raise ValueError(f'Missing correctness value: {task}')
        n = sum(type(r.get('passed')) is bool for r in pred)
        correct = sum(r.get('passed') is True for r in pred)
        section = result_section(metrics, task)
        score = section.get('pass_at_1', section.get('accuracy'))
        if (n <= 0 or section.get('n') != n or section.get('correct') != correct
                or not isinstance(score, (int, float)) or not math.isclose(score, correct / n, abs_tol=1e-12)):
            raise ValueError(f'Summary disagrees with saved predictions: {task}')
        files.append(path)
    return files


def validate_train(out, train_file):
    summary = load(out / 'training_summary.json')
    if (not (out / 'COMPLETED').is_file() or summary.get('completed_steps', 0) <= 0
            or not math.isfinite(summary.get('training_loss', float('nan')))
            or summary.get('train_file_sha256') != sha256(train_file)):
        raise ValueError('Incomplete or mismatched training output')
    files = [out / 'training_summary.json', out / 'COMPLETED',
             out / 'final_adapter/adapter_model.safetensors', out / 'final_adapter/adapter_config.json']
    if any(not f.is_file() or not f.stat().st_size for f in files):
        raise ValueError('Missing adapter files')
    return files


def require_complete_checkpoints(out):
    required = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                'adapter_model.safetensors', 'adapter_config.json')
    checkpoints = list((out / 'trainer_checkpoints').glob('checkpoint-*'))
    if not checkpoints:
        raise ValueError('No checkpoint to resume; preserve this run and choose a fresh output directory')
    for cp in checkpoints:
        if cp.is_symlink() or not cp.name.removeprefix('checkpoint-').isdigit():
            raise ValueError('Unexpected checkpoint path')
        if any(not (cp / name).is_file() for name in required):
            raise ValueError(f'Incomplete checkpoint preserved, not resumed: {cp}')
