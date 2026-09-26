"""Plan independent model/arm jobs; execute only when explicitly requested."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .common import REPO_ROOT, external, load, save, sha256
from .scheduler import run_dynamic
from .validation import TASKS, require_complete_checkpoints, validate_eval, validate_train

TRAIN_OPTIONS = {'max_seq_len', 'per_device_batch_size', 'gradient_accumulation_steps',
                 'epochs', 'warmup_ratio', 'weight_decay', 'lora_r', 'lora_alpha', 'lora_dropout',
                 'target_profile', 'logging_steps', 'gradient_checkpointing'}
EVAL_OPTIONS = {'tasks', 'prompt_mode', 'generation_batch_size', 'choice_batch_size',
                'temperature', 'top_p', 'max_new_tokens_code', 'max_new_tokens_gsm8k'}


def options(mapping, allowed):
    if set(mapping) - allowed:
        raise ValueError('Unsupported options: ' + ', '.join(sorted(set(mapping) - allowed)))
    result = []
    for name, value in mapping.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend([flag, ','.join(value) if isinstance(value, list) else str(value)])
    return result


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Unsafe model/arm identifier')
    return name


def plan(config, manifest_path, output, seeds, eval_seed, template_date, include_base=False):
    manifest_path = external(manifest_path)
    output = external(output)
    datetime.strptime(template_date, '%d %b %Y')
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Supply unique training random states')
    if set(config) - {'models', 'arms', 'training', 'evaluation', 'data_root', 'gpus'}:
        raise ValueError('Unexpected top-level configuration key')
    manifest = load(manifest_path)
    eval_options = dict(config.get('evaluation', {}))
    tasks = eval_options.setdefault('tasks', list(TASKS))
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(TASKS):
        raise ValueError('Provide an explicit list of supported tasks')
    data_root = external(config['data_root'])
    dataset_hashes = {t: sha256(data_root / TASKS[t][1]) for t in tasks}
    code_hashes = {p.relative_to(REPO_ROOT).as_posix(): sha256(p)
                   for prefix in ('experiments', 'evaluation', 'stages/stage01_main_kodcode_lora_ablation')
                   for p in (REPO_ROOT / prefix).rglob('*.py')}
    selected = config.get('arms', list(manifest['arms']))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Provide nonempty, unique arm names')
    if not config['models']:
        raise ValueError('No models configured')
    jobs = []
    for key, model in config['models'].items():
        safe_name(key)
        if set(model) != {'path', 'prompt_mode', 'learning_rate'}:
            raise ValueError('Each model needs path, prompt_mode and learning_rate')
        if model['prompt_mode'] not in ('native', 'plain') or float(model['learning_rate']) <= 0:
            raise ValueError('Invalid model training options')
        model_path = external(model['path'])
        if not model_path.is_dir():
            raise ValueError('A locally prepared model directory is required')
        model_identity = {p.relative_to(model_path).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
                          for p in model_path.rglob('*') if p.is_file()}
        if not model_identity:
            raise ValueError('Empty model directory')
        model_eval_options = dict(eval_options)
        model_eval_options.setdefault('prompt_mode', 'chat' if model['prompt_mode'] == 'native' else 'plain')
        selections = [(s, a) for s in seeds for a in selected]
        if include_base:
            selections.insert(0, (None, 'base'))
        for seed, arm in selections:
            safe_name(arm)
            root = output / key / ('base' if seed is None else f'run_{seed}') / arm
            prefix = [sys.executable, '-m', 'experiments.category_sft.entry']
            train = None
            train_file = None
            if seed is not None:
                rec = manifest['arms'][arm]
                train_file = (manifest_path.parent / rec['path']).resolve()
                if manifest_path.parent not in train_file.parents or sha256(train_file) != rec['sha256']:
                    raise ValueError('Arm path or prepared-data hash mismatch')
                train = prefix + ['train', '--template-date', template_date,
                    '--model-key', key, '--model-path', str(model_path), '--train-file', str(train_file),
                    '--output-dir', str(root / 'train'), '--seed', str(seed), '--prompt-mode', model['prompt_mode'],
                    '--learning-rate', str(model['learning_rate']), '--save-resume', '--resume']
                train += options(config.get('training', {}), TRAIN_OPTIONS)
            evaluate = prefix + ['eval', '--template-date', template_date,
                '--model-name', f'{key}__{arm}', '--base-model-path', str(model_path),
                '--output-dir', str(root / 'eval'), '--data-root', str(data_root), '--seed', str(eval_seed)]
            if train:
                evaluate += ['--adapter-path', str(root / 'train/final_adapter')]
            evaluate += options(model_eval_options, EVAL_OPTIONS)
            jobs.append(dict(root=str(root), train=train, evaluate=evaluate, train_file=str(train_file) if train_file else None,
                train_sha256=sha256(train_file) if train_file else None, tasks=tasks, data_root=str(data_root),
                dataset_hashes=dataset_hashes, model_identity=model_identity, code_hashes=code_hashes))
    return jobs


@contextmanager
def lock(path):
    import fcntl  # The execution path is deliberately Linux-only.
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute(job, gpu):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Use independent single-device lanes, not a multi-rank launch')
    root = external(job['root'])
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with lock(root / 'run.lock'):
        contract = root / 'contract.json'
        if not contract.exists() and any(p.name != 'run.lock' for p in root.iterdir()):
            raise ValueError('Output directory has untracked artifacts; choose a new output root')
        if contract.exists() and load(contract) != job:
            raise ValueError('Refusing to reuse an output directory with a different contract')
        save(contract, job)
        for phase in ('train', 'eval'):
            cmd = job['train' if phase == 'train' else 'evaluate']
            if cmd is None:
                continue
            out = root / phase
            out.mkdir(exist_ok=True)
            receipt = out / 'verified_complete.json'

            def validate():
                if phase == 'train':
                    return validate_train(out, Path(job['train_file']))
                return validate_eval(out, Path(job['data_root']), job['tasks'])

            if receipt.exists():
                for name, digest in load(receipt).items():
                    if sha256(out / name) != digest:
                        raise ValueError('Previously completed output changed')
                validate()
                continue
            started = out / 'started.json'
            if started.exists():
                # A process may have finished just before its receipt was written.
                marker = out / ('COMPLETED' if phase == 'train' else 'metrics.json')
                if marker.is_file() and (phase == 'train' or load(marker).get('completed_at')):
                    save(receipt, {p.relative_to(out).as_posix(): sha256(p) for p in validate()})
                    continue
                if phase == 'eval':
                    raise RuntimeError('Partial evaluation preserved; use a fresh output root for another attempt')
                require_complete_checkpoints(out)
            save(started, {'command': cmd})
            with (out / 'run.log').open('a', encoding='utf-8') as log:
                subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            save(receipt, {p.relative_to(out).as_posix(): sha256(p) for p in validate()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--eval-seed', type=int, required=True)
    p.add_argument('--template-date', required=True)
    p.add_argument('--include-base', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    config = load(args.config)
    jobs = plan(config, args.manifest, args.output_dir, args.seeds, args.eval_seed, args.template_date, args.include_base)
    if not args.execute:
        print(json.dumps({'plan_only': True, 'jobs': jobs}, indent=2))
        return
    if sys.platform != 'linux':
        raise RuntimeError('GPU execution and sandboxing require Linux')
    devices = [str(x) for x in config.get('gpus', ['0', '1', '2', '3'])]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError('GPU lanes must be nonempty and unique')
    from .sandbox_runner import self_test
    self_test()
    run_dynamic(jobs, lambda job, lane: execute(job, devices[lane]), workers=len(devices),
                on_event=lambda state, lane, job: print(state, devices[lane], job['root'], flush=True))


if __name__ == '__main__':
    main()
