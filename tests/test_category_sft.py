import ast
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from experiments.category_sft.build_data import allocation, build_arms, select_ids
from experiments.category_sft.common import REPO_ROOT, external, save, sha256
from experiments.category_sft.run import options, plan, safe_name
from experiments.category_sft.scheduler import run_dynamic
from experiments.category_sft.template import bind
from experiments.category_sft.trainer_compat import OLD, NEW, patched_source
from experiments.category_sft.validation import result_section, validate_eval, require_complete_checkpoints
from validate_repository import has_private_machine_path, validate


def rows():
    return [dict(problem_id=f'{cat}-{i}', primary_category=cat, prompt='question', response='answer')
            for cat, n in [('a', 3), ('b', 5), ('c', 7)] for i in range(n)]


class DataTests(unittest.TestCase):
    def test_order_independent(self):
        recipe = [dict(name='one', mode='single', categories=['b'])]
        self.assertEqual(build_arms(rows(), recipe, 7), build_arms(list(reversed(rows())), recipe, 7))

    def test_single_nested_in_pair(self):
        recipe = [dict(name='one', mode='single', categories=['b']),
                  dict(name='pair', mode='mixture', categories=['a', 'b'])]
        arms = build_arms(rows(), recipe, 7)
        self.assertEqual(len(arms['one']), 3)
        self.assertEqual(len(arms['pair']), 6)
        self.assertTrue({r['problem_id'] for r in arms['one']} <= {r['problem_id'] for r in arms['pair']})

    def test_full_single(self):
        arm = build_arms(rows(), [dict(name='one', mode='single', categories=['c'], per_category='all')], 7)
        self.assertEqual(len(arm['one']), 7)

    def test_leave_out_and_control(self):
        result = build_arms(rows(), [dict(name='drop', mode='leave_out', categories=['a']),
                                   dict(name='random', mode='random', total=8)], 7)
        self.assertEqual(len(result['drop']), 8)
        self.assertEqual(len(result['random']), 8)
        self.assertNotIn('a', {r['primary_category'] for r in result['drop']})

    def test_allocation(self):
        self.assertEqual(sum(allocation({'a': 5, 'b': 7}, 9, 7).values()), 9)
        self.assertEqual(allocation({'a': 2, 'b': 8}, 10, 7, True), {'a': 2, 'b': 8})
        with self.assertRaises(ValueError):
            allocation({'a': 1, 'b': 9}, 8, 7)

    def test_bad_recipe(self):
        for recipe in ([dict(name='../bad', mode='full')], [dict(name='base', mode='full')],
                       [dict(name='a', mode='single', categories=['a', 'b'])],
                       [dict(name='a', mode='single', categories=['a'], per_category=8)]):
            with self.assertRaises(ValueError):
                build_arms(rows(), recipe, 7)

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            build_arms(rows() + rows()[:1], [dict(name='all', mode='full')], 7)
        with self.assertRaises(ValueError):
            select_ids(['a', 'a'], 1, 7, 'x')

    def test_builder_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / 'source.json', rows())
            save(root / 'valid.json', [{'problem_id': 'held-out'}])
            save(root / 'recipe.json', [dict(name='mix', mode='mixture')])
            cmd = [sys.executable, '-m', 'experiments.category_sft.build_data', '--source', str(root / 'source.json'),
                   '--validation', str(root / 'valid.json'), '--recipe', str(root / 'recipe.json'),
                   '--output-dir', str(root / 'prepared'), '--seed', '7']
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True)
            manifest = json.loads((root / 'prepared/manifest.json').read_text())
            self.assertEqual(manifest['arms']['mix']['rows'], 9)
            self.assertEqual(manifest['arms']['mix']['sha256'], sha256(root / 'prepared/mix.jsonl'))
            self.assertNotEqual(subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True).returncode, 0)

    def test_overlap_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / 'source.json', rows())
            save(root / 'valid.json', rows()[:1])
            save(root / 'recipe.json', [dict(name='mix', mode='mixture')])
            with patch.object(sys, 'argv', ['build', '--source', str(root / 'source.json'), '--validation', str(root / 'valid.json'),
                 '--recipe', str(root / 'recipe.json'), '--output-dir', str(root / 'out'), '--seed', '7']):
                from experiments.category_sft.build_data import main
                with self.assertRaises(ValueError):
                    main()
            self.assertFalse((root / 'out').exists())


class RuntimeTests(unittest.TestCase):
    def test_planned_flags_accepted_by_core_parsers(self):
        # Exercise the real argument definitions without importing GPU libraries.
        paths = [REPO_ROOT / 'stages/stage01_main_kodcode_lora_ablation/train_lora.py',
                 REPO_ROOT / 'evaluation/paper_suite.py']
        commands = [
            ['--model-key', 'model', '--model-path', '/external/model', '--train-file', '/external/train',
             '--output-dir', '/external/out', '--seed', '7', '--prompt-mode', 'native',
             '--learning-rate', '0.00002', '--save-resume', '--resume', '--gradient-checkpointing'],
            ['--model-name', 'model', '--base-model-path', '/external/model', '--data-root', '/external/data',
             '--output-dir', '/external/out', '--seed', '7', '--max-new-tokens-gsm8k', '512'],
        ]
        for path, argv in zip(paths, commands):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            namespace = dict(argparse=argparse, Path=Path, _DATA_TEST_ENV=None)
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            namespace[target.id] = value
            func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
            exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), 'exec'), namespace)
            with patch.object(sys, 'argv', ['core', *argv]):
                self.assertEqual(namespace['parse_args']().seed, 7)
            no_seed = argv.copy()
            pos = no_seed.index('--seed')
            del no_seed[pos:pos+2]
            with patch.object(sys, 'argv', ['core', *no_seed]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    namespace['parse_args']()

    def test_scheduler_exclusive_lanes(self):
        active, finished = set(), []
        lock = threading.Lock()
        def run(job, lane):
            with lock:
                self.assertNotIn(lane, active)
                active.add(lane)
            with lock:
                active.remove(lane)
                finished.append(job)
        self.assertEqual(run_dynamic(range(19), run, workers=4), 19)
        self.assertEqual(sorted(finished), list(range(19)))

    def test_scheduler_stops_dispatching(self):
        seen = []
        def run(job, lane):
            seen.append(job)
            raise ValueError('stop')
        with self.assertRaises(ValueError):
            run_dynamic(range(5), run, workers=1)
        self.assertEqual(seen, [0])

    def test_fixed_template_date(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kw):
                return kw
        obj = bind(Tokenizer(), '01 Jan 2000')
        self.assertEqual(obj.apply_chat_template([])['date_string'], '01 Jan 2000')
        with self.assertRaises(ValueError):
            obj.apply_chat_template([], date_string='different')

    def test_process_local_patch_shape(self):
        source = 'def _inner_training_loop(self):\n    ' + OLD + '\n'
        before, after = patched_source(source)
        self.assertIn(OLD, before)
        self.assertIn(NEW, after)
        with self.assertRaises(RuntimeError):
            patched_source(after)

    def test_safety_guards(self):
        with self.assertRaises(ValueError):
            external(REPO_ROOT / 'data')
        with self.assertRaises(ValueError):
            safe_name('../outside')
        with self.assertRaises(ValueError):
            options({'seed': 7}, {'epochs'})

    def test_missing_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                require_complete_checkpoints(Path(temp))

    def test_metadata_not_task_completion(self):
        with self.assertRaises(ValueError):
            result_section({'task_max_new_tokens': {'gsm8k': 256}}, 'gsm8k')

    def test_eval_count_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data/main_test').mkdir(parents=True)
            (root / 'out').mkdir()
            (root / 'data/main_test/arc_challenge.jsonl').write_text('{"id":"x"}\n', encoding='utf-8')
            (root / 'out/arc_challenge_results.jsonl').write_text('{"id":"x","passed":true}\n', encoding='utf-8')
            metrics = dict(completed_at='done', scoreable_dataset_names=['arc_challenge'],
                           main={'arc_challenge': dict(n=1, correct=1, accuracy=1)})
            save(root / 'out/metrics.json', metrics)
            self.assertEqual(len(validate_eval(root / 'out', root / 'data', ['arc_challenge'])), 2)
            metrics['main']['arc_challenge']['n'] = 2
            save(root / 'out/metrics.json', metrics)
            with self.assertRaises(ValueError):
                validate_eval(root / 'out', root / 'data', ['arc_challenge'])

    def test_plan_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'model').mkdir()
            save(root / 'model/config.json', {'model_type': 'test'})
            (root / 'data/main_test').mkdir(parents=True)
            (root / 'data/main_test/arc_challenge.jsonl').write_text('{"id":"x"}\n', encoding='utf-8')
            train = root / 'train.jsonl'
            train.write_text('{"problem_id":"x","prompt":"p","response":"r"}\n', encoding='utf-8')
            save(root / 'manifest.json', {'arms': {'one': {'path': train.name, 'sha256': sha256(train)}}})
            config = dict(models={'test': dict(path=str(root / 'model'), prompt_mode='native', learning_rate=1e-5)},
                          data_root=str(root / 'data'), evaluation={'tasks': ['arc_challenge']})
            jobs = plan(config, root / 'manifest.json', root / 'out', [7, 11], 3, '01 Jan 2000', True)
            self.assertEqual(len(jobs), 3)
            self.assertIsNone(jobs[0]['train'])
            self.assertIn('--resume', jobs[1]['train'])
            self.assertNotIn('--resume', jobs[1]['evaluate'])
            self.assertFalse((root / 'out').exists())

    def test_source_only_repository(self):
        self.assertGreater(validate(), 30)

    def test_private_paths_are_detected_without_named_users(self):
        for root in ('home', 'Users', 'data', 'mnt', 'root'):
            candidate = '/'.join(('', root, 'example-user', 'project'))
            with self.subTest(root=root):
                self.assertTrue(has_private_machine_path(candidate))
        for separator in ('/', chr(92)):
            candidate = 'X:' + separator + 'example-project' + separator + 'data'
            self.assertTrue(has_private_machine_path(candidate))
            self.assertTrue(has_private_machine_path('~' + separator + 'project'))

    def test_generic_paths_and_upstream_links_are_allowed(self):
        for candidate in ('/path/to/models', '${CG_DATA_ROOT}/train.jsonl',
                          'data/train.jsonl', '/usr/bin/python3', '/tmp/sandbox',
                          'https://example.org/data/project', 'no host data/home/proc'):
            with self.subTest(candidate=candidate):
                self.assertFalse(has_private_machine_path(candidate))
