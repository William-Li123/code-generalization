"""Explicit, process-local Transformers 4.52.4 resume-tail compatibility fix.

It changes one condition only; installed site-packages remain untouched.
"""
import ast
import hashlib
import inspect
import json
import linecache
import textwrap


VERSION = '4.52.4'
ISSUE = 'https://github.com/huggingface/transformers/issues/38939'
SOURCE_URL = ('https://raw.githubusercontent.com/huggingface/transformers/'
              'v4.52.4/src/transformers/trainer.py')
OLD = ('do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 '
       'or (step + 1) == steps_in_epoch')
NEW = ('do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 '
       'or (step + 1 + steps_skipped) == steps_in_epoch')
MARKER = '_codelogic_resume_tail_compat'


def source_sha(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def patched_source(source):
    """Return the one-line replacement, rejecting an unexpected source shape."""
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != '_inner_training_loop'
            or tree.body[0].decorator_list):
        raise RuntimeError('Expected one undecorated Trainer._inner_training_loop definition')
    if source.count(OLD) != 1 or NEW in source:
        raise RuntimeError('Expected the unpatched resume-tail condition exactly once')
    result = source.replace(OLD, NEW, 1)
    ast.parse(result)
    return source, result


def install():
    """Patch the live Trainer class only, and return a reproducibility receipt."""
    import transformers
    from transformers import Trainer

    if transformers.__version__ != VERSION:
        raise RuntimeError(f'Resume-tail patch requires transformers=={VERSION}, '
                           f'found {transformers.__version__}')
    original = Trainer._inner_training_loop
    installed = getattr(original, MARKER, None)
    if installed is not None:
        current = textwrap.dedent(inspect.getsource(original))
        if source_sha(current) != installed['patched_function_sha256']:
            raise RuntimeError('Installed resume-tail patch has changed unexpectedly')
        return dict(installed)

    before, after = patched_source(inspect.getsource(original))
    receipt = {
        'patch': 'codelogic_transformers_4_52_4_resume_tail_v1',
        'transformers': VERSION,
        'scope': 'process-local Trainer._inner_training_loop; no site-packages edits',
        'upstream_source': SOURCE_URL,
        'upstream_issue': ISSUE,
        'original_function_sha256': source_sha(before),
        'patched_function_sha256': source_sha(after),
        'old_condition': OLD,
        'new_condition': NEW,
        'changed_occurrences': 1,
        'effect': 'Include skipped microbatches when deciding the final optimizer update on resume.',
    }
    filename = f"<codelogic-resume-tail-{receipt['patched_function_sha256'][:16]}>"
    linecache.cache[filename] = (len(after), None, after.splitlines(keepends=True), filename)
    namespace = {}
    # Defining this single function uses the original module globals. No copy of
    # Trainer state or loss/optimizer implementation is introduced.
    exec(compile(after, filename, 'exec'), original.__globals__, namespace)
    replacement = namespace['_inner_training_loop']
    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__
    setattr(replacement, MARKER, dict(receipt))
    Trainer._inner_training_loop = replacement
    print('CODELOGIC_TRAINER_COMPAT ' + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt
