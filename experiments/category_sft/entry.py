"""Process-local compatibility and isolation wrapper around the existing cores."""
import argparse
import importlib.util
import sys

from .common import REPO_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['train', 'eval'])
    p.add_argument('--template-date', required=True, help='Native template date, formatted DD Mon YYYY')
    args, remaining = p.parse_known_args()
    from .template import install
    install(args.template_date)
    if args.phase == 'train':
        import transformers
        if transformers.__version__ == '4.52.4':
            from .trainer_compat import install as install_compat
            install_compat()
        path = REPO_ROOT / 'stages/stage01_main_kodcode_lora_ablation/train_lora.py'
    else:
        if '--resume' in remaining:
            raise ValueError('Partial evaluation resume is disabled; use a fresh output directory')
        path = REPO_ROOT / 'evaluation/paper_suite.py'
    spec = importlib.util.spec_from_file_location('runtime_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.phase == 'eval':
        from .sandbox_runner import run_python, self_test
        self_test()  # Fail before loading a model if isolation is unavailable.

        def isolated(script, timeout_seconds=8):
            rc, out, err = run_python(script, timeout_seconds)
            return rc == 0, 'timeout' if rc == 124 else (out + '\n' + err).strip()

        module.run_python = isolated
    sys.argv = [str(path), *remaining]
    module.main()


if __name__ == '__main__':
    main()
