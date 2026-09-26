"""Validate the source-only checkout without importing GPU dependencies."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def validate():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                                    cwd=ROOT, text=True).splitlines()
    files = sorted({ROOT / name for name in names if (ROOT / name).is_file()})
    forbidden_dirs = {'analysis', 'reference_results', 'metadata', 'plan', 'progress', 'docs'}
    forbidden_exts = {'.csv', '.tsv', '.jsonl', '.parquet', '.png', '.jpg', '.pdf', '.zip', '.safetensors', '.pt'}
    for path in files:
        rel = path.relative_to(ROOT)
        if rel.parts[0] in forbidden_dirs or path.suffix in forbidden_exts:
            raise AssertionError(f'Non-code release artifact: {rel}')
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.py':
            ast.parse(text, filename=str(rel))
        if path.suffix == '.json':
            json.loads(text)
        if re.search(r'(?<![A-Za-z0-9])202\d{5}(?![A-Za-z0-9])', text):
            raise AssertionError(f'Historical run identifier: {rel}')
        if re.search(r'/(?:mnt/aoss-[\w-]+|data/yuzheng)/|[A-Z]:[/\\](?:Users|桌面)', text):
            raise AssertionError(f'Private machine path: {rel}')
        if re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}', text):
            raise AssertionError(f'Credential-like content: {rel}')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for ref in re.findall(r'`((?:configs|dapo|evaluation|experiments|stages|tests)/[^` ]+)`', readme):
        if not (ROOT / ref).exists():
            raise AssertionError(f'Broken README path: {ref}')
    return len(files)


if __name__ == '__main__':
    print(f'Validated {validate()} source/configuration files')
