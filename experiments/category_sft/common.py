"""Small file helpers shared by the portable runners."""
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def external(path):
    path = Path(path).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError('Datasets and run outputs must be outside the checkout')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def read_rows(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    if path.suffix == '.json':
        value = load(path)
        if not isinstance(value, list):
            raise ValueError('Expected a JSON array')
        return value
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]
