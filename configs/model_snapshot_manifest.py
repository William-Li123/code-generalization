#!/usr/bin/env python3
"""Build or verify a content-addressed manifest for external model snapshots.

Hub revisions were not preserved in the historical copies.  A complete file
manifest (including all weight shards) is therefore the strongest available
identity for bitwise model inputs. Hidden caches and VCS metadata are excluded
because they are not loaded by Transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXCLUDED_PARTS = {".cache", ".git", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def snapshot(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise FileNotFoundError(f"not a Transformers model snapshot: {root}")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files_under(root)
    ]
    weight_files = [
        entry
        for entry in files
        if Path(entry["path"]).suffix.lower() in {".safetensors", ".bin"}
    ]
    return {
        "folder": root.name,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(entry["bytes"] for entry in weight_files),
    }


def load_paths(path: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("--model-paths must be a non-empty JSON object")
    paths = {str(key): Path(value).expanduser().resolve() for key, value in raw.items()}
    folders = [value.name for value in paths.values()]
    if len(folders) != len(set(folders)):
        raise ValueError("model paths must have unique snapshot folder names")
    return paths


def build(paths: dict[str, Path]) -> dict[str, Any]:
    models = {}
    for key, root in sorted(paths.items()):
        print(f"hashing {key}: {root}", flush=True)
        models[key] = snapshot(root)
    return {
        "schema_version": 1,
        "identity": "SHA256 of every runtime file, including all model weight shards",
        "excluded_runtime_irrelevant_parts": sorted(EXCLUDED_PARTS),
        "models": models,
    }


def verify(paths: dict[str, Path], reference: dict[str, Any]) -> None:
    expected = reference.get("models", {})
    if set(paths) != set(expected):
        raise ValueError(
            f"model keys differ: missing={sorted(set(expected) - set(paths))}, "
            f"extra={sorted(set(paths) - set(expected))}"
        )
    actual = build(paths)
    if actual["models"] != expected:
        for key in sorted(paths):
            if actual["models"][key] != expected[key]:
                raise ValueError(f"model snapshot differs from frozen manifest: {key}")
        raise ValueError("model snapshot manifest differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--model-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    paths = load_paths(args.model_paths.resolve())
    if args.action == "build":
        if args.output is None:
            raise ValueError("build requires --output")
        payload = build(paths)
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output.resolve()}")
        return
    if args.manifest is None:
        raise ValueError("verify requires --manifest")
    reference = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    verify(paths, reference)
    print("model snapshot verification: ok")


if __name__ == "__main__":
    main()
