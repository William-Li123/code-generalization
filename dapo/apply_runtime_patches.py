#!/usr/bin/env python3
"""Install and verify the archived VERL compatibility patches.

The historical jobs copied five modules into a job-local Python environment
before importing VERL.  This utility reproduces that operation, but refuses to
touch an interpreter whose site-packages directory is outside the explicitly
provided environment.  Original files are backed up once and a hash manifest
is written into that external environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATCH_ROOT = ROOT / "runtime_patches"
VERL_COMMIT = "e0f4dc2ccdd51e8e445a683b828090577c625534"
PATCHES = {
    "device.py": (
        "verl/utils/device.py",
        "9083e353dc6281ed4705bae7e16acdf0920f4c378364019b8ad07ccfcfea2995",
    ),
    "attention_utils.py": (
        "verl/utils/attention_utils.py",
        "2d3c03ef7bba66f2c37040abd90b7cd1a7c0eff90c82ae0af73c3f31a2375585",
    ),
    "chat_template.py": (
        "verl/utils/chat_template.py",
        "ceb0da8907272603fa35dc1012664e5bc6ec62ce279319d7574bcef70f1a7492",
    ),
    "engine_workers.py": (
        "verl/workers/engine_workers.py",
        "050595c6f8cbe948bf1780bcbd7bc336df65d8d185d16e2f79d6593facfb2f15",
    ),
    "fsdp_transformer_impl.py": (
        "verl/workers/engine/fsdp/transformer_impl.py",
        "86e31d7378f8c65373d7689beec2d71ac1978a378dd58487d1a5db2aca8b3133",
    ),
}
VLLM_UTILS = "verl/workers/rollout/vllm_rollout/utils.py"
VLLM_OLD = 'VLLM_LORA_PATH = "simon_lora_path"'
VLLM_NEW = 'VLLM_LORA_PATH = "/tmp/simon_lora_path"'
MANIFEST_NAME = ".code_generalization_runtime_patch_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def interpreter_for(environment: Path) -> Path:
    python = environment / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"isolated environment has no bin/python: {environment}")
    return python


def site_packages(environment: Path, python: Path) -> Path:
    command = [
        str(python),
        "-c",
        "import json,site; print(json.dumps(site.getsitepackages()))",
    ]
    raw = subprocess.check_output(command, text=True)
    candidates = [Path(item).resolve() for item in json.loads(raw)]
    matches = [path for path in candidates if path.name == "site-packages" and inside(path, environment)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one site-packages inside {environment}, got {candidates}"
        )
    return matches[0]


def recipe_commit(recipe: Path, allow_unverified: bool) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(recipe), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        if allow_unverified:
            return "unverified"
        raise RuntimeError(f"cannot identify VERL recipe commit under {recipe}") from exc
    if commit != VERL_COMMIT and not allow_unverified:
        raise RuntimeError(f"VERL recipe commit {commit} != frozen {VERL_COMMIT}")
    return commit


def backup_once(path: Path) -> Path:
    backup = path.with_name(path.name + ".code_generalization_original")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def verify_sources() -> None:
    for source_name, (_, expected) in PATCHES.items():
        source = PATCH_ROOT / source_name
        if not source.is_file() or sha256(source) != expected:
            raise RuntimeError(f"archived patch identity mismatch: {source}")


def apply(site: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for source_name, (relative, expected) in PATCHES.items():
        source = PATCH_ROOT / source_name
        target = site / relative
        if not target.is_file():
            raise FileNotFoundError(f"VERL target is absent: {target}")
        backup = backup_once(target)
        shutil.copy2(source, target)
        if sha256(target) != expected:
            raise RuntimeError(f"failed to install patch: {target}")
        records.append(
            {
                "source": source_name,
                "target": relative,
                "source_sha256": expected,
                "original_sha256": sha256(backup),
                "patched_sha256": sha256(target),
            }
        )

    vllm = site / VLLM_UTILS
    if not vllm.is_file():
        raise FileNotFoundError(f"VERL vLLM compatibility target is absent: {vllm}")
    backup = backup_once(vllm)
    text = vllm.read_text(encoding="utf-8")
    if VLLM_NEW not in text:
        if VLLM_OLD not in text:
            raise RuntimeError(f"neither expected vLLM path assignment is present in {vllm}")
        vllm.write_text(text.replace(VLLM_OLD, VLLM_NEW, 1), encoding="utf-8")
    records.append(
        {
            "source": "controlled_text_replacement",
            "target": VLLM_UTILS,
            "original_sha256": sha256(backup),
            "patched_sha256": sha256(vllm),
        }
    )
    return records


def check(site: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for source_name, (relative, expected) in PATCHES.items():
        target = site / relative
        actual = sha256(target) if target.is_file() else "missing"
        if actual != expected:
            raise RuntimeError(f"VERL runtime patch mismatch: {relative}: {actual} != {expected}")
        records.append(
            {
                "source": source_name,
                "target": relative,
                "source_sha256": expected,
                "patched_sha256": actual,
            }
        )
    vllm = site / VLLM_UTILS
    if not vllm.is_file() or VLLM_NEW not in vllm.read_text(encoding="utf-8"):
        raise RuntimeError(f"vLLM LoRA path compatibility edit is absent: {vllm}")
    records.append(
        {
            "source": "controlled_text_replacement",
            "target": VLLM_UTILS,
            "patched_sha256": sha256(vllm),
        }
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "check"))
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--allow-unverified-recipe", action="store_true")
    args = parser.parse_args()

    environment = args.environment.expanduser().resolve()
    recipe = args.recipe.expanduser().resolve()
    if environment in {Path("/"), Path.home().resolve()}:
        raise ValueError("refusing a broad or home-directory environment target")
    python = interpreter_for(environment)
    site = site_packages(environment, python)
    verify_sources()
    commit = recipe_commit(recipe, args.allow_unverified_recipe)
    records = apply(site) if args.action == "apply" else check(site)
    payload = {
        "schema_version": 1,
        "action": args.action,
        "environment": str(environment),
        "site_packages": str(site),
        "verl_recipe_commit": commit,
        "frozen_verl_recipe_commit": VERL_COMMIT,
        "patches": records,
    }
    manifest = environment / MANIFEST_NAME
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
