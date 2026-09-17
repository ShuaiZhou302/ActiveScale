#!/usr/bin/env python3
"""Build a minimal, auditable inference-checkpoint release directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

from safetensors import safe_open

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _json_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _copy_file(source: pathlib.Path, target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_release(
    checkpoint: pathlib.Path,
    output: pathlib.Path,
    config: pathlib.Path,
    metadata: pathlib.Path,
    release_type: str = "inference",
    model_card: pathlib.Path | None = None,
) -> None:
    if release_type not in {"inference", "initialization"}:
        raise ValueError(f"Unsupported release type: {release_type}")

    model = checkpoint / "model.safetensors"
    assets = checkpoint / "assets"
    if not model.is_file():
        raise FileNotFoundError(f"Missing checkpoint weights: {model}")
    if release_type == "inference" and (not assets.is_dir() or not any(assets.rglob("*"))):
        raise FileNotFoundError(f"Missing checkpoint normalization assets: {assets}")
    _json_object(config)
    release_metadata = _json_object(metadata)
    required_metadata = {"activescale_commit", "training_config", "checkpoint_step"}
    missing_metadata = sorted(required_metadata - release_metadata.keys())
    if missing_metadata:
        raise ValueError(f"Release metadata is missing keys: {', '.join(missing_metadata)}")

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with safe_open(model, framework="pt", device="cpu") as weights:
        if not weights.keys():
            raise ValueError(f"Safetensors file has no tensors: {model}")

    _copy_file(model, output / "model.safetensors")
    if assets.is_dir() and any(assets.rglob("*")):
        shutil.copytree(assets, output / "assets", dirs_exist_ok=True)
    _copy_file(config, output / "config.json")
    _copy_file(metadata, output / "metadata.json")
    _copy_file(model_card or REPO_ROOT / "MODEL_CARD_TEMPLATE.md", output / "README.md")
    for filename in ("LICENSE", "LICENSE_GEMMA.txt"):
        _copy_file(REPO_ROOT / filename, output / filename)

    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    checksum_lines = [f"{_sha256(path)}  {path.relative_to(output).as_posix()}" for path in artifacts]
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    print(f"Prepared {len(artifacts)} files in {output}")
    print("Review README.md and metadata.json, then run a clean load/inference smoke test before upload.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model-card",
        type=pathlib.Path,
        help="Completed model card to include instead of MODEL_CARD_TEMPLATE.md.",
    )
    parser.add_argument(
        "--release-type",
        choices=("inference", "initialization"),
        default="inference",
        help="Inference releases require normalization assets; initialization releases do not.",
    )
    args = parser.parse_args()
    prepare_release(
        args.checkpoint.resolve(),
        args.output.resolve(),
        args.config.resolve(),
        args.metadata.resolve(),
        args.release_type,
        args.model_card.resolve() if args.model_card else None,
    )


if __name__ == "__main__":
    main()
