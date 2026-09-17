#!/usr/bin/env python3
"""Install the openpi Transformers replacements into the active environment."""

from pathlib import Path
import shutil

import transformers


def main() -> None:
    source = Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
    destination = Path(transformers.__file__).resolve().parent
    for path in source.rglob("*.py"):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f"Installed Transformers replacements into {destination}")


if __name__ == "__main__":
    main()
