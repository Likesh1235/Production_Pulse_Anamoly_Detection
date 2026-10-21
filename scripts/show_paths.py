"""Print the resolved project root, configuration and folders (read-only).

Run from ANY folder, e.g.:  python scripts/show_paths.py
Nothing is created or modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production_pulse import __version__, get_paths, load_config  # noqa: E402


def main() -> int:
    paths = get_paths()
    config = load_config(root=paths.root)
    print(f"Production Pulse {__version__}")
    print(f"Python           : {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Project root     : {paths.root}")
    print(f"Random seed      : {config.random_seed}")
    print("\nOriginal files (existence only, not opened):")
    for f in paths.protected_files:
        print(f"  [{'OK' if f.is_file() else 'MISSING'}] {f.relative_to(paths.root).as_posix()}")
    print("\nOutput folders (created later on demand):")
    for d in paths.output_dirs:
        print(f"  [{'exists' if d.is_dir() else 'not yet'}] {d.relative_to(paths.root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
