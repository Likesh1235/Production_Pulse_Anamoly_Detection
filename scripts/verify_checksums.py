"""Verify the original files against docs/DATA_CHECKSUMS.md (read-only).

Usage:  python scripts/verify_checksums.py
Exit code 0 = all MATCH, 1 = any MISMATCH/MISSING, 2 = baseline unreadable.
Files are read in 1 MB chunks, so large datasets are never loaded into memory.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production_pulse import get_paths  # noqa: E402

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([\d,]+)\s*\|\s*`([0-9a-fA-F]{64})`\s*\|")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_baseline(md_file: Path) -> list[tuple[str, int, str]]:
    rows = []
    for line in md_file.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            rows.append((m.group(1), int(m.group(2).replace(",", "")), m.group(3).lower()))
    return rows


def main() -> int:
    paths = get_paths()
    baseline_file = paths.root / "docs" / "DATA_CHECKSUMS.md"
    try:
        rows = read_baseline(baseline_file)
    except OSError as exc:
        print(f"BASELINE UNAVAILABLE: {exc}")
        return 2
    if not rows:
        print(f"BASELINE UNAVAILABLE: no checksum rows found in {baseline_file}")
        return 2

    ok = True
    for name, size, expected in rows:
        f = paths.root / name
        if not f.is_file():
            print(f"MISSING   {name}")
            ok = False
            continue
        actual = sha256_of(f)
        status = "MATCH" if (actual == expected and f.stat().st_size == size) else "MISMATCH"
        ok &= status == "MATCH"
        print(f"{status:<9} {name}  ({f.stat().st_size:,} bytes)")
    print("\nALL MATCH" if ok else "\nPROBLEM FOUND - do not continue until resolved")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
