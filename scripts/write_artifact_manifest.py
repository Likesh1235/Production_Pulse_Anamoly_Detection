r"""Create (or just check) the integrity manifest of a saved model folder.

    python scripts\write_artifact_manifest.py                   # artifacts/models/autoencoder_v1
    python scripts\write_artifact_manifest.py --check           # verify only, write nothing
    python scripts\write_artifact_manifest.py path\to\model_dir

It ADDS ``manifest.json`` (SHA-256 of model.keras, metadata.json, history.csv, fingerprints.npz) next to
the files of an artifact that was saved before manifests existed. It never modifies the model, the
threshold or the scaler, and it refuses to replace an existing manifest. After this, loading the model
(offline or through the API) fails if any of those files changes. This detects accidental damage
or edits; it is not a security signature.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from production_pulse.autoencoder import AutoencoderError, verify_manifest, write_manifest  # noqa: E402
from production_pulse.paths import get_paths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", nargs="?", default="artifacts/models/autoencoder_v1")
    parser.add_argument("--check", action="store_true", help="only verify an existing manifest")
    args = parser.parse_args(argv)
    paths = get_paths(ROOT)
    directory = Path(args.model_dir)
    directory = directory if directory.is_absolute() else ROOT / directory
    try:
        state = verify_manifest(directory)
        if args.check:
            print(f"manifest: {state} ({directory})")
            return 0 if state == "verified" else 1
        if state == "verified":
            print(f"A valid manifest already exists in {directory}; nothing to do.")
            return 0
        target = write_manifest(directory, paths, overwrite=False)
        print(f"wrote {target}")
        print(f"manifest: {verify_manifest(directory)}")
        return 0
    except (AutoencoderError, FileExistsError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
