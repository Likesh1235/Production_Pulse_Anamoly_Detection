"""Preflight: confirm the interpreter and import every required dependency (prints versions)."""

import importlib
import sys
import time

REQUIRED = ["numpy", "pandas", "sklearn", "scipy", "pyreadr", "pyarrow", "matplotlib", "seaborn",
            "fastapi", "uvicorn", "httpx", "pydantic", "starlette", "pytest", "nbconvert", "nbclient",
            "ipykernel", "jupyterlab", "tensorflow", "keras", "h5py"]

print("sys.executable:", sys.executable)
print("sys.version:", sys.version.replace("\n", " "))
print("sys.prefix:", sys.prefix)
print("in .venv313:", sys.prefix.lower().rstrip("\\/").endswith(".venv313"))
missing = []
for name in REQUIRED:
    t = time.perf_counter()
    try:
        mod = importlib.import_module(name)
        print(f"OK      {name:<11} {getattr(mod, '__version__', '?'):<12} ({time.perf_counter() - t:.1f}s)")
    except Exception as exc:  # noqa: BLE001 - report every failure
        missing.append(name)
        print(f"FAILED  {name:<11} {type(exc).__name__}: {exc}")
try:
    import tensorflow as tf
    print("tensorflow devices:", [d.device_type for d in tf.config.list_physical_devices()])
except Exception as exc:  # noqa: BLE001
    print("tensorflow device query failed:", exc)
print("RESULT:", "ALL IMPORTS OK" if not missing else f"MISSING {missing}")
sys.exit(1 if missing else 0)
