"""API smoke test: start OUR OWN Uvicorn server on a free port, check real responses, stop only that server.

    python scripts\\verify\\api_smoke.py                                   # picks a free port automatically
    $env:PP_API_SMOKE_PORT = "8011"; python scripts\\verify\\api_smoke.py  # use exactly this port

Port: ``PP_API_SMOKE_PORT`` (1024-65535) or ``auto`` (default: the operating system picks a free port). The
same port is used for the availability check, the Uvicorn command and the HTTP client. A server you already
run (e.g. on 8000 from VS Code) is never touched: an occupied port is refused with exit code 3, and only the
subprocess this script started is terminated.

Isolation: the server uses its own history database ``data/api/verification_smoke.sqlite3`` (reset at the
start of each run), so smoke predictions never enter your real prediction history, and a throw-away admin
token so the protected reload endpoint can be checked.

Writes logs/verification/30_api_results.json (every call, expected vs actual, failures) and 30_uvicorn.log.
Exit codes: 0 all checks passed, 1 at least one check failed, 2 server did not start, 3 port unavailable,
4 invalid PP_API_SMOKE_PORT. The /predict samples come from EVALUATION runs of the fault-free testing file
(same seeded split as the notebooks); holdout runs are never used.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "logs" / "verification"
sys.path.insert(0, str(ROOT / "src"))

from production_pulse import get_paths, load_config  # noqa: E402
from production_pulse.api.settings import apply_env_overrides  # noqa: E402
from production_pulse.data import FEATURE_COLUMNS, load_raw_dataset  # noqa: E402
from production_pulse.splitting import apply_split, split_runs  # noqa: E402

PORT_ENV = "PP_API_SMOKE_PORT"
HOST = "127.0.0.1"
SMOKE_DB = "data/api/verification_smoke.sqlite3"     # inside data/ as the API requires
STARTUP_TIMEOUT_S = 180


class PortError(ValueError):
    """PP_API_SMOKE_PORT is not a valid port."""


def find_free_port() -> int:
    """Ask the operating system for a currently free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


def resolve_port(environ: Optional[dict] = None) -> tuple[int, str]:
    """Return (port, how it was chosen). Raises PortError for an invalid value."""
    raw = (environ if environ is not None else os.environ).get(PORT_ENV, "auto").strip().lower()
    if raw in ("", "auto", "0"):
        return find_free_port(), "auto (chosen by the operating system)"
    try:
        port = int(raw)
    except ValueError:
        raise PortError(f"{PORT_ENV}={raw!r} is not a number; use a port 1024-65535 or 'auto'") from None
    if not 1024 <= port <= 65535:
        raise PortError(f"{PORT_ENV}={port} is outside 1024-65535")
    return port, f"{PORT_ENV}={port}"


def port_status(port: int) -> tuple[bool, str]:
    """(available, reason). Something accepting connections, or a failed bind, means NOT available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex((HOST, port)) == 0:
            return False, f"something is already accepting connections on {HOST}:{port}"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):            # Windows: refuse to share the port
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            s.bind((HOST, port))
        except OSError as exc:
            return False, f"cannot bind {HOST}:{port} ({exc.__class__.__name__}: {exc})"
    return True, "free"


def evaluation_samples(n: int = 5) -> tuple[list[dict], dict]:
    paths = get_paths()
    cfg = load_config(root=paths.root)
    test_raw = load_raw_dataset("fault_free_testing", paths=paths, config=cfg)
    split = split_runs(test_raw, cfg.evaluation.fractions(), seed=cfg.random_seed, group_columns=cfg.split.group_columns)
    ev = apply_split(test_raw, split)["evaluation"].sort_values(["simulationRun", "sample"]).head(n)
    holdout_runs = set(split.runs("holdout"))
    assert not set(ev["simulationRun"].astype(int)) & holdout_runs, "holdout run selected"
    samples = [{"sample_id": f"eval-run{int(r.simulationRun)}-s{int(r.sample)}",
                "features": {c: float(getattr(r, c)) for c in FEATURE_COLUMNS}} for r in ev.itertuples()]
    info = {"source": "fault_free_testing / evaluation runs (seeded split)", "runs_used": sorted(set(ev["simulationRun"].astype(int))),
            "n_holdout_runs_untouched": len(holdout_runs)}
    return samples, info


def allowed_origin() -> Optional[str]:
    """First CORS origin the API will actually allow (config + PP_API_CORS_ORIGINS), or None if CORS is off."""
    paths = get_paths()
    api = apply_env_overrides(load_config(root=paths.root).api, os.environ)
    return api.cors_origins[0] if api.cors_origins else None


def server_env(admin_token: str) -> dict:
    env = dict(os.environ)
    for k in ("PP_API_PORT", "PP_API_HOST"):          # the command line decides host/port
        env.pop(k, None)
    env["PP_API_DATABASE_PATH"] = SMOKE_DB
    env["PP_API_ADMIN_TOKEN"] = admin_token
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def reset_smoke_db() -> None:
    """Remove only the smoke test's OWN database files from a previous run."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = ROOT / (SMOKE_DB + suffix)
        if p.is_file():
            p.unlink()


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"calls": [], "failures": [], "passed": False}
    out_file = LOG / "30_api_results.json"

    def save() -> None:
        out_file.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print("results written to", out_file)

    try:
        port, how = resolve_port()
    except PortError as exc:
        print(f"ERROR: {exc}")
        results["failures"].append(str(exc))
        results["exit_code"] = 4
        save()
        return 4
    available, reason = port_status(port)
    results.update({"port": port, "port_choice": how, "port_check": reason, "base_url": f"http://{HOST}:{port}"})
    print(f"port {port} ({how}): {reason}")
    if not available:
        msg = (f"Port {port} is not available: {reason}. This test never stops other servers. Use another port, "
               f"e.g.  $env:{PORT_ENV}='auto'  (or a free number), then rerun. To see the owner on Windows: "
               f"Get-NetTCPConnection -LocalPort {port} | Format-List; Get-Process -Id <OwningProcess>.")
        print(msg)
        results["failures"].append(msg)
        results["exit_code"] = 3
        save()
        return 3

    reset_smoke_db()
    admin_token = secrets.token_urlsafe(24)
    cmd = [sys.executable, "-m", "uvicorn", "production_pulse.api.main:app", "--app-dir", "src",
           "--host", HOST, "--port", str(port)]
    results["command"] = cmd
    results["server_env"] = {"PP_API_DATABASE_PATH": SMOKE_DB, "PP_API_ADMIN_TOKEN": "<random, not logged>"}
    print("starting:", " ".join(cmd))
    proc: Optional[subprocess.Popen] = None
    exit_code = 1
    with open(LOG / "30_uvicorn.log", "w", encoding="utf-8") as log_file:
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, env=server_env(admin_token))
            results["server_pid"] = proc.pid
            with httpx.Client(base_url=f"http://{HOST}:{port}", timeout=60) as client:
                t0 = time.time()
                while True:
                    if proc.poll() is not None:
                        msg = (f"Our Uvicorn process exited with code {proc.returncode} before answering "
                               f"(port taken in the meantime, or a startup error). See 30_uvicorn.log.")
                        print(msg)
                        results["failures"].append(msg)
                        exit_code = 2
                        return exit_code
                    try:
                        client.get("/health")
                        break
                    except httpx.TransportError:
                        if time.time() - t0 > STARTUP_TIMEOUT_S:
                            msg = f"API did not answer within {STARTUP_TIMEOUT_S}s; see 30_uvicorn.log"
                            print(msg)
                            results["failures"].append(msg)
                            exit_code = 2
                            return exit_code
                        time.sleep(0.5)
                results["startup_seconds"] = round(time.time() - t0, 1)
                if proc.poll() is not None:     # something answered, but our process is gone
                    msg = "A server answered, but it is NOT the one this test started; results would be meaningless."
                    print(msg)
                    results["failures"].append(msg)
                    exit_code = 3
                    return exit_code

                def call(name: str, method: str, path: str, expect: int,
                         check: Optional[Callable[[httpx.Response], Optional[str]]] = None, **kw) -> httpx.Response:
                    r = client.request(method, path, **kw)
                    try:
                        body: Any = r.json()
                    except ValueError:
                        body = r.text
                    problem = None if r.status_code == expect else f"expected HTTP {expect}, got {r.status_code}"
                    if problem is None and check is not None:
                        problem = check(r)
                    entry = {"name": name, "request": f"{method} {path}", "status": r.status_code, "expected_status": expect,
                             "ok": problem is None, "body": body}
                    acao = {k: v for k, v in r.headers.items() if k.lower().startswith("access-control")}
                    if acao:
                        entry["response_headers"] = acao
                    if problem:
                        entry["problem"] = problem
                        results["failures"].append(f"{name}: {problem}")
                    results["calls"].append(entry)
                    print(f"{'OK  ' if problem is None else 'FAIL'} {name:<26} {method} {path} -> {r.status_code}"
                          + (f"   ({problem})" if problem else ""))
                    return r

                call("health", "GET", "/health", 200,
                     lambda r: None if r.json().get("model", {}).get("status") == "ready"
                     and r.json().get("database", {}).get("status") == "ready" else "model or database not ready")
                status = call("model_status", "GET", "/model/status", 200,
                              lambda r: None if r.json().get("available") and r.json()["feature_schema"]["n_features"] == 52
                              else "model not available or wrong feature count").json()
                saved_threshold = (status.get("metadata") or {}).get("threshold")
                call("metrics", "GET", "/metrics", 200, lambda r: None if "available" in r.json() else "no 'available' field")
                call("history_empty", "GET", "/history", 200,
                     lambda r: None if r.json().get("total") == 0 else f"fresh smoke DB should be empty, total={r.json().get('total')}",
                     params={"limit": 5})
                samples, info = evaluation_samples()
                results["predict_input"] = info

                def check_predict(r: httpx.Response) -> Optional[str]:
                    b = r.json()
                    if b.get("n_samples") != len(samples) or len(b.get("results", [])) != len(samples):
                        return "wrong number of results"
                    if saved_threshold is not None and abs(b["threshold"] - float(saved_threshold)) > 1e-12:
                        return f"threshold {b['threshold']} differs from model metadata {saved_threshold}"
                    for res in b["results"]:
                        if not res["reconstruction_error"] >= 0 or res["is_anomaly"] != (res["reconstruction_error"] > b["threshold"]):
                            return f"inconsistent decision for sample {res['index']}"
                    return None if b.get("history_stored") else "predictions were not stored"

                call("predict_eval_runs", "POST", "/predict", 200, check_predict, json={"samples": samples})
                missing = {k: v for k, v in samples[0]["features"].items() if k != "xmv_3"}
                call("predict_missing_feature", "POST", "/predict", 422, json={"features": missing})
                bad = dict(samples[0]["features"], xmeas_1="not-a-number")
                call("predict_non_numeric", "POST", "/predict", 422, json={"features": bad})
                call("predict_malformed_json", "POST", "/predict", 422, content=b"{not json",
                     headers={"Content-Type": "application/json"})
                huge = dict(samples[0]["features"], xmeas_1=1e300)
                call("predict_out_of_range", "POST", "/predict", 422,
                     lambda r: None if r.json().get("error", {}).get("code") == "value_out_of_range" else "wrong error code",
                     json={"features": huge})
                call("history_after", "GET", "/history", 200,
                     lambda r: None if r.json().get("total") == len(samples)
                     else f"expected {len(samples)} stored rows, total={r.json().get('total')}",
                     params={"limit": 5})
                origin = allowed_origin()
                if origin is None:
                    results["notes"] = ["CORS is disabled in the configuration; allowed-origin check skipped"]
                else:
                    call("cors_preflight_allowed", "OPTIONS", "/predict", 200,
                         lambda r: None if r.headers.get("access-control-allow-origin") == origin else "allow-origin header missing",
                         headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
                call("cors_preflight_other", "OPTIONS", "/predict", 400,
                     lambda r: None if "access-control-allow-origin" not in r.headers else "foreign origin was allowed",
                     headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
                call("reload_no_token", "POST", "/model/reload", 401)
                call("reload_wrong_token", "POST", "/model/reload", 401, headers={"X-Admin-Token": "x" * 32})
                call("reload_valid_token", "POST", "/model/reload", 200,
                     lambda r: None if r.json().get("status") == "ready" else "model not ready after reload",
                     headers={"X-Admin-Token": admin_token})
                results["model_ready"] = bool(status.get("available"))
                exit_code = 0 if not results["failures"] else 1
                return exit_code
        except Exception as exc:  # noqa: BLE001 - record any unexpected failure, then clean up
            msg = f"unexpected error: {exc.__class__.__name__}: {exc}"
            print(msg)
            results["failures"].append(msg)
            exit_code = 1
            return exit_code
        finally:
            if proc is not None and proc.poll() is None:     # stop ONLY the process we started
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            results["server_exit_code"] = None if proc is None else proc.returncode
            results["passed"] = exit_code == 0 and not results["failures"]
            results["exit_code"] = exit_code
            save()
            print("SMOKE TEST", "PASSED" if results["passed"] else f"FAILED (exit {exit_code})")


if __name__ == "__main__":
    raise SystemExit(main())
