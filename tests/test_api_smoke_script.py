"""Checks for scripts/verify/api_smoke.py that need no model, no data and no server start."""

import importlib.util
import json
import socket
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify" / "api_smoke.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("api_smoke", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_port_auto_returns_a_bindable_port(smoke):
    for env in ({}, {"PP_API_SMOKE_PORT": "auto"}, {"PP_API_SMOKE_PORT": ""}, {"PP_API_SMOKE_PORT": "0"}):
        port, how = smoke.resolve_port(env)
        assert 1024 <= port <= 65535 and how.startswith("auto")
        assert smoke.port_status(port)[0]


def test_resolve_port_explicit_and_invalid(smoke):
    assert smoke.resolve_port({"PP_API_SMOKE_PORT": " 8011 "}) == (8011, "PP_API_SMOKE_PORT=8011")
    for bad in ("abc", "80", "70000", "8000.5"):
        with pytest.raises(smoke.PortError):
            smoke.resolve_port({"PP_API_SMOKE_PORT": bad})


def test_port_status_detects_a_listening_server(smoke):
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        ok, reason = smoke.port_status(port)
        assert not ok and str(port) in reason


def test_server_env_isolates_history_and_ignores_host_port(smoke, monkeypatch):
    monkeypatch.setenv("PP_API_PORT", "8000")
    monkeypatch.setenv("PP_API_HOST", "0.0.0.0")
    env = smoke.server_env("t" * 24)
    assert "PP_API_PORT" not in env and "PP_API_HOST" not in env
    assert env["PP_API_DATABASE_PATH"] == "data/api/verification_smoke.sqlite3"
    assert env["PP_API_DATABASE_PATH"] != "data/api/predictions.sqlite3"
    assert env["PP_API_ADMIN_TOKEN"] == "t" * 24


def test_occupied_port_exits_3_and_leaves_the_other_server_alone(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "LOG", tmp_path)
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        monkeypatch.setenv("PP_API_SMOKE_PORT", str(port))
        assert smoke.main() == 3
        assert srv.fileno() != -1                      # still open: nothing touched it
    res = json.loads((tmp_path / "30_api_results.json").read_text(encoding="utf-8"))
    assert res["port"] == port and res["exit_code"] == 3 and not res["passed"] and "command" not in res


def test_invalid_port_exits_4(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "LOG", tmp_path)
    monkeypatch.setenv("PP_API_SMOKE_PORT", "not-a-port")
    assert smoke.main() == 4
    assert json.loads((tmp_path / "30_api_results.json").read_text(encoding="utf-8"))["exit_code"] == 4
