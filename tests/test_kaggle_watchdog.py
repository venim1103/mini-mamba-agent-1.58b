import types
from unittest.mock import MagicMock, mock_open

import kaggle_watchdog


def test_set_repo_directory_success(monkeypatch):
    monkeypatch.setattr(kaggle_watchdog.os.path, "exists", lambda _: True)

    changed = {}
    monkeypatch.setattr(kaggle_watchdog.os, "chdir", lambda p: changed.setdefault("path", p))
    monkeypatch.setattr(kaggle_watchdog.os, "getcwd", lambda: "/kaggle/working/mini-mamba-agent-1.58b")

    logs = []
    ok = kaggle_watchdog.set_repo_directory(logger=logs.append)

    assert ok is True
    assert changed["path"] == kaggle_watchdog.REPO_DIR
    assert any("Directory set to:" in msg for msg in logs)


def test_set_repo_directory_missing(monkeypatch):
    monkeypatch.setattr(kaggle_watchdog.os.path, "exists", lambda _: False)
    logs = []

    ok = kaggle_watchdog.set_repo_directory(logger=logs.append)

    assert ok is False
    assert any("Repository not found" in msg for msg in logs)


def test_load_kaggle_secrets_sets_env(monkeypatch):
    class FakeSecretsClient:
        def get_secret(self, name):
            return {"HF_TOKEN": "hf", "WANDB_API_KEY": "wb", "REPO_ID": "repo"}[name]

    fake_module = types.SimpleNamespace(UserSecretsClient=FakeSecretsClient)
    monkeypatch.setitem(__import__("sys").modules, "kaggle_secrets", fake_module)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("REPO_ID", raising=False)

    out = kaggle_watchdog.load_kaggle_secrets(logger=lambda _: None)

    assert out == {"HF_TOKEN": "hf", "WANDB_API_KEY": "wb", "REPO_ID": "repo"}
    assert kaggle_watchdog.os.environ["HF_TOKEN"] == "hf"
    assert kaggle_watchdog.os.environ["WANDB_API_KEY"] == "wb"
    assert kaggle_watchdog.os.environ["REPO_ID"] == "repo"


def test_start_background_sync_launches_process(monkeypatch):
    popen = MagicMock(return_value="proc")
    monkeypatch.setattr(kaggle_watchdog.subprocess, "Popen", popen)
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setenv("HF_TOKEN", "hf")

    proc = kaggle_watchdog.start_background_sync(log_path="sync_logs.txt")

    assert proc == "proc"
    popen.assert_called_once()
    args, kwargs = popen.call_args
    assert args[0] == ["python", "-u", "background_sync.py"]
    assert kwargs["stderr"] == kaggle_watchdog.subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"


def test_run_sync_self_check_runs_subprocess(monkeypatch):
    run = MagicMock(return_value=types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(kaggle_watchdog.subprocess, "run", run)
    monkeypatch.setattr("builtins.open", mock_open())

    code = kaggle_watchdog.run_sync_self_check(log_path="sync_logs.txt")

    assert code == 0
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == ["python", "-u", "background_sync.py", "--self-check"]
    assert kwargs["stderr"] == kaggle_watchdog.subprocess.STDOUT
    assert kwargs["check"] is False


def test_main_returns_1_when_repo_missing(monkeypatch):
    monkeypatch.setattr(kaggle_watchdog, "set_repo_directory", lambda _repo_dir: False)

    assert kaggle_watchdog.main() == 1


def test_main_happy_path(monkeypatch):
    calls = {"secrets": 0, "check": 0, "start": 0}
    monkeypatch.setattr(kaggle_watchdog, "set_repo_directory", lambda _repo_dir: True)
    monkeypatch.setattr(
        kaggle_watchdog,
        "load_kaggle_secrets",
        lambda: calls.__setitem__("secrets", calls["secrets"] + 1),
    )
    monkeypatch.setattr(
        kaggle_watchdog,
        "run_sync_self_check",
        lambda log_path: calls.__setitem__("check", calls["check"] + 1) or 0,
    )
    monkeypatch.setattr(
        kaggle_watchdog,
        "start_background_sync",
        lambda log_path: calls.__setitem__("start", calls["start"] + 1),
    )

    assert kaggle_watchdog.main() == 0
    assert calls == {"secrets": 1, "check": 1, "start": 1}


def test_main_returns_2_when_self_check_fails(monkeypatch):
    monkeypatch.setattr(kaggle_watchdog, "set_repo_directory", lambda _repo_dir: True)
    monkeypatch.setattr(kaggle_watchdog, "load_kaggle_secrets", lambda: None)
    monkeypatch.setattr(kaggle_watchdog, "run_sync_self_check", lambda log_path: 2)

    assert kaggle_watchdog.main() == 2
