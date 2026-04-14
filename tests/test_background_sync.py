from unittest.mock import MagicMock

import pytest

import background_sync


def test_iter_new_checkpoint_files_filters_seen_and_extension(monkeypatch):
    monkeypatch.setattr(background_sync.os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        background_sync.os,
        "walk",
        lambda _: [
            ("checkpoints", [], ["a.pt", "b.txt"]),
            ("checkpoints/sub", [], ["c.pt"]),
        ],
    )

    seen = {"checkpoints/sub/c.pt"}
    found = list(background_sync.iter_new_checkpoint_files("checkpoints", seen))

    assert found == ["checkpoints/a.pt"]


def test_upload_checkpoint_success_records_file(monkeypatch):
    api = MagicMock()
    uploaded = set()
    logs = []

    monkeypatch.setattr(background_sync.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(background_sync, "_is_old_enough", lambda _p, _age: True)
    monkeypatch.setattr(background_sync, "_is_stable", lambda _p, _w, sleep_fn=None: True)
    monkeypatch.setattr(background_sync, "_looks_like_valid_torch_zip", lambda _p: True)

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
        min_file_age_seconds=0,
        stabilize_window_seconds=0,
        validate_zip=True,
        sleep_fn=lambda _: None,
        logger=logs.append,
    )

    assert ok is True
    assert "checkpoints/model.pt" in uploaded
    api.upload_file.assert_called_once_with(
        path_or_fileobj="checkpoints/model.pt",
        path_in_repo="checkpoints/model.pt",
        repo_id="org/repo",
        repo_type="model",
    )
    assert any("Backed up model.pt" in msg for msg in logs)


def test_upload_checkpoint_failure_does_not_record_file(monkeypatch):
    api = MagicMock()
    api.upload_file.side_effect = RuntimeError("boom")
    uploaded = set()
    logs = []

    monkeypatch.setattr(background_sync.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(background_sync, "_is_old_enough", lambda _p, _age: True)
    monkeypatch.setattr(background_sync, "_is_stable", lambda _p, _w, sleep_fn=None: True)
    monkeypatch.setattr(background_sync, "_looks_like_valid_torch_zip", lambda _p: True)

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
        min_file_age_seconds=0,
        stabilize_window_seconds=0,
        validate_zip=True,
        sleep_fn=lambda _: None,
        logger=logs.append,
    )

    assert ok is False
    assert uploaded == set()
    assert any("Upload failed model.pt" in msg for msg in logs)


def test_sync_once_counts_successes(monkeypatch):
    api = MagicMock()
    uploaded = set()

    monkeypatch.setattr(
        background_sync,
        "iter_new_checkpoint_files",
        lambda *_: iter(["checkpoints/a.pt", "checkpoints/b.pt"]),
    )

    def fake_upload(_api, filepath, *_args, **_kwargs):
        uploaded.add(filepath)
        return filepath.endswith("a.pt")

    monkeypatch.setattr(background_sync, "upload_checkpoint", fake_upload)

    count = background_sync.sync_once(api, "org/repo", "checkpoints", uploaded)

    assert count == 1
    assert uploaded == {"checkpoints/a.pt", "checkpoints/b.pt"}


def test_upload_checkpoint_defers_when_too_new(monkeypatch):
    api = MagicMock()
    uploaded = set()
    logs = []

    monkeypatch.setattr(background_sync.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(background_sync, "_is_old_enough", lambda _p, _age: False)

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
        min_file_age_seconds=60,
        stabilize_window_seconds=0,
        validate_zip=False,
        sleep_fn=lambda _: None,
        logger=logs.append,
    )

    assert ok is False
    api.upload_file.assert_not_called()
    assert any("too new" in msg for msg in logs)


def test_upload_checkpoint_defers_when_unstable(monkeypatch):
    api = MagicMock()
    uploaded = set()
    logs = []

    monkeypatch.setattr(background_sync.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(background_sync, "_is_old_enough", lambda _p, _age: True)
    monkeypatch.setattr(background_sync, "_is_stable", lambda _p, _w, sleep_fn=None: False)

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
        min_file_age_seconds=60,
        stabilize_window_seconds=10,
        validate_zip=False,
        sleep_fn=lambda _: None,
        logger=logs.append,
    )

    assert ok is False
    api.upload_file.assert_not_called()
    assert any("still changing" in msg for msg in logs)


def test_upload_checkpoint_defers_when_zip_invalid(monkeypatch):
    api = MagicMock()
    uploaded = set()
    logs = []

    monkeypatch.setattr(background_sync.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(background_sync, "_is_old_enough", lambda _p, _age: True)
    monkeypatch.setattr(background_sync, "_is_stable", lambda _p, _w, sleep_fn=None: True)
    monkeypatch.setattr(background_sync, "_looks_like_valid_torch_zip", lambda _p: False)

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
        min_file_age_seconds=60,
        stabilize_window_seconds=10,
        validate_zip=True,
        sleep_fn=lambda _: None,
        logger=logs.append,
    )

    assert ok is False
    api.upload_file.assert_not_called()
    assert any("not valid yet" in msg for msg in logs)


def test_upload_checkpoint_defers_on_os_error_during_preflight(monkeypatch):
    api = MagicMock()
    uploaded = set()
    logs = []

    monkeypatch.setattr(background_sync.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(background_sync, "_is_old_enough", lambda _p, _age: (_ for _ in ()).throw(OSError("file vanished")))

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
        min_file_age_seconds=60,
        stabilize_window_seconds=0,
        validate_zip=False,
        sleep_fn=lambda _: None,
        logger=logs.append,
    )

    assert ok is False
    api.upload_file.assert_not_called()
    assert any("OS error during preflight" in msg for msg in logs)


def test_main_requires_repo_id(monkeypatch):
    monkeypatch.delenv("REPO_ID", raising=False)

    with pytest.raises(ValueError, match="REPO_ID"):
        background_sync.main()


def test_main_initializes_api_and_runs_loop(monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setenv("REPO_ID", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(background_sync, "HfApi", MagicMock(return_value=fake_api))

    called = {}

    def fake_run_sync_loop(
        api,
        repo_id,
        checkpoint_dir,
        poll_interval,
        sleep_before_upload,
        min_file_age_seconds,
        stabilize_window_seconds,
        validate_zip,
        logger,
    ):
        called["api"] = api
        called["repo_id"] = repo_id
        called["checkpoint_dir"] = checkpoint_dir
        called["poll_interval"] = poll_interval
        called["sleep_before_upload"] = sleep_before_upload
        called["min_file_age_seconds"] = min_file_age_seconds
        called["stabilize_window_seconds"] = stabilize_window_seconds
        called["validate_zip"] = validate_zip
        called["logger"] = logger

    monkeypatch.setattr(background_sync, "run_sync_loop", fake_run_sync_loop)

    background_sync.main()

    fake_api.create_repo.assert_called_once_with("org/repo", repo_type="model", private=True)
    assert called == {
        "api": fake_api,
        "repo_id": "org/repo",
        "checkpoint_dir": background_sync.CHECKPOINT_DIR,
        "poll_interval": background_sync.DEFAULT_POLL_INTERVAL,
        "sleep_before_upload": background_sync.DEFAULT_SLEEP_BEFORE_UPLOAD,
        "min_file_age_seconds": background_sync.DEFAULT_MIN_FILE_AGE,
        "stabilize_window_seconds": background_sync.DEFAULT_STABILIZE_WINDOW,
        "validate_zip": background_sync.DEFAULT_VALIDATE_ZIP,
        "logger": background_sync._log,
    }


def test_main_uses_sync_timing_env_overrides(monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setenv("REPO_ID", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("SYNC_POLL_INTERVAL", "3")
    monkeypatch.setenv("SYNC_SLEEP_BEFORE_UPLOAD", "1")
    monkeypatch.setenv("SYNC_MIN_FILE_AGE", "4")
    monkeypatch.setenv("SYNC_STABILIZE_WINDOW", "2")
    monkeypatch.setenv("SYNC_VALIDATE_ZIP", "0")
    monkeypatch.setattr(background_sync, "HfApi", MagicMock(return_value=fake_api))

    called = {}

    def fake_run_sync_loop(
        api,
        repo_id,
        checkpoint_dir,
        poll_interval,
        sleep_before_upload,
        min_file_age_seconds,
        stabilize_window_seconds,
        validate_zip,
        logger,
    ):
        called["poll_interval"] = poll_interval
        called["sleep_before_upload"] = sleep_before_upload
        called["min_file_age_seconds"] = min_file_age_seconds
        called["stabilize_window_seconds"] = stabilize_window_seconds
        called["validate_zip"] = validate_zip

    monkeypatch.setattr(background_sync, "run_sync_loop", fake_run_sync_loop)
    background_sync.main()

    assert called == {
        "poll_interval": 3,
        "sleep_before_upload": 1,
        "min_file_age_seconds": 4,
        "stabilize_window_seconds": 2,
        "validate_zip": False,
    }


def test_read_bool_env_accepts_common_values(monkeypatch):
    monkeypatch.setenv("SYNC_VALIDATE_ZIP", "yes")
    assert background_sync._read_bool_env("SYNC_VALIDATE_ZIP", False, logger=lambda _: None) is True

    monkeypatch.setenv("SYNC_VALIDATE_ZIP", "0")
    assert background_sync._read_bool_env("SYNC_VALIDATE_ZIP", True, logger=lambda _: None) is False


def test_main_self_check_mode_success(monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setenv("REPO_ID", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(background_sync, "HfApi", MagicMock(return_value=fake_api))
    monkeypatch.setattr(background_sync, "run_self_check", MagicMock(return_value=True))

    code = background_sync.main(["--self-check"])

    assert code == 0
    background_sync.run_self_check.assert_called_once_with(
        fake_api, "org/repo", background_sync.CHECKPOINT_DIR, logger=background_sync._log
    )


def test_main_self_check_mode_failure(monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setenv("REPO_ID", "org/repo")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(background_sync, "HfApi", MagicMock(return_value=fake_api))
    monkeypatch.setattr(background_sync, "run_self_check", MagicMock(return_value=False))

    code = background_sync.main(["--self-check"])

    assert code == 2


def test_run_self_check_repo_create_failure_but_repo_info_ok(monkeypatch):
    api = MagicMock()
    api.whoami.return_value = {"name": "user"}
    api.create_repo.side_effect = RuntimeError("already exists")
    api.repo_info.return_value = {"id": "org/repo"}
    logs = []

    ok = background_sync.run_self_check(api, "org/repo", "missing_dir", logger=logs.append)

    assert ok is True
    api.repo_info.assert_called_once_with(repo_id="org/repo", repo_type="model")
    assert any("Repo access confirmed via repo_info" in msg for msg in logs)


def test_run_self_check_repo_access_failure_returns_false():
    api = MagicMock()
    api.whoami.return_value = {"name": "user"}
    api.create_repo.side_effect = RuntimeError("forbidden")
    api.repo_info.side_effect = RuntimeError("403")
    logs = []

    ok = background_sync.run_self_check(api, "org/repo", "missing_dir", logger=logs.append)

    assert ok is False
    assert any("Repo access failed" in msg for msg in logs)

