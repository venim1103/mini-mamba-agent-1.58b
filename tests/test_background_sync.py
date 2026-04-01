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


def test_upload_checkpoint_success_records_file():
    api = MagicMock()
    uploaded = set()
    logs = []

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
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


def test_upload_checkpoint_failure_does_not_record_file():
    api = MagicMock()
    api.upload_file.side_effect = RuntimeError("boom")
    uploaded = set()
    logs = []

    ok = background_sync.upload_checkpoint(
        api,
        "checkpoints/model.pt",
        "org/repo",
        uploaded,
        sleep_before_upload=0,
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


def test_main_requires_repo_id(monkeypatch):
    monkeypatch.setattr(background_sync, "REPO_ID", None)

    with pytest.raises(ValueError, match="REPO_ID"):
        background_sync.main()


def test_main_initializes_api_and_runs_loop(monkeypatch):
    fake_api = MagicMock()
    monkeypatch.setattr(background_sync, "REPO_ID", "org/repo")
    monkeypatch.setattr(background_sync, "HF_TOKEN", "token")
    monkeypatch.setattr(background_sync, "HfApi", MagicMock(return_value=fake_api))

    called = {}

    def fake_run_sync_loop(api, repo_id, checkpoint_dir):
        called["api"] = api
        called["repo_id"] = repo_id
        called["checkpoint_dir"] = checkpoint_dir

    monkeypatch.setattr(background_sync, "run_sync_loop", fake_run_sync_loop)

    background_sync.main()

    fake_api.create_repo.assert_called_once_with("org/repo", repo_type="model", private=True)
    assert called == {
        "api": fake_api,
        "repo_id": "org/repo",
        "checkpoint_dir": background_sync.CHECKPOINT_DIR,
    }
