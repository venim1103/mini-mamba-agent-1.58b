# Copyright 2026 venim1103
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for run_validation.py.

All tests are fully offline — no network calls, no HF API, no subprocess.
Heavy imports (huggingface_hub) are only imported inside functions in the
source file, so they are naturally monkeypatched at the call site.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make run_validation importable without triggering __main__ execution
# ---------------------------------------------------------------------------
import importlib.util

_RV_PATH = Path(__file__).resolve().parents[1] / "run_validation.py"
_spec = importlib.util.spec_from_file_location("run_validation", _RV_PATH)
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _base_env(tmp_path: Path) -> dict[str, str]:
    """Minimum valid environment with VAL_LOCAL_BASE pointing to a temp dir."""
    return {"VAL_LOCAL_BASE": str(tmp_path)}


def _cfg(tmp_path: Path, extra: dict[str, str] | None = None) -> dict:
    env = {**_base_env(tmp_path), **(extra or {})}
    with patch.dict(os.environ, env, clear=True):
        return rv.build_config()


SAMPLE_FILES = [
    "checkpoints/bitmamba_scout/step_010000.pt",
    "checkpoints/bitmamba_scout/step_050000.pt",
    "checkpoints/bitmamba_scout/step_100000.pt",
    "checkpoints/bitmamba_scout/step_200000.pt",
    "checkpoints/bitmamba_scout/step_300000.pt",
]


# ════════════════════════════════════════════════════════════════════════════
# 1. _step_from_path
# ════════════════════════════════════════════════════════════════════════════

class TestStepFromPath:
    def test_standard_filename(self):
        assert rv._step_from_path("checkpoints/bitmamba_scout/step_012345.pt") == 12345

    def test_zero_padded(self):
        assert rv._step_from_path("step_000001.pt") == 1

    def test_large_step(self):
        assert rv._step_from_path("step_1000000.pt") == 1_000_000

    def test_non_matching_returns_minus_one(self):
        assert rv._step_from_path("sft_final.pt") == -1

    def test_non_matching_no_extension(self):
        assert rv._step_from_path("latest") == -1

    def test_path_with_subdirectory(self):
        assert rv._step_from_path("a/b/c/step_042.pt") == 42


# ════════════════════════════════════════════════════════════════════════════
# 2. _env and _flag helpers
# ════════════════════════════════════════════════════════════════════════════

class TestEnvHelpers:
    def test_env_returns_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("SOME_KEY", raising=False)
        assert rv._env("SOME_KEY", "default_val") == "default_val"

    def test_env_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("SOME_KEY", "  hello  ")
        assert rv._env("SOME_KEY") == "hello"

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "True", "YES", "ON"])
    def test_flag_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("SOME_FLAG", value)
        assert rv._flag("SOME_FLAG") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "False"])
    def test_flag_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("SOME_FLAG", value)
        assert rv._flag("SOME_FLAG") is False

    def test_flag_missing_is_false(self, monkeypatch):
        monkeypatch.delenv("SOME_FLAG", raising=False)
        assert rv._flag("SOME_FLAG") is False


# ════════════════════════════════════════════════════════════════════════════
# 3. build_config
# ════════════════════════════════════════════════════════════════════════════

class TestBuildConfig:
    def test_raises_when_local_base_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                rv.build_config()

    def test_defaults_are_applied(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg["repo_id"] == "venim1103/mini-mamba-agent-1b58"
        assert cfg["ckpt_subdir"] == "checkpoints/bitmamba_scout"
        assert cfg["mode"] == "scout"
        assert cfg["batch_size"] == "4"
        assert cfg["wandb_project"] == "Agentic-1.58b-Validation"
        assert cfg["wandb_mode"] == "online"
        assert cfg["dry_run"] is False
        assert cfg["skip_eval"] is False
        assert cfg["list_only"] is False

    def test_local_base_sets_derived_paths(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg["local_base"] == str(tmp_path)
        assert cfg["local_ckpt_dir"] == str(tmp_path / "checkpoints/bitmamba_scout")

    def test_default_manifest_path_derived_from_local_base(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg["manifest_path"] == str(tmp_path / "val_data" / "validation_manifest.json")

    def test_explicit_manifest_path_overrides_default(self, tmp_path):
        cfg = _cfg(tmp_path, {"VAL_MANIFEST_PATH": "/custom/manifest.json"})
        assert cfg["manifest_path"] == "/custom/manifest.json"

    def test_custom_ckpt_subdir(self, tmp_path):
        cfg = _cfg(tmp_path, {"VAL_CKPT_SUBDIR": "checkpoints/bitmamba_parent"})
        assert cfg["ckpt_subdir"] == "checkpoints/bitmamba_parent"
        assert cfg["local_ckpt_dir"] == str(tmp_path / "checkpoints/bitmamba_parent")

    def test_flag_dry_run(self, tmp_path):
        cfg = _cfg(tmp_path, {"VAL_DRY_RUN": "1"})
        assert cfg["dry_run"] is True

    def test_flag_skip_eval(self, tmp_path):
        cfg = _cfg(tmp_path, {"VAL_SKIP_EVAL": "yes"})
        assert cfg["skip_eval"] is True

    def test_flag_list_only(self, tmp_path):
        cfg = _cfg(tmp_path, {"VAL_LIST_ONLY": "true"})
        assert cfg["list_only"] is True

    def test_hf_token_read_from_env(self, tmp_path):
        cfg = _cfg(tmp_path, {"HF_TOKEN": "hf_abc123"})
        assert cfg["hf_token"] == "hf_abc123"

    def test_hf_token_empty_when_not_set(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert cfg["hf_token"] == ""


# ════════════════════════════════════════════════════════════════════════════
# 4. select_checkpoints
# ════════════════════════════════════════════════════════════════════════════

class TestSelectCheckpoints:
    def _make_cfg(self, steps="", indices="", n_latest=""):
        return {"steps": steps, "indices": indices, "n_latest": n_latest}

    def test_n_latest_default_five(self):
        files = SAMPLE_FILES.copy()
        result = rv.select_checkpoints(files, self._make_cfg())
        assert result == files[-5:]

    def test_n_latest_explicit(self):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(n_latest="2"))
        assert result == SAMPLE_FILES[-2:]

    def test_n_latest_larger_than_list(self):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(n_latest="100"))
        assert result == SAMPLE_FILES

    def test_steps_selects_matching(self):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(steps="10000,100000"))
        assert result == [SAMPLE_FILES[0], SAMPLE_FILES[2]]

    def test_steps_no_match_returns_empty(self, capsys):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(steps="999"))
        assert result == []
        assert "No checkpoints matched" in capsys.readouterr().out

    def test_steps_takes_priority_over_indices(self):
        result = rv.select_checkpoints(
            SAMPLE_FILES, self._make_cfg(steps="50000", indices="0,1,2")
        )
        assert result == [SAMPLE_FILES[1]]

    def test_steps_takes_priority_over_n_latest(self):
        result = rv.select_checkpoints(
            SAMPLE_FILES, self._make_cfg(steps="300000", n_latest="2")
        )
        assert result == [SAMPLE_FILES[-1]]

    def test_indices_positive(self):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(indices="0,2,4"))
        assert result == [SAMPLE_FILES[0], SAMPLE_FILES[2], SAMPLE_FILES[4]]

    def test_indices_negative(self):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(indices="-1"))
        assert result == [SAMPLE_FILES[-1]]

    def test_indices_mixed_positive_negative(self):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(indices="0,-1"))
        assert result == [SAMPLE_FILES[0], SAMPLE_FILES[-1]]

    def test_indices_deduplicates_preserving_order(self):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(indices="1,1,1"))
        assert result == [SAMPLE_FILES[1]]

    def test_indices_out_of_range_skipped(self, capsys):
        result = rv.select_checkpoints(SAMPLE_FILES, self._make_cfg(indices="0,999"))
        assert result == [SAMPLE_FILES[0]]
        assert "out of range" in capsys.readouterr().out

    def test_indices_takes_priority_over_n_latest(self):
        result = rv.select_checkpoints(
            SAMPLE_FILES, self._make_cfg(indices="0", n_latest="3")
        )
        assert result == [SAMPLE_FILES[0]]

    def test_empty_file_list_n_latest(self):
        result = rv.select_checkpoints([], self._make_cfg(n_latest="3"))
        assert result == []

    def test_empty_file_list_steps(self, capsys):
        result = rv.select_checkpoints([], self._make_cfg(steps="10000"))
        assert result == []

    def test_comma_separated_steps_with_spaces(self):
        result = rv.select_checkpoints(
            SAMPLE_FILES, self._make_cfg(steps=" 10000 , 50000 ")
        )
        assert result == [SAMPLE_FILES[0], SAMPLE_FILES[1]]


# ════════════════════════════════════════════════════════════════════════════
# 5. download_checkpoints
# ════════════════════════════════════════════════════════════════════════════

class TestDownloadCheckpoints:
    def _make_cfg(self, tmp_path: Path, dry_run=False) -> dict:
        subdir = "checkpoints/bitmamba_scout"
        return {
            "repo_id": "org/repo",
            "hf_token": "hf_tok",
            "local_base": str(tmp_path),
            "local_ckpt_dir": str(tmp_path / subdir),
            "dry_run": dry_run,
        }

    def test_skips_existing_files(self, tmp_path, capsys):
        cfg = self._make_cfg(tmp_path)
        existing = tmp_path / "checkpoints/bitmamba_scout/step_010000.pt"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"fake")

        mock_dl = MagicMock()
        with patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
            rv.download_checkpoints(
                ["checkpoints/bitmamba_scout/step_010000.pt"], cfg
            )

        out = capsys.readouterr().out
        assert "[skip]" in out
        mock_dl.assert_not_called()

    def test_dry_run_does_not_call_hf_hub_download(self, tmp_path, capsys):
        cfg = self._make_cfg(tmp_path, dry_run=True)
        mock_dl = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
            rv.download_checkpoints(["checkpoints/bitmamba_scout/step_010000.pt"], cfg)

        mock_dl.assert_not_called()
        assert "[download]" in capsys.readouterr().out

    def test_returns_correct_local_paths(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        repo_path = "checkpoints/bitmamba_scout/step_010000.pt"
        mock_dl = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
            result = rv.download_checkpoints([repo_path], cfg)

        assert result == [str(tmp_path / repo_path)]

    def test_calls_hf_hub_download_with_correct_args(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        repo_path = "checkpoints/bitmamba_scout/step_010000.pt"
        mock_dl = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
            rv.download_checkpoints([repo_path], cfg)

        mock_dl.assert_called_once_with(
            repo_id="org/repo",
            filename=repo_path,
            repo_type="model",
            local_dir=str(tmp_path),
            token="hf_tok",
        )

    def test_empty_token_passed_as_none(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        cfg["hf_token"] = ""
        repo_path = "checkpoints/bitmamba_scout/step_010000.pt"
        mock_dl = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
            rv.download_checkpoints([repo_path], cfg)

        assert mock_dl.call_args.kwargs["token"] is None

    def test_creates_local_ckpt_dir(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        ckpt_dir = Path(cfg["local_ckpt_dir"])
        assert not ckpt_dir.exists()
        mock_dl = MagicMock()

        with patch.dict(sys.modules, {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
            rv.download_checkpoints(["checkpoints/bitmamba_scout/step_010000.pt"], cfg)

        assert ckpt_dir.exists()


# ════════════════════════════════════════════════════════════════════════════
# 6. run_validation
# ════════════════════════════════════════════════════════════════════════════

class TestRunValidation:
    def _make_cfg(self, tmp_path: Path, manifest_exists=True) -> dict:
        manifest = tmp_path / "val_data" / "validation_manifest.json"
        if manifest_exists:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}")
        return {
            "local_base": str(tmp_path),
            "local_ckpt_dir": str(tmp_path / "checkpoints/bitmamba_scout"),
            "manifest_path": str(manifest),
            "mode": "scout",
            "batch_size": "4",
            "wandb_project": "TestProject",
            "wandb_mode": "disabled",
        }

    def test_returns_1_when_validate_script_missing(self, tmp_path, capsys):
        cfg = self._make_cfg(tmp_path)
        result = rv.run_validation(cfg, n_checkpoints=2)
        assert result == 1
        assert "validate.py not found" in capsys.readouterr().out

    def test_returns_1_when_manifest_missing(self, tmp_path, capsys):
        cfg = self._make_cfg(tmp_path, manifest_exists=False)
        # Create validate.py so it passes that check
        (tmp_path / "validate.py").write_text("# stub")
        result = rv.run_validation(cfg, n_checkpoints=2)
        assert result == 1
        assert "Manifest not found" in capsys.readouterr().out

    def test_builds_correct_subprocess_command(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        (tmp_path / "validate.py").write_text("# stub")

        with patch("run_validation.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rv.run_validation(cfg, n_checkpoints=3)

        cmd = mock_run.call_args.args[0]
        assert str(tmp_path / "validate.py") in cmd
        assert "--mode" in cmd and "scout" in cmd
        assert "--max-checkpoints" in cmd and "3" in cmd
        assert "--batch-size" in cmd and "4" in cmd
        assert "--wandb-project" in cmd and "TestProject" in cmd
        assert "--wandb-mode" in cmd and "disabled" in cmd

    def test_passes_cwd_as_local_base(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        (tmp_path / "validate.py").write_text("# stub")

        with patch("run_validation.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            rv.run_validation(cfg, n_checkpoints=1)

        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)

    def test_returns_subprocess_returncode(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        (tmp_path / "validate.py").write_text("# stub")

        with patch("run_validation.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=42)
            result = rv.run_validation(cfg, n_checkpoints=1)

        assert result == 42


# ════════════════════════════════════════════════════════════════════════════
# 7. _load_secrets
# ════════════════════════════════════════════════════════════════════════════

class TestLoadSecrets:
    def test_no_op_when_all_secrets_present(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_present")
        monkeypatch.setenv("WANDB_API_KEY", "wb_present")
        # Neither Kaggle nor Colab should be imported
        monkeypatch.delitem(sys.modules, "kaggle_secrets", raising=False)
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        rv._load_secrets()  # must not raise

    def test_loads_from_kaggle_when_missing(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        fake_client = MagicMock()
        fake_client.get_secret.side_effect = lambda k: f"kaggle_{k.lower()}"
        fake_module = types.SimpleNamespace(UserSecretsClient=lambda: fake_client)
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_module)
        monkeypatch.delitem(sys.modules, "google", raising=False)
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)

        rv._load_secrets()

        assert os.environ["HF_TOKEN"] == "kaggle_hf_token"
        assert os.environ["WANDB_API_KEY"] == "kaggle_wandb_api_key"

    def test_falls_through_to_colab_when_kaggle_partial(self, monkeypatch):
        """If Kaggle only provides HF_TOKEN, Colab should fill WANDB_API_KEY."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        # Kaggle returns HF_TOKEN only
        fake_kaggle_client = MagicMock()
        def kaggle_get(key):
            return "hf_from_kaggle" if key == "HF_TOKEN" else ""
        fake_kaggle_client.get_secret.side_effect = kaggle_get
        fake_kaggle = types.SimpleNamespace(UserSecretsClient=lambda: fake_kaggle_client)
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_kaggle)

        # Colab returns WANDB_API_KEY
        fake_colab_userdata = MagicMock()
        def colab_get(key):
            return "wb_from_colab" if key == "WANDB_API_KEY" else ""
        fake_colab_userdata.get.side_effect = colab_get
        fake_colab = types.SimpleNamespace(userdata=fake_colab_userdata)
        monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(colab=fake_colab))
        monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

        rv._load_secrets()

        assert os.environ["HF_TOKEN"] == "hf_from_kaggle"
        assert os.environ["WANDB_API_KEY"] == "wb_from_colab"

    def test_skips_colab_when_all_already_loaded_by_kaggle(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        fake_kaggle_client = MagicMock()
        fake_kaggle_client.get_secret.side_effect = lambda k: f"kaggle_{k}"
        fake_kaggle = types.SimpleNamespace(UserSecretsClient=lambda: fake_kaggle_client)
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_kaggle)

        colab_called = []
        fake_colab = types.SimpleNamespace(userdata=MagicMock(get=lambda k: colab_called.append(k) or ""))
        monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(colab=fake_colab))
        monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

        rv._load_secrets()

        assert colab_called == [], "Colab should not be queried when Kaggle covered all secrets"

    def test_no_crash_when_both_providers_unavailable(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delitem(sys.modules, "kaggle_secrets", raising=False)
        monkeypatch.delitem(sys.modules, "google", raising=False)
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        rv._load_secrets()  # must not raise


# ════════════════════════════════════════════════════════════════════════════
# 8. list_remote_checkpoints
# ════════════════════════════════════════════════════════════════════════════

class TestListRemoteCheckpoints:
    def _make_cfg(self, hf_token="tok", ckpt_subdir="checkpoints/bitmamba_scout"):
        return {
            "repo_id": "org/repo",
            "hf_token": hf_token,
            "ckpt_subdir": ckpt_subdir,
        }

    def test_filters_to_subdir_and_pt_extension(self):
        fake_api = MagicMock()
        fake_api.list_repo_files.return_value = [
            "checkpoints/bitmamba_scout/step_001000.pt",
            "checkpoints/bitmamba_scout/step_002000.pt",
            "checkpoints/bitmamba_parent/step_001000.pt",  # wrong subdir
            "checkpoints/bitmamba_scout/config.json",       # wrong extension
            "README.md",
        ]
        fake_hf = MagicMock(HfApi=MagicMock(return_value=fake_api))

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            result = rv.list_remote_checkpoints(self._make_cfg())

        assert result == [
            "checkpoints/bitmamba_scout/step_001000.pt",
            "checkpoints/bitmamba_scout/step_002000.pt",
        ]

    def test_results_sorted_by_step(self):
        fake_api = MagicMock()
        fake_api.list_repo_files.return_value = [
            "checkpoints/bitmamba_scout/step_300000.pt",
            "checkpoints/bitmamba_scout/step_010000.pt",
            "checkpoints/bitmamba_scout/step_100000.pt",
        ]
        fake_hf = MagicMock(HfApi=MagicMock(return_value=fake_api))

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            result = rv.list_remote_checkpoints(self._make_cfg())

        steps = [rv._step_from_path(f) for f in result]
        assert steps == sorted(steps)

    def test_empty_token_passed_as_none(self):
        fake_api = MagicMock()
        fake_api.list_repo_files.return_value = []
        fake_hf_cls = MagicMock(return_value=fake_api)
        fake_hf = MagicMock(HfApi=fake_hf_cls)

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            rv.list_remote_checkpoints(self._make_cfg(hf_token=""))

        fake_hf_cls.assert_called_once_with(token=None)

    def test_non_empty_token_forwarded(self):
        fake_api = MagicMock()
        fake_api.list_repo_files.return_value = []
        fake_hf_cls = MagicMock(return_value=fake_api)
        fake_hf = MagicMock(HfApi=fake_hf_cls)

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            rv.list_remote_checkpoints(self._make_cfg(hf_token="hf_abc"))

        fake_hf_cls.assert_called_once_with(token="hf_abc")
