import pytest
from unittest.mock import MagicMock, patch, mock_open
import os


class TestKaggleWatchdog:
    """Test kaggle_watchdog.py execution."""

    def test_repo_dir_check(self):
        with patch("kaggle_watchdog.os.path.exists") as MockExists, \
             patch("kaggle_watchdog.os.chdir") as MockChdir, \
             patch("kaggle_watchdog.UserSecretsClient") as MockClient, \
             patch("kaggle_watchdog.subprocess.Popen"):
            
            MockExists.return_value = True
            
            mock_client = MagicMock()
            mock_client.get_secret.return_value = "fake_token"
            MockClient.return_value = mock_client
            
            import kaggle_watchdog
            
            MockExists.assert_called_with("/kaggle/working/mini-mamba-agent-1.58b")

    def test_repo_dir_not_found(self, capsys):
        with patch("kaggle_watchdog.os.path.exists", return_value=False), \
             patch("kaggle_watchdog.UserSecretsClient") as MockClient, \
             patch("kaggle_watchdog.subprocess.Popen"):
            
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            
            import kaggle_watchdog
            
            captured = capsys.readouterr()
            assert "Repository not found" in captured.out

    def test_fetches_hf_token(self):
        with patch("kaggle_watchdog.os.path.exists", return_value=True), \
             patch("kaggle_watchdog.os.chdir"), \
             patch("kaggle_watchdog.UserSecretsClient") as MockClient, \
             patch("kaggle_watchdog.subprocess.Popen"):
            
            mock_client = MagicMock()
            mock_client.get_secret.side_effect = lambda k: {"HF_TOKEN": "hf_xxx", "WANDB_API_KEY": "wandb_xxx"}[k]
            MockClient.return_value = mock_client
            
            import kaggle_watchdog
            
            assert os.environ.get("HF_TOKEN") == "hf_xxx"

    def test_fetches_wandb_key(self):
        with patch("kaggle_watchdog.os.path.exists", return_value=True), \
             patch("kaggle_watchdog.os.chdir"), \
             patch("kaggle_watchdog.UserSecretsClient") as MockClient, \
             patch("kaggle_watchdog.subprocess.Popen"):
            
            mock_client = MagicMock()
            mock_client.get_secret.side_effect = lambda k: {"HF_TOKEN": "hf_xxx", "WANDB_API_KEY": "wandb_xxx"}[k]
            MockClient.return_value = mock_client
            
            import kaggle_watchdog
            
            assert os.environ.get("WANDB_API_KEY") == "wandb_xxx"

    def test_launches_background_process(self):
        with patch("kaggle_watchdog.os.path.exists", return_value=True), \
             patch("kaggle_watchdog.os.chdir"), \
             patch("kaggle_watchdog.UserSecretsClient") as MockClient, \
             patch("kaggle_watchdog.subprocess.Popen") as MockPopen:
            
            mock_client = MagicMock()
            mock_client.get_secret.side_effect = lambda k: {"HF_TOKEN": "hf_xxx", "WANDB_API_KEY": "wandb_xxx"}[k]
            MockClient.return_value = mock_client
            
            import kaggle_watchdog
            
            MockPopen.assert_called_once()
            call_args = MockPopen.call_args[0][0]
            assert "background_sync.py" in call_args

    def test_opens_log_file(self):
        with patch("kaggle_watchdog.os.path.exists", return_value=True), \
             patch("kaggle_watchdog.os.chdir"), \
             patch("kaggle_watchdog.UserSecretsClient") as MockClient, \
             patch("kaggle_watchdog.subprocess.Popen") as MockPopen:
            
            mock_client = MagicMock()
            mock_client.get_secret.side_effect = lambda k: {"HF_TOKEN": "hf_xxx", "WANDB_API_KEY": "wandb_xxx"}[k]
            MockClient.return_value = mock_client
            
            import kaggle_watchdog
            
            MockPopen.assert_called_once()
            kwargs = MockPopen.call_args[1]
            assert "stdout" in kwargs


class TestKaggleWatchdogConstants:
    """Test kaggle_watchdog.py constants."""

    def test_repo_dir_path(self):
        from kaggle_watchdog import repo_dir
        assert repo_dir == "/kaggle/working/mini-mamba-agent-1.58b"
