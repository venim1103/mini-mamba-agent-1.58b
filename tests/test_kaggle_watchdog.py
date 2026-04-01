import pytest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Mock heavy dependencies before importing
mock_modules = {
    'kaggle_secrets': MagicMock(),
    'kaggle_secrets.UserSecretsClient': MagicMock(),
    'subprocess': MagicMock(),
}
for name, obj in mock_modules.items():
    if name not in sys.modules:
        sys.modules[name] = obj


class TestKaggleWatchdog:
    """Test kaggle_watchdog.py execution."""

    def test_repo_dir_check(self):
        with patch("kaggle_watchdog.os.path.exists") as MockExists:
            MockExists.return_value = True
            
            import kaggle_watchdog
            
            MockExists.assert_called_with("/kaggle/working/mini-mamba-agent-1.58b")

    def test_repo_dir_not_found(self, capsys):
        with patch("kaggle_watchdog.os.path.exists", return_value=False):
            
            import kaggle_watchdog
            
            captured = capsys.readouterr()
            assert "Repository not found" in captured.out


class TestKaggleWatchdogConstants:
    """Test kaggle_watchdog.py constants."""

    def test_repo_dir_path(self):
        from kaggle_watchdog import repo_dir
        assert repo_dir == "/kaggle/working/mini-mamba-agent-1.58b"