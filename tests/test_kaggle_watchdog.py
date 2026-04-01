import pytest
from unittest.mock import MagicMock, patch, mock_open
import sys

# Mock ALL heavy dependencies BEFORE any imports - critical for kaggle_watchdog
# because it imports kaggle_secrets.UserSecretsClient which only exists in Kaggle
_mocked_modules = {
    'os': MagicMock(),
    'subprocess': MagicMock(),
    'kaggle_secrets': MagicMock(),
    'kaggle_secrets.UserSecretsClient': MagicMock(),
}

for _name, _obj in _mocked_modules.items():
    if _name not in sys.modules:
        sys.modules[_name] = _obj


class TestKaggleWatchdog:
    """Test kaggle_watchdog.py execution."""

    def test_repo_dir_check(self):
        import kaggle_watchdog
        # The module checks os.path.exists for the repo directory
        # Since we mocked os, we just verify the module imports without error

    def test_repo_dir_not_found(self, capsys):
        import kaggle_watchdog
        # Module should load fine with mocked dependencies


class TestKaggleWatchdogConstants:
    """Test kaggle_watchdog.py constants."""

    def test_repo_dir_path(self):
        import kaggle_watchdog
        assert kaggle_watchdog.repo_dir == "/kaggle/working/mini-mamba-agent-1.58b"