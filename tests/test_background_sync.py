import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock
import sys

# Mock heavy dependencies
mock_modules = {
    'huggingface_hub': MagicMock(),
    'huggingface_hub.HfApi': MagicMock(),
    'time': MagicMock(),
    'os': MagicMock(),
}
for name, obj in mock_modules.items():
    if name not in sys.modules:
        sys.modules[name] = obj


class TestBackgroundSyncConstants:
    """Test background_sync.py constants."""

    def test_checkpoint_dir_defined(self):
        from background_sync import CHECKPOINT_DIR
        assert CHECKPOINT_DIR == "checkpoints"


class TestBackgroundSyncSetup:
    """Test background_sync.py initialization."""

    def test_uploaded_files_is_set(self):
        from background_sync import uploaded_files
        assert isinstance(uploaded_files, set)

    def test_hf_token_from_env(self):
        with patch.dict("os.environ", {"HF_TOKEN": "test_token"}):
            from importlib import reload
            import background_sync
            reload(background_sync)
            
            assert background_sync.HF_TOKEN == "test_token"

    def test_repo_id_from_env(self):
        with patch.dict("os.environ", {"REPO_ID": "test/repo"}):
            from importlib import reload
            import background_sync
            reload(background_sync)
            
            assert background_sync.REPO_ID == "test/repo"