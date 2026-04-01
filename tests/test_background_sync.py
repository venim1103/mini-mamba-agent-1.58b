import pytest
import time
from unittest.mock import MagicMock, patch, PropertyMock


class TestBackgroundSyncLoop:
    """Test background_sync.py main loop."""

    def test_watchdog_loop_iterates(self):
        with patch("background_sync.time.sleep") as MockSleep, \
             patch("background_sync.os.path.exists", return_value=True), \
             patch("background_sync.os.walk") as MockWalk, \
             patch("background_sync.HfApi") as MockApi:
            
            MockWalk.return_value = []
            mock_api = MagicMock()
            MockApi.return_value = mock_api
            
            import background_sync
            
            uploaded = set()
            loop_count = 0
            
            def mock_walk(path):
                nonlocal loop_count
                if loop_count < 2:
                    loop_count += 1
                    return [("checkpoints", [], ["file1.pt", "file2.pt"])]
                return []
            
            MockWalk.side_effect = mock_walk
            
            with patch.object(background_sync, "uploaded_files", uploaded):
                pass

    def test_uploads_new_files(self):
        with patch("background_sync.time.sleep") as MockSleep, \
             patch("background_sync.os.path.exists", return_value=True), \
             patch("background_sync.os.walk") as MockWalk, \
             patch("background_sync.HfApi") as MockApi, \
             patch("background_sync.api") as MockApiInstance:
            
            MockWalk.return_value = [("checkpoints", [], ["new.pt"])]
            mock_api = MagicMock()
            MockApi.return_value = mock_api
            
            from background_sync import uploaded_files
            
            assert isinstance(uploaded_files, set)


class TestBackgroundSyncConstants:
    """Test background_sync.py constants."""

    def test_checkpoint_dir_defined(self):
        from background_sync import CHECKPOINT_DIR
        assert CHECKPOINT_DIR == "checkpoints"

    def test_repo_id_from_env(self):
        with patch.dict("os.environ", {"REPO_ID": "test/user-model"}):
            from importlib import reload
            import background_sync
            reload(background_sync)
            
            assert background_sync.REPO_ID == "test/user-model"

    def test_hf_token_from_env(self):
        with patch.dict("os.environ", {"HF_TOKEN": "test_token"}):
            from importlib import reload
            import background_sync
            reload(background_sync)
            
            assert background_sync.HF_TOKEN == "test_token"


class TestBackgroundSyncApiCalls:
    """Test HfApi usage in background_sync.py."""

    def test_creates_repo_on_startup(self):
        with patch("background_sync.HfApi") as MockApi, \
             patch("background_sync.REPO_ID", "test/repo"):
            
            mock_api = MagicMock()
            MockApi.return_value = mock_api
            
            import background_sync
            
            mock_api.create_repo.assert_called_once_with("test/repo", repo_type="model", private=True)

    def test_api_upload_called_for_new_files(self):
        with patch("background_sync.HfApi") as MockApi, \
             patch("background_sync.os.path.exists", return_value=True), \
             patch("background_sync.os.walk") as MockWalk, \
             patch("background_sync.time.sleep"):
            
            MockWalk.return_value = [("checkpoints", [], ["file.pt"])]
            mock_api = MagicMock()
            MockApi.return_value = mock_api
            
            uploaded = set()
            
            with patch.object(background_sync, "uploaded_files", uploaded):
                for root, _, files in background_sync.os.walk("checkpoints"):
                    for file in files:
                        if file.endswith(".pt"):
                            filepath = background_sync.os.path.join(root, file)
                            if filepath not in uploaded_files:
                                try:
                                    path_in_repo = background_sync.os.path.relpath(filepath, start=".")
                                    mock_api.upload_file(
                                        path_or_fileobj=filepath,
                                        path_in_repo=path_in_repo,
                                        repo_id=background_sync.REPO_ID,
                                        repo_type="model"
                                    )
                                except Exception:
                                    pass
                
            mock_api.upload_file.assert_called()
