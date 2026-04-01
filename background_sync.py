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

import os
import time
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
# =================================================================
# CHANGE THIS TO YOUR HUGGING FACE USERNAME/REPO
REPO_ID = os.environ.get("REPO_ID")
# =================================================================
CHECKPOINT_DIR = "checkpoints"

def iter_new_checkpoint_files(checkpoint_dir: str, uploaded_files: set[str]):
    """Yield unseen .pt checkpoint file paths under checkpoint_dir."""
    if not os.path.exists(checkpoint_dir):
        return
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if not file.endswith(".pt"):
                continue
            filepath = os.path.join(root, file)
            if filepath not in uploaded_files:
                yield filepath


def upload_checkpoint(
    api: HfApi,
    filepath: str,
    repo_id: str,
    uploaded_files: set[str],
    *,
    sleep_before_upload: int = 60,
    sleep_fn=time.sleep,
    logger=print,
) -> bool:
    """Upload a checkpoint file and record it in uploaded_files on success."""
    sleep_fn(sleep_before_upload)
    try:
        path_in_repo = os.path.relpath(filepath, start=".")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded_files.add(filepath)
        logger(f"Backed up {os.path.basename(filepath)}")
        return True
    except Exception as exc:
        logger(f"Upload failed {os.path.basename(filepath)}: {exc}")
        return False


def sync_once(
    api: HfApi,
    repo_id: str,
    checkpoint_dir: str,
    uploaded_files: set[str],
    *,
    sleep_before_upload: int = 60,
    sleep_fn=time.sleep,
    logger=print,
) -> int:
    """Perform one scan/upload cycle and return number of successful uploads."""
    uploaded_count = 0
    for filepath in iter_new_checkpoint_files(checkpoint_dir, uploaded_files):
        if upload_checkpoint(
            api,
            filepath,
            repo_id,
            uploaded_files,
            sleep_before_upload=sleep_before_upload,
            sleep_fn=sleep_fn,
            logger=logger,
        ):
            uploaded_count += 1
    return uploaded_count


def run_sync_loop(
    api: HfApi,
    repo_id: str,
    checkpoint_dir: str = CHECKPOINT_DIR,
    *,
    poll_interval: int = 300,
    sleep_before_upload: int = 60,
    sleep_fn=time.sleep,
    logger=print,
):
    """Run continuous sync loop."""
    uploaded_files: set[str] = set()
    logger(f"Watching {checkpoint_dir}/ for new .pt files...")
    while True:
        sync_once(
            api,
            repo_id,
            checkpoint_dir,
            uploaded_files,
            sleep_before_upload=sleep_before_upload,
            sleep_fn=sleep_fn,
            logger=logger,
        )
        sleep_fn(poll_interval)


def main():
    if not REPO_ID:
        raise ValueError("REPO_ID environment variable is required")
    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(REPO_ID, repo_type="model", private=True)
    except Exception:
        # Repository may already exist; keep syncing in either case.
        pass
    run_sync_loop(api, REPO_ID, CHECKPOINT_DIR)


if __name__ == "__main__":
    main()