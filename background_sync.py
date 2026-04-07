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
import argparse
from huggingface_hub import HfApi

CHECKPOINT_DIR = "checkpoints"


def _log(message: str):
    """Emit timestamped log lines with immediate flush for notebook log files."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


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
        try:
            uploaded_count = sync_once(
                api,
                repo_id,
                checkpoint_dir,
                uploaded_files,
                sleep_before_upload=sleep_before_upload,
                sleep_fn=sleep_fn,
                logger=logger,
            )
            logger(
                f"Sync cycle complete: uploaded={uploaded_count}, tracked_total={len(uploaded_files)}"
            )
        except Exception as exc:
            logger(f"Sync cycle failed: {exc}")
        sleep_fn(poll_interval)


def run_self_check(api: HfApi, repo_id: str, checkpoint_dir: str = CHECKPOINT_DIR, *, logger=print) -> bool:
    """Run a one-shot diagnostics check for auth/repo access and checkpoint discovery."""
    logger("Running sync self-check...")
    try:
        who = api.whoami()
        user_name = who.get("name") if isinstance(who, dict) else str(who)
        logger(f"HF auth OK (user={user_name})")
    except Exception as exc:
        logger(f"HF auth failed: {exc}")
        return False

    try:
        api.create_repo(repo_id, repo_type="model", private=True)
        logger(f"Repo access OK: {repo_id}")
    except Exception as exc:
        # If create fails (e.g. already exists), explicitly verify we can access it.
        logger(f"Repo create/check returned: {exc}")
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            logger(f"Repo access confirmed via repo_info: {repo_id}")
        except Exception as repo_exc:
            logger(f"Repo access failed: {repo_exc}")
            return False

    if not os.path.exists(checkpoint_dir):
        logger(f"Checkpoint directory missing: {checkpoint_dir}")
        return True

    pt_files = []
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))

    logger(f"Checkpoint scan OK: found {len(pt_files)} .pt file(s) under {checkpoint_dir}")
    if pt_files:
        logger(f"Newest candidate: {max(pt_files, key=os.path.getmtime)}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Background checkpoint uploader")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run one-time diagnostics (auth/repo/checkpoint scan) and exit",
    )
    args = parser.parse_args(argv if argv is not None else [])

    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("REPO_ID")

    if not repo_id:
        raise ValueError("REPO_ID environment variable is required")

    _log("Starting background checkpoint sync")
    _log(f"Python PID={os.getpid()}")
    _log(f"HF token present={bool(hf_token)} | repo_id={repo_id}")

    api = HfApi(token=hf_token)

    if args.self_check:
        ok = run_self_check(api, repo_id, CHECKPOINT_DIR, logger=_log)
        _log(f"Self-check {'passed' if ok else 'failed'}")
        return 0 if ok else 2

    try:
        api.create_repo(repo_id, repo_type="model", private=True)
        _log("Ensured Hugging Face repo exists")
    except Exception as exc:
        # Repository may already exist; keep syncing in either case.
        _log(f"create_repo skipped/failed (continuing): {exc}")
    run_sync_loop(api, repo_id, CHECKPOINT_DIR, logger=_log)


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
