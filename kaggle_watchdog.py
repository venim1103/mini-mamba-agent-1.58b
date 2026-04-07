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
import subprocess

REPO_DIR = "/kaggle/working/mini-mamba-agent-1.58b"


def set_repo_directory(repo_dir: str = REPO_DIR, *, logger=print) -> bool:
    """Switch to the repository directory if it exists."""
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)
        logger(f"Directory set to: {os.getcwd()}")
        return True
    logger("Repository not found! Please run Cell 1 first to clone the code.")
    return False


def load_kaggle_secrets(*, logger=print) -> dict[str, str]:
    """Fetch secrets from Kaggle and export required env vars."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError("kaggle_secrets is only available in Kaggle notebooks") from exc

    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
    repo_id = user_secrets.get_secret("REPO_ID")

    os.environ["HF_TOKEN"] = hf_token
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["REPO_ID"] = repo_id

    logger("Loaded HF_TOKEN, WANDB_API_KEY, and REPO_ID from Kaggle secrets")
    return {"HF_TOKEN": hf_token, "WANDB_API_KEY": wandb_api_key, "REPO_ID": repo_id}


def start_background_sync(*, log_path: str = "sync_logs.txt"):
    """Start background sync script and redirect output to a log file."""
    logger_msg = "Starting background sync watchdog..."
    print(logger_msg)
    child_env = os.environ.copy()
    # Force immediate line flushing so Kaggle log files are populated in real time.
    child_env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "a", buffering=1) as log_file:
        log_file.write("[watchdog] launching background_sync.py\n")
        log_file.flush()
        return subprocess.Popen(
            ["python", "-u", "background_sync.py"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )


def run_sync_self_check(*, log_path: str = "sync_logs.txt") -> int:
    """Run one-shot uploader diagnostics before launching the long-running sync loop."""
    print("Running background sync self-check...")
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "a", buffering=1) as log_file:
        log_file.write("[watchdog] running background_sync.py --self-check\n")
        log_file.flush()
        result = subprocess.run(
            ["python", "-u", "background_sync.py", "--self-check"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
            check=False,
        )
    print(f"Background sync self-check exit code: {result.returncode}")
    return result.returncode


def main():
    if not set_repo_directory(REPO_DIR):
        return 1
    load_kaggle_secrets()
    check_code = run_sync_self_check(log_path="sync_logs.txt")
    if check_code != 0:
        print("Background sync self-check failed. See sync_logs.txt for details.")
        return 2
    proc = start_background_sync(log_path="sync_logs.txt")
    pid = getattr(proc, "pid", "unknown")
    print(f"Background sync started with PID {pid}. Tail sync_logs.txt to monitor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
