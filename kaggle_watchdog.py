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
REQUIRED_SECRET_KEYS = ("HF_TOKEN", "WANDB_API_KEY", "REPO_ID")
REPO_ENV_VAR = "WATCHDOG_REPO_DIR"


def _candidate_repo_dirs(explicit_repo_dir: str | None = None) -> list[str]:
    candidates = []
    if explicit_repo_dir:
        candidates.append(explicit_repo_dir)
    env_repo_dir = os.environ.get(REPO_ENV_VAR)
    if env_repo_dir:
        candidates.append(env_repo_dir)
    candidates.append(REPO_DIR)
    candidates.append(os.getcwd())
    candidates.append(os.path.dirname(os.path.abspath(__file__)))
    # De-duplicate while preserving order.
    return list(dict.fromkeys(candidates))


def set_repo_directory(repo_dir: str | None = None, *, logger=print) -> bool:
    """Switch to a usable repository directory across Kaggle, Colab, or local setups."""
    for candidate in _candidate_repo_dirs(repo_dir):
        if os.path.exists(candidate):
            os.chdir(candidate)
            logger(f"Directory set to: {os.getcwd()}")
            return True
    logger(
        "Repository not found! Set WATCHDOG_REPO_DIR or run this script from the cloned repo root."
    )
    return False


def _load_kaggle_secrets() -> dict[str, str]:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    return {key: user_secrets.get_secret(key) for key in REQUIRED_SECRET_KEYS}


def _load_colab_secrets() -> dict[str, str]:
    from google.colab import userdata

    return {key: userdata.get(key) for key in REQUIRED_SECRET_KEYS}


def load_runtime_secrets(*, logger=print) -> dict[str, str]:
    """Resolve required secrets from env, then optional notebook providers."""
    resolved = {
        key: os.environ.get(key) for key in REQUIRED_SECRET_KEYS if os.environ.get(key)
    }
    missing = [key for key in REQUIRED_SECRET_KEYS if key not in resolved]
    if not missing:
        logger("Using secrets already present in environment variables")
        return resolved

    providers = (
        ("Kaggle secrets", _load_kaggle_secrets),
        ("Colab userdata", _load_colab_secrets),
    )
    for provider_name, provider_fn in providers:
        try:
            provider_values = provider_fn()
        except Exception:
            continue
        for key in missing:
            value = provider_values.get(key)
            if value:
                resolved[key] = value
                os.environ[key] = value
        missing = [key for key in REQUIRED_SECRET_KEYS if key not in resolved]
        if not missing:
            logger(f"Loaded {', '.join(REQUIRED_SECRET_KEYS)} from {provider_name}")
            return resolved

    raise RuntimeError(
        f"Missing required secrets: {', '.join(missing)}. "
        "Provide them via environment variables or notebook secret manager."
    )


def load_kaggle_secrets(*, logger=print) -> dict[str, str]:
    """Backward-compatible wrapper around cross-environment secret loading."""
    return load_runtime_secrets(logger=logger)


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
    if not set_repo_directory():
        return 1
    load_runtime_secrets()
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
