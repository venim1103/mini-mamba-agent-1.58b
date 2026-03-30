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
from kaggle_secrets import UserSecretsClient

# 1. Force the notebook into the correct repository folder
repo_dir = "/kaggle/working/mini-mamba-agent-1.58b"
if os.path.exists(repo_dir):
    os.chdir(repo_dir)
    print(f"Directory set to: {os.getcwd()}")
else:
    print("Repository not found! Please run Cell 1 first to clone the code.")

# 2. Securely fetch API keys from Kaggle Secrets
user_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")

# 3. Launch watchdog in the background
print("Starting background sync watchdog...")
with open("sync_logs.txt", "w") as log_file:
    subprocess.Popen(
        ["python", "background_sync.py"], 
        stdout=log_file, 
        stderr=subprocess.STDOUT
    )
