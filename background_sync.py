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

api = HfApi(token=HF_TOKEN)
try:
    api.create_repo(REPO_ID, repo_type="model", private=True)
except Exception:
    pass 

uploaded_files = set()
print(f"Watching {CHECKPOINT_DIR}/ for new .pt files...")

while True:
    if os.path.exists(CHECKPOINT_DIR):
        for root, _, files in os.walk(CHECKPOINT_DIR):
            for file in files:
                if file.endswith(".pt"):
                    filepath = os.path.join(root, file)
                    if filepath not in uploaded_files:
                        time.sleep(60) 
                        try:
                            path_in_repo = os.path.relpath(filepath, start=".")
                            api.upload_file(
                                path_or_fileobj=filepath, path_in_repo=path_in_repo,
                                repo_id=REPO_ID, repo_type="model"
                            )
                            uploaded_files.add(filepath)
                            print(f"Backed up {file}")
                        except Exception as e:
                            print(f"Upload failed {file}: {e}")
    time.sleep(300) 