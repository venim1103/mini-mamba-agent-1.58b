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

import torch
import os
from model import BitMambaLLM

def upscaler(small_ckpt_path, output_ckpt_path):
    print(f"Loading trained 40-Layer BitMamba Parent from {small_ckpt_path}...")
    checkpoint = torch.load(small_ckpt_path, map_location="cpu")
    small_state_dict = checkpoint['model_state_dict']
    
    print("Initializing new 64-Layer Upscaled BitMamba Model...")
    big_model = BitMambaLLM(
        vocab_size=64000, dim=1024, n_layers=64, d_state=128, expand=2
    )
    big_state_dict = big_model.state_dict()

    print("Transplanting weights via SOLAR duplication...")
    for key in big_state_dict.keys():
        if key.startswith("tok_embeddings.") or key.startswith("norm.") or key.startswith("output."):
            big_state_dict[key] = small_state_dict[key]
        elif key.startswith("layers."):
            parts = key.split('.')
            target_layer_idx = int(parts[1])
            
            if target_layer_idx < 32:
                source_layer_idx = target_layer_idx
            else:
                source_layer_idx = target_layer_idx - 24
                
            parts[1] = str(source_layer_idx)
            source_key = '.'.join(parts)
            big_state_dict[key] = small_state_dict[source_key]

    print("Loading mapped weights into the new model...")
    big_model.load_state_dict(big_state_dict)
    
    print(f"Saving Upscaled Checkpoint to {output_ckpt_path}...")
    torch.save({'step': 0, 'model_state_dict': big_state_dict}, output_ckpt_path)
    print("Done! You can now point train.py to this new checkpoint.")

if __name__ == "__main__":
    os.makedirs("checkpoints/upscaled", exist_ok=True)
    upscaler("checkpoints/bitmamba_parent/step_1000000.pt", "checkpoints/upscaled/step_000000_1B_mamba.pt")

