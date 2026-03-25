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

"""Quick GPU smoke test for BitMamba/Triton path.

Usage:
    python gpu_smoke_test.py
"""

import sys
import torch

from model import BitMambaLLM


def main():
    print(f"torch={torch.__version__}, cuda_compiled={torch.version.cuda}")

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False")
        sys.exit(1)

    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model = BitMambaLLM(
        vocab_size=64000,
        dim=512,
        n_layers=24,
        d_state=64,
        expand=2,
        use_attn=True,
    ).to(device)
    model.eval()

    x = torch.randint(0, 64000, (1, 32), device=device)

    with torch.no_grad():
        y = model(x)

    print("PASS")
    print(f"input_shape={tuple(x.shape)} output_shape={tuple(y.shape)} device={y.device}")


if __name__ == "__main__":
    main()
