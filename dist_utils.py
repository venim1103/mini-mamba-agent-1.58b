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

"""
Distributed training utilities.

Auto-detects multi-GPU setups and handles DDP initialization/teardown.

Single-GPU:
    python train.py              # works as before

Multi-GPU (2x T4 on Kaggle, etc.):
    torchrun --nproc_per_node=2 train.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initialize distributed training if launched via torchrun, else single-GPU.

    Returns:
        rank:       int — global rank (0 for single-GPU)
        local_rank: int — local GPU index (0 for single-GPU / CPU)
        world_size: int — total number of processes (1 for single-GPU)
        device:     str — device string (e.g. 'cuda', 'cuda:1', 'cpu')
    """
    if "RANK" in os.environ:
        # Launched via torchrun / torch.distributed.launch
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if rank == 0:
            print(f"[DDP] Initialized {world_size} processes (nccl)")
    elif torch.cuda.is_available():
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda"
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            script = os.path.basename(sys.argv[0]) if sys.argv else "train.py"
            print(f"[dist_utils] {n_gpus} GPUs detected but running single-process.")
            print(f"[dist_utils] For multi-GPU, launch with:  "
                  f"torchrun --nproc_per_node={n_gpus} {script}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cpu"

    # Seed differently per rank so each process samples different data
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return rank, local_rank, world_size, device


def cleanup_distributed():
    """Destroy the process group if distributed training was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True if this is rank 0 (or not distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def wrap_model_ddp(model, local_rank):
    """Wrap a model in DistributedDataParallel when distributed is active.

    Uses static_graph=True for compatibility with torch.compile / CUDA graphs.
    Returns the model unwrapped if not distributed.
    """
    if dist.is_initialized():
        return DDP(model, device_ids=[local_rank], static_graph=True)
    return model


def unwrap_model(model):
    """Return the underlying module, stripping DDP wrapper if present."""
    if isinstance(model, DDP):
        return model.module
    return model


def barrier():
    """Synchronize all processes. No-op when not distributed."""
    if dist.is_initialized():
        dist.barrier()


def get_world_size():
    """Return the world size (1 when not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
