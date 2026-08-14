## 2026.08.13, put the code for configure_torch_runtime here
##             simplify the code to fit CODEX ESCC and Xenium Lung dataset

"""Torch runtime setup utilities for notebooks/CLI.

These helpers keep notebooks clean while preserving behavior:
- verify CUDA visibility after CUDA_VISIBLE_DEVICES masking
- select logical cuda:0 (the only visible device)
- create AMP GradScaler
- configure reproducibility knobs
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TorchRuntime:
    device: "object"  # torch.device
    amp_scaler: "object | None"  # torch.amp.GradScaler or None
    cuda_device_index: int | None
    torch_cuda_index: int
    physical_gpu: str


def configure_torch_runtime(
    *,
    seed: int = 42,
    torch_cuda_index: int = 0,
    allow_cpu_env: str = "NCRT_ALLOW_CPU_TRAIN",
    physical_gpu_envs: tuple[str, str] = ("CUDA_VISIBLE_DEVICES", "NCRT_CUDA_DEVICE"),
    cublas_workspace_config: str = ":4096:8",
    deterministic: bool = True,
    verbose: bool = True,
) -> TorchRuntime:
    """Configure torch device + determinism. Call after GPU pinning is done.

    Assumes that after CUDA_VISIBLE_DEVICES masking, the desired physical GPU is
    visible as logical ``cuda:0``.
    """

    import torch  # local import to keep module import lightweight
    import random
    import numpy as np

    physical_gpu = os.environ.get(physical_gpu_envs[0], os.environ.get(physical_gpu_envs[1], "0"))

    if verbose:
        print(sys.executable)
        print(
            f"torch {torch.__version__} | built with CUDA: {torch.version.cuda} | "
            f"cuda.is_available: {torch.cuda.is_available()}"
        )
        print(
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} -> "
            f"physical GPU(s) {physical_gpu!r}"
        )

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if verbose and n_gpu != 1:
            print(
                f"WARNING: expected 1 visible GPU after CUDA_VISIBLE_DEVICES mask, saw {n_gpu}. "
                "Restart kernel and run the GPU pinning cell before this cell."
            )
        torch.cuda.set_device(torch_cuda_index)
        device = torch.device(f"cuda:{torch_cuda_index}")
        cuda_device_index = torch_cuda_index
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
        amp_scaler = torch.amp.GradScaler("cuda")
        if verbose:
            print(
                f"Using physical GPU {physical_gpu} as logical cuda:{torch_cuda_index} | "
                f"{torch.cuda.get_device_name(torch_cuda_index)}"
            )
    else:
        allow_cpu = os.environ.get(allow_cpu_env, "").strip().lower() in ("1", "true", "yes")
        if allow_cpu:
            device, amp_scaler, cuda_device_index = torch.device("cpu"), None, None
            if verbose:
                print("WARNING: CUDA unavailable, training on CPU (set NCRT_ALLOW_CPU_TRAIN=1).")
        else:
            raise RuntimeError(
                "CUDA unavailable. Please use a CUDA-enabled PyTorch build, "
                "or set NCRT_ALLOW_CPU_TRAIN=1 to run on CPU."
            )

    if verbose:
        if getattr(device, "type", None) == "cuda":
            print(
                f"device: {device} | physical GPU {physical_gpu} | "
                f"{torch.cuda.get_device_name(torch_cuda_index)}"
            )
        else:
            print(f"device: {device}")

    # Reproducibility knobs.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True)

    if verbose:
        print(f"Reproducibility configured with SEED={seed}")

    return TorchRuntime(
        device=device,
        amp_scaler=amp_scaler,
        cuda_device_index=cuda_device_index,
        torch_cuda_index=torch_cuda_index,
        physical_gpu=physical_gpu,
    )

