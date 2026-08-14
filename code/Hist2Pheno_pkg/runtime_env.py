## 2026.08.13, put the code for configure_notebook_runtime here

"""Small runtime helpers for notebooks/CLI.

This module centralizes environment setup that must happen before importing
heavy libraries (e.g. torch), while keeping notebooks clean.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable


def configure_notebook_runtime(
    *,
    cuda_visible_devices_env: str = "CUDA_VISIBLE_DEVICES",
    cuda_device_env: str = "NCRT_CUDA_DEVICE",
    default_cuda_device: str = "0",
    tmpdir_env: str = "NCRT_TMPDIR",
    default_tmpdir: str | None = None,
    set_temp_envs: Iterable[str] = ("TMPDIR", "TEMP", "TMP"),
    apply_tempfile_tempdir: bool = True,
    verbose: bool = True,
) -> dict[str, str]:
    """Pin GPU + configure a writable temp directory.

    - **GPU pinning**: if ``CUDA_VISIBLE_DEVICES`` is unset, set it from
      ``NCRT_CUDA_DEVICE`` (default "0"). This matches the notebook behavior
      used across datasets.
    - **Temp dir**: set TMPDIR/TEMP/TMP to ``NCRT_TMPDIR`` (default ~/ssd2/tmp),
      and optionally set ``tempfile.tempdir``.
    """

    # Pin to ONE physical GPU before any torch import.
    if cuda_visible_devices_env not in os.environ:
        os.environ[cuda_visible_devices_env] = os.environ.get(
            cuda_device_env, default_cuda_device
        )
    cuda_visible = os.environ.get(cuda_visible_devices_env, "")
    physical_gpu = cuda_visible.split(",")[0].strip() if cuda_visible else ""

    # Configure a writable temp directory (PyTorch/torch.distributed can use /tmp).
    if default_tmpdir is None:
        default_tmpdir = str(Path.home() / "ssd2" / "tmp")
    tmpdir = os.environ.get(tmpdir_env, default_tmpdir)
    try:
        os.makedirs(tmpdir, exist_ok=True)
        for k in set_temp_envs:
            os.environ[k] = tmpdir
        if apply_tempfile_tempdir:
            tempfile.tempdir = tmpdir
    except OSError:
        # Keep going even if the temp dir cannot be created.
        pass

    if verbose:
        if physical_gpu:
            print(
                f"{cuda_visible_devices_env}={os.environ.get(cuda_visible_devices_env)} "
                f"(physical GPU {physical_gpu})"
            )
        else:
            print(f"{cuda_visible_devices_env}={os.environ.get(cuda_visible_devices_env, '')}")

    return {
        "cuda_visible_devices": os.environ.get(cuda_visible_devices_env, ""),
        "physical_gpu": physical_gpu,
        "tmpdir": tmpdir,
    }

