"""Capture local runtime capabilities without silently changing devices."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from pydantic import BaseModel

from listing_to_reel.core.config import DeviceTarget


class DeviceCapabilities(BaseModel):
    mps_available: bool
    cuda_available: bool


class EnvironmentSnapshot(BaseModel):
    python_version: str
    platform: str
    machine: str
    git_commit_sha: str | None
    torch_version: str | None
    capabilities: DeviceCapabilities


class RequestedDeviceUnavailableError(RuntimeError):
    """Raised when a requested profile cannot run on the current machine."""


def _git_commit_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _torch_module() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def capabilities_from_torch(torch_module: Any | None) -> DeviceCapabilities:
    """Read PyTorch capabilities defensively so phase-0 tests need no GPU package."""
    if torch_module is None:
        return DeviceCapabilities(mps_available=False, cuda_available=False)

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    cuda_backend = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda_backend and cuda_backend.is_available())
    return DeviceCapabilities(mps_available=mps_available, cuda_available=cuda_available)


def assert_device_available(device: DeviceTarget, capabilities: DeviceCapabilities) -> None:
    """Fail explicitly instead of silently falling back to a different backend."""
    available = (
        capabilities.mps_available if device is DeviceTarget.MPS else capabilities.cuda_available
    )
    if not available:
        raise RequestedDeviceUnavailableError(
            f"Requested {device.value!r} runtime is unavailable; refusing to fall back to CPU."
        )


def collect_environment_snapshot() -> EnvironmentSnapshot:
    """Return stable system facts for a future run record."""
    torch_module = _torch_module()
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None

    return EnvironmentSnapshot(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        git_commit_sha=_git_commit_sha(),
        torch_version=torch_version,
        capabilities=capabilities_from_torch(torch_module),
    )
