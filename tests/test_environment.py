import pytest

from listing_to_reel.core.config import DeviceTarget
from listing_to_reel.core.environment import (
    DeviceCapabilities,
    RequestedDeviceUnavailableError,
    assert_device_available,
    capabilities_from_torch,
)


class AvailableBackend:
    @staticmethod
    def is_available() -> bool:
        return True


class FakeTorch:
    class backends:
        mps = AvailableBackend()

    cuda = AvailableBackend()


def test_detects_available_backends_from_torch() -> None:
    assert capabilities_from_torch(FakeTorch()) == DeviceCapabilities(
        mps_available=True, cuda_available=True
    )


def test_missing_torch_reports_no_accelerator() -> None:
    assert capabilities_from_torch(None) == DeviceCapabilities(
        mps_available=False, cuda_available=False
    )


def test_requested_unavailable_device_fails_without_cpu_fallback() -> None:
    with pytest.raises(RequestedDeviceUnavailableError, match="refusing to fall back"):
        assert_device_available(
            DeviceTarget.CUDA, DeviceCapabilities(mps_available=True, cuda_available=False)
        )
