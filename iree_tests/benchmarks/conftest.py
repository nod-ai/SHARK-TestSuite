import pytest

def pytest_addoption(parser):
    parser.addoption("--goldentime-rocm-e2e-ms", action="store", default=1657.2, type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-unet-ms", action="store", default=450.2, type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-clip-ms", action="store", default=18.7, type=float, help="Golden time to test benchmark")
    parser.addoption("--goldentime-rocm-vae-ms", action="store", default=288.5, type=float, help="Golden time to test benchmark")

@pytest.fixture
def goldentime_rocm_e2e(request):
    return request.config.getoption("--goldentime-rocm-e2e-ms")

@pytest.fixture
def goldentime_rocm_unet(request):
    return request.config.getoption("--goldentime-rocm-unet-ms")

@pytest.fixture
def goldentime_rocm_clip(request):
    return request.config.getoption("--goldentime-rocm-clip-ms")

@pytest.fixture
def goldentime_rocm_vae(request):
    return request.config.getoption("--goldentime-rocm-vae-ms")
