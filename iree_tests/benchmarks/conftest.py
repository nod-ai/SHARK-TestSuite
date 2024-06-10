import pytest

def pytest_addoption(parser):
    parser.addoption("--goldentime-rocm-ms", action="store", default=5565, type=int, help="Golden time to test benchmark")

@pytest.fixture
def goldentime_rocm(request):
    return request.config.getoption("--goldentime-rocm-ms")
