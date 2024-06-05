import pytest

def pytest_addoption(parser):
    parser.addoption("--goldentime", action="store", default=5565, type=int, help="Golden time to test benchmark")

@pytest.fixture
def goldentime(request):
    return request.config.getoption("--goldentime")
