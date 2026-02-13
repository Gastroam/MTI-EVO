import pytest
import shutil
import tempfile
import os
import sys

# Add src to path for tests to run without install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

@pytest.fixture
def temp_home():
    """
    Provides a temporary directory that simulates a user home or app data dir.
    Cleans up after test.
    """
    tmp = tempfile.mkdtemp(prefix="mti_test_")
    old_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        yield tmp
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)

@pytest.fixture
def mock_gpu():
    """
    Mock for GPU utilities to prevent actual CUDA calls during tests.
    """
    # This could patch mti_evo.tools.gpu_utils calls
    pass

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
