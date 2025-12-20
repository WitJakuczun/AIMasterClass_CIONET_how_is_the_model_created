import subprocess
import pytest

def test_ruff_installed():
    """Verify that ruff is installed and accessible via CLI."""
    try:
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True, check=True)
        assert "ruff" in result.stdout.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.fail("Ruff is not installed or not in PATH")
