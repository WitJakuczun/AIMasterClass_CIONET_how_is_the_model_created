import subprocess

import pytest


def test_ruff_installed():
    """Verify that ruff is installed and accessible via CLI."""
    try:
        result = subprocess.run(
            ["ruff", "--version"], capture_output=True, text=True, check=True
        )
        assert "ruff" in result.stdout.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.fail("Ruff is not installed or not in PATH")


def test_ruff_configuration_present():
    """Verify that ruff configuration is present in pyproject.toml."""
    with open("pyproject.toml", "r") as f:
        content = f.read()
    assert "[tool.ruff]" in content
    assert "select =" in content
    # Check for specific rule categories
    for rule in ["E", "F", "I", "D", "S"]:
        assert f'"{rule}"' in content or f"'{rule}'" in content


def test_ruff_config_valid():
    """Verify that ruff recognizes the configuration as valid."""
    result = subprocess.run(
        ["ruff", "check", "--no-cache", "pyproject.toml"],
        capture_output=True,
        text=True,
    )
    # Even if there are lint errors in pyproject.toml (unlikely),
    # it shouldn't report "Failed to parse"
    assert "Failed to parse" not in result.stderr
    assert result.returncode in [0, 1]  # 0 if clean, 1 if lint errors found
