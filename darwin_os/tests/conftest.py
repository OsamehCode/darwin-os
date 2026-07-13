"""Pytest fixtures & configuration for DARWIN-OS.

The `headless` marker was failing because pytest requires custom markers
to be declared in the configuration. We now declare it both ways
(pyproject.toml's [tool.pytest.ini_options].markers AND here).
"""

import pytest


def pytest_configure(config):
    """Register custom markers so pytest stops complaining."""
    config.addinivalue_line(
        "markers",
        "headless: test runs without a display server (always true in this project)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark all tests as headless-safe."""
    headless_marker = pytest.mark.headless
    for item in items:
        # add_marker respects existing markers, but it's fine to apply again.
        item.add_marker(headless_marker)
