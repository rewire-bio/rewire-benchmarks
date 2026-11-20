"""Fixtures shared by the backend, model and MCP server tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from recipe_fixtures import gpu_recipe_data, model_recipe_data, write_recipes


@pytest.fixture
def fixture_recipes(tmp_path: Path) -> Path:
    """A recipes directory holding the GPU fixture and the model-step recipe."""
    return write_recipes(tmp_path / "checkout", gpu_recipe_data(), model_recipe_data())
