"""Bundle top-level recipes/ into sdist and wheel builds without breaking sdist-to-wheel."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RecipesBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return
        root = Path(self.root)
        embedded = root / "src" / "rewirebench" / "resources" / "recipes"
        if embedded.is_dir():
            return
        repo_recipes = (root / ".." / ".." / "recipes").resolve()
        if not (repo_recipes / "recipe.schema.json").is_file():
            raise FileNotFoundError(f"Expected recipes directory at {repo_recipes}")
        target = (
            "src/rewirebench/resources/recipes"
            if self.target_name == "sdist"
            else "rewirebench/resources/recipes"
        )
        build_data.setdefault("force_include", {})[str(repo_recipes)] = target
