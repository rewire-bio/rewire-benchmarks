"""The local backend: this machine, CPU only, no cost."""

from __future__ import annotations

from pathlib import Path

from rewirebench import receipts
from rewirebench.backends import base
from rewirebench.receipts import Receipt, StartedBy
from rewirebench.recipes import LoadedRecipe


class LocalBackend:
    name = "local"

    def plan(
        self, loaded: LoadedRecipe, *, recipes_dir: Path, output: str | Path | None = None,
        inputs: str | Path | None = None, started_by: StartedBy = "human",
        kg_release_id: str | None = None,
    ) -> base.Plan:
        recipe = loaded.recipe
        reason = (None if recipe.backends.local is not None
                  else f"Recipe {recipe.id} does not name the local backend")
        reason = reason or base.pending_reason(recipe)
        hardware, hardware_note = base.hardware_needs(recipe)
        expected, basis = base.expected_wall(recipe)
        return base.make_plan(
            recipe=base.recipe_pointer(loaded, recipes_dir),
            backend="local",
            platform=None,
            image=None,
            hardware=hardware,
            hardware_note=hardware_note,
            expected_wall_seconds=expected,
            expected_wall_seconds_basis=basis,
            time_limit_seconds=base.time_limit(recipe),
            quote=base.owned_hardware_quote(),
            confirmation=base.Confirmation(required=False),
            command=base.run_arguments(
                recipe.id, recipes_dir=str(recipes_dir),
                output=str(output) if output is not None else base.OUTPUT_PLACEHOLDER,
                backend="local", started_by=started_by,
                inputs=str(inputs) if inputs is not None else None,
                kg_release_id=kg_release_id,
            ),
            runnable=reason is None,
            not_runnable_reason=reason,
            notes=["Runs in this Python environment with CUDA devices hidden.",
                   "hardware_floor is checked for host memory only."],
        )

    def run(
        self, loaded: LoadedRecipe, *, recipes_dir: Path, output: str | Path,
        inputs: str | Path | None = None, started_by: StartedBy | None = None,
        kg_release_id: str | None = None,
    ) -> Receipt:
        return receipts.run_recipe(
            loaded, output=output, recipes_dir=recipes_dir, inputs=inputs,
            started_by=started_by, kg_release_id=kg_release_id, backend="local",
        )
