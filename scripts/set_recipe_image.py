"""Write a published image digest into a recipe's ``environment.images``.

The Runner image workflow (.github/workflows/image.yml) pushes the image and uploads
``image-digests.json``. Download that file from the run on main, then, from the repository root:

    uv run python scripts/set_recipe_image.py mfass-v2-kmer-baseline \
        --digests image-digests.json --environment core

That sets both platforms. To set one platform from a reference you copied by hand:

    uv run python scripts/set_recipe_image.py mfass-v2-kmer-baseline \
        --platform linux/arm64 --reference ghcr.io/rewire-bio/rewirebench@sha256:<64 hex>

Only the platform's entry under ``environment.images`` changes. Comments, key order and every
other byte of the file are kept. The plugin's copy of the recipe is written too, because
tests/test_agent_plugin.py requires the two to be identical. The script does not contact the
registry: check the digest first with ``docker buildx imagetools inspect <reference>``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from rewirebench.recipes import PLATFORMS, RecipeError, parse_recipe

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ghcr.io/rewire-bio/rewirebench"
REFERENCE = re.compile(re.escape(IMAGE) + r"@sha256:[a-f0-9]{64}")
# Notes written while no image existed. They are reported so a person can bring them up to date.
STALE = ("no container image is published", "No rewirebench image is published")


class ImageError(ValueError):
    """The recipe cannot be given this image."""


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def set_image(text: str, platform: str, reference: str) -> str:
    """``text`` with ``environment.images.<platform>`` replaced by ``reference``."""
    if platform not in PLATFORMS:
        raise ImageError(f"Unknown platform {platform}. Use one of: {', '.join(PLATFORMS)}")
    if not REFERENCE.fullmatch(reference):
        raise ImageError(f"{reference} is not {IMAGE}@sha256:<64 lowercase hex characters>")
    lines = text.splitlines(keepends=True)
    try:
        environment = next(i for i, line in enumerate(lines) if line.rstrip() == "environment:")
        images = next(i for i in range(environment + 1, len(lines))
                      if lines[i].rstrip() == "  images:" or _indent(lines[i]) == 0 and lines[i].strip())
        if lines[images].rstrip() != "  images:":
            raise StopIteration
        key = next(i for i in range(images + 1, len(lines))
                   if lines[i].rstrip() == f"    {platform}:" or _indent(lines[i]) <= 2 and lines[i].strip())
        if lines[key].rstrip() != f"    {platform}:":
            raise StopIteration
    except StopIteration:
        raise ImageError(f"No environment.images.{platform} block written as a nested mapping") from None
    end = key + 1
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > 4):
        end += 1
    while not lines[end - 1].strip():
        end -= 1
    changed = "".join(lines[:key + 1] + [f"      reference: {reference}\n"] + lines[end:])
    # Nothing but that one entry may differ, and the result must still be a valid recipe.
    before, after = yaml.safe_load(text), yaml.safe_load(changed)
    before["environment"]["images"][platform] = {"reference": reference}
    if before != after:
        raise ImageError("The edit would change more than the image entry; edit the recipe by hand")
    try:
        parse_recipe(changed)
    except RecipeError as error:
        raise ImageError(f"The recipe would no longer validate: {error}") from error
    return changed


def references(args: argparse.Namespace) -> dict[str, str]:
    """Platform to reference, from ``--reference`` or from the workflow's digest file."""
    if (args.reference is None) == (args.digests is None):
        raise ImageError("Give exactly one of --reference or --digests")
    if args.reference is not None:
        if args.platform is None:
            raise ImageError("--reference needs --platform")
        return {args.platform: args.reference}
    if args.environment is None:
        raise ImageError("--digests needs --environment (core or sequence)")
    published = json.loads(Path(args.digests).read_text())
    if args.environment not in published:
        raise ImageError(f"{args.digests} has no environment {args.environment}: {sorted(published)}")
    entry = published[args.environment]
    wanted = [args.platform] if args.platform else list(PLATFORMS)
    return {platform: f"{entry['image']}@{entry['platforms'][platform]}" for platform in wanted}


def recipe_paths(recipe_id: str, recipes_dir: Path, copies: list[Path]) -> list[Path]:
    found = sorted(recipes_dir.rglob(f"{recipe_id}.yaml"))
    if len(found) != 1:
        raise ImageError(f"Expected one {recipe_id}.yaml under {recipes_dir}, found {len(found)}")
    paths = [found[0]]
    for copy in copies:
        twin = copy / found[0].relative_to(recipes_dir)
        if not twin.is_file() or twin.read_bytes() != found[0].read_bytes():
            raise ImageError(f"{twin} is not an identical copy of {found[0]}")
        paths.append(twin)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recipe", help="Recipe id, for example mfass-v2-kmer-baseline")
    parser.add_argument("--platform", choices=PLATFORMS)
    parser.add_argument("--reference", help=f"{IMAGE}@sha256:<digest of the platform's manifest>")
    parser.add_argument("--digests", help="image-digests.json from the Runner image workflow")
    parser.add_argument("--environment", help="Environment in --digests: core or sequence")
    parser.add_argument("--recipes-dir", type=Path, default=ROOT / "recipes")
    parser.add_argument("--copy-dir", type=Path, action="append", default=None,
                        help="A copy of recipes/ to keep identical (default: the plugin's copy)")
    args = parser.parse_args(argv)
    copies = args.copy_dir
    if copies is None:
        copies = [ROOT / "plugins/bio-model-benchmarks/recipes"] if args.recipes_dir == ROOT / "recipes" else []
    try:
        paths = recipe_paths(args.recipe, args.recipes_dir, copies)
        text = paths[0].read_text()
        for platform, reference in references(args).items():
            text = set_image(text, platform, reference)
            print(f"{args.recipe} {platform}: {reference}")
    except (ImageError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        path.write_text(text)
        print(f"wrote {path}")
    for number, line in enumerate(text.splitlines(), 1):
        if any(note in line for note in STALE):
            print(f"check line {number}, written before an image existed: {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
