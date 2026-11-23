#!/usr/bin/env bash
# Checks a built rewirebench image against the native environment of the same checkout.
#
#   containers/check_image.sh ENGINE IMAGE ENVIRONMENT NATIVE_JSON CONTAINER_JSON
#
# ENGINE is podman or docker. NATIVE_JSON is the output of
# `containers/parity.py --environment ENVIRONMENT` on the host. The image's parity output is
# written to CONTAINER_JSON and compared with it. Run from the repository root, after
# `uv sync`, on a host with the same architecture as the image.
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 ENGINE IMAGE ENVIRONMENT NATIVE_JSON CONTAINER_JSON" >&2
  exit 2
fi
engine=$1 image=$2 environment=$3 native=$4 container=$5
schema=$(mktemp)
trap 'rm -f "$schema"' EXIT

"$engine" run --rm --network none --read-only --tmpfs /tmp:rw,size=512m --entrypoint python \
  -v "$PWD:/workspace:ro" "$image" /workspace/containers/parity.py --environment "$environment" > "$container"
uv run --no-sync python containers/parity.py --compare "$native" "$container"
"$engine" run --rm --network none --read-only "$image" protocols
# The recipe and receipt commands need pydantic and PyYAML from the hash lock.
"$engine" run --rm --network none --read-only "$image" recipe schema > "$schema"
cmp "$schema" recipes/recipe.schema.json
"$engine" run --rm --network none --read-only "$image" recipe list
"$engine" run --rm --network none --read-only -v "$PWD/recipes:/recipes:ro" "$image" recipe validate --recipes-dir /recipes
