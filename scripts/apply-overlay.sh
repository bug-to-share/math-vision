#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo 'Usage: bash scripts/apply-overlay.sh <clean pinned veRL checkout>' >&2
    exit 2
fi
release=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkout=$(cd "$1" && pwd)

test "$(git -C "$checkout" rev-parse HEAD)" = 483b8a009ba3a97563edee3a19887e4862b8094a
test "$(git -C "$checkout/recipe" rev-parse --show-toplevel)" = "$checkout/recipe"
test "$(git -C "$checkout/recipe" rev-parse HEAD)" = e7f889574b8301cc0f0fc1d57c6d67f31ffeb689

while IFS= read -r -d '' file; do
    relative=${file#"$release/overlay/"}
    if [[ -e "$checkout/$relative" || -L "$checkout/$relative" ]]; then
        printf 'Overlay destination already exists: %s\n' "$relative" >&2
        exit 1
    fi
done < <(find "$release/overlay" -type f -print0)

git -C "$checkout" apply --check "$release/patches/verl-compat.patch"
git -C "$checkout/recipe" apply --check "$release/patches/verl-recipe-compat.patch"
git -C "$checkout" apply "$release/patches/verl-compat.patch"
git -C "$checkout/recipe" apply "$release/patches/verl-recipe-compat.patch"
cp -a "$release/overlay/." "$checkout/"
