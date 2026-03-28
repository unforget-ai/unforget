#!/bin/bash
# Usage: ./scripts/release.sh patch|minor|major
#
# Examples:
#   ./scripts/release.sh patch   # 0.2.0 → 0.2.1
#   ./scripts/release.sh minor   # 0.2.0 → 0.3.0
#   ./scripts/release.sh major   # 0.2.0 → 1.0.0

set -euo pipefail

BUMP=${1:-patch}

if [[ "$BUMP" != "patch" && "$BUMP" != "minor" && "$BUMP" != "major" ]]; then
    echo "Usage: $0 patch|minor|major"
    exit 1
fi

# Get current version
CURRENT=$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

# Bump
case $BUMP in
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
    patch) PATCH=$((PATCH + 1)) ;;
esac

NEW="${MAJOR}.${MINOR}.${PATCH}"

echo "Bumping: $CURRENT → $NEW"

# Update pyproject.toml
sed -i '' "s/^version = \"$CURRENT\"/version = \"$NEW\"/" pyproject.toml

# Commit, tag, push
git add pyproject.toml
git commit -m "release: v${NEW}"
git tag "v${NEW}"
git push && git push origin "v${NEW}"

echo "Released v${NEW} — CI will publish to PyPI"
