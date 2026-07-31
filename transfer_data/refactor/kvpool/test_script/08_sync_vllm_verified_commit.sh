#!/usr/bin/env bash
set -euo pipefail

TARGET_COMMIT="${VLLM_VERIFIED_COMMIT:-d02df748bf9efd99022f1a062597dc3cb3808485}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"

cd "${VLLM_DIR}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "vllm working tree is dirty; stop before switching commit." >&2
  git status --short >&2
  exit 1
fi

if ! git cat-file -e "${TARGET_COMMIT}^{commit}" 2>/dev/null; then
  git fetch --depth=1 origin "${TARGET_COMMIT}"
fi

git switch --detach "${TARGET_COMMIT}"

echo "VLLM_COMMIT=$(git rev-parse HEAD)"
git status --short --branch
