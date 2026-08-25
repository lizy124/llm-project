#!/bin/bash
echo "=== vllm-ascend repo ==="
cd /vllm-workspace/vllm-ascend || exit 1
git remote -v | head -4
git branch --show-current
git log --oneline -1
git status --short | head -10
echo "=== vllm repo ==="
cd /vllm-workspace/vllm || exit 1
git remote -v | head -4
git branch --show-current
git log --oneline -1
git status --short | head -10
