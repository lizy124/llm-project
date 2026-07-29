#!/usr/bin/env bash
set -euo pipefail

# Install Node.js LTS from NodeSource, then install Claude Code CLI.
# This script is based on utils/install_log.log.

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run this script as root, or with sudo." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "==> Configure NodeSource LTS repository"
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -

echo "==> Install Node.js"
apt-get install -y nodejs

echo "==> Verify Node.js and npm"
node --version
npm --version

echo "==> Install Claude Code CLI"
npm install -g @anthropic-ai/claude-code

echo "==> Verify Claude Code CLI"
claude --version

echo "==> Installation completed"
