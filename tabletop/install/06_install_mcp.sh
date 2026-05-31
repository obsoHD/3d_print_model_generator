#!/usr/bin/env bash
# 06_install_mcp.sh — install the Claude MCP bridge.

set -euo pipefail
tabletop_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$tabletop_root/mcp"

[ -d venv ] || python3.11 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "MCP bridge installed. Read tabletop/mcp/README.md for the Claude"
echo "registration snippet."
