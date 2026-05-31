# 06_install_mcp.ps1
# Install the Claude MCP bridge for the tabletop pipeline.
#
# After this, register the MCP server in Claude Desktop (mcp.json) or
# Claude Code (.mcp.json) per the README in tabletop/mcp/.

$ErrorActionPreference = "Stop"
$tabletop_root = Split-Path -Parent $PSScriptRoot
$mcp_dir       = Join-Path $tabletop_root "mcp"

Push-Location $mcp_dir
try {
    if (-not (Test-Path "venv")) {
        python -m venv venv
    }
    & .\venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    Write-Host ""
    Write-Host "MCP bridge installed. Next:" -ForegroundColor Green
    Write-Host "  Read tabletop\mcp\README.md for the Claude Desktop /"
    Write-Host "  Claude Code registration snippet."
}
finally {
    Pop-Location
}
