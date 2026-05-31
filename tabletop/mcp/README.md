# mcp/  —  Claude bridge

A small Model Context Protocol server that exposes the tabletop pipeline
as named tools Claude can call.

## What it exposes

| Tool | What it does |
|---|---|
| `tabletop_concept_image` | Generate a FLUX concept image from a prompt. Returns PNG path. |
| `tabletop_generate_mesh` | Run Sparc3D (default) or Hunyuan3D on an image. Returns GLB. |
| `tabletop_cleanup_for_print` | Headless Blender pass. Returns STL + cleanup JSON. |
| `tabletop_full_pipeline` | One-shot: prompt → STL. Returns all paths. |
| `tabletop_analyze_for_slicing` | Orientation + support advisory for a finished STL. |
| `tabletop_list_models` | Report which engines + LoRAs are available locally. |

## Install

```powershell
# Windows
.\install\06_install_mcp.ps1
```
```bash
# Linux
bash install/06_install_mcp.sh
```

## Register with Claude Desktop

Add this block to your Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`
on Windows, `~/Library/Application Support/Claude/claude_desktop_config.json` on Mac/Linux):

```json
{
  "mcpServers": {
    "tabletop": {
      "command": "A:/Project False Face/3D generation/tabletop/mcp/venv/Scripts/python.exe",
      "args": [
        "A:/Project False Face/3D generation/tabletop/mcp/server.py"
      ]
    }
  }
}
```

(Linux path: `tabletop/mcp/venv/bin/python`.)

Restart Claude Desktop. The tools should appear in the tool list. Try:

> "Make me a stone watchtower for tabletop terrain, 100mm tall."

Claude should call `tabletop_full_pipeline` and walk you through it.

## Register with Claude Code

Drop into `.mcp.json` at the project root:

```json
{
  "mcpServers": {
    "tabletop": {
      "type": "stdio",
      "command": "A:/Project False Face/3D generation/tabletop/mcp/venv/Scripts/python.exe",
      "args": ["A:/Project False Face/3D generation/tabletop/mcp/server.py"]
    }
  }
}
```

## Customizing tool surface

`mcp/tools.py` declares the tools — each is a thin wrapper around the
`pipeline.orchestrate` functions. Add or remove tools by editing that
file. Keep tools focused (one task each); Claude works better with a
clear small toolset.
