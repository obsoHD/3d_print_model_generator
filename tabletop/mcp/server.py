"""server.py - MCP server exposing the tabletop pipeline to Claude.

Uses the official `mcp` Python SDK in stdio mode. Each tool is a thin
wrapper around the pipeline orchestrator.

Run standalone for debug:
    python server.py --debug
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
sys.path.insert(0, str(TABLETOP_ROOT / "pipeline"))

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("error: mcp package not installed. run install/06_install_mcp.{ps1,sh}",
          file=sys.stderr)
    sys.exit(2)

# Pipeline imports
import orchestrate as op
import comfy_client
import sparc3d_client
import hunyuan_client
import blender_cleanup
import slicer_prep


# ---------------------------------------------------------------------------
# Tool implementations (thin wrappers)
# ---------------------------------------------------------------------------

async def tabletop_full_pipeline(prompt: str, kind: str = "mini",
                                  engine: str = "sparc3d",
                                  printer: str = "resin",
                                  scale_mm: float | None = None) -> dict:
    return op.run(prompt=prompt, kind=kind, engine=engine, printer=printer,
                   scale_mm=scale_mm)


async def tabletop_concept_image(prompt: str, kind: str = "mini",
                                  seed: int | None = None) -> dict:
    workflow = op.KIND_DEFAULTS[kind]["workflow"]
    out_path = comfy_client.generate(prompt=prompt, workflow=workflow,
                                       kind=kind, seed=seed)
    return {"image_path": out_path}


async def tabletop_generate_mesh(image_path: str, engine: str = "sparc3d",
                                  target_scale_mm: float = 32.0) -> dict:
    img = Path(image_path)
    out_dir = TABLETOP_ROOT / "outputs" / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{img.stem}.glb"
    if engine == "sparc3d":
        sparc3d_client.generate(image_path=str(img), out_path=str(out),
                                  target_scale_mm=target_scale_mm)
    elif engine == "hunyuan":
        hunyuan_client.generate(image_path=str(img),
                                 out_path=str(out), mesh_only=True)
    else:
        raise ValueError(f"unknown engine: {engine}")
    return {"mesh_path": str(out), "engine": engine}


async def tabletop_cleanup_for_print(mesh_path: str, printer: str = "resin",
                                       scale_mm: float = 32.0) -> dict:
    src = Path(mesh_path)
    out_stl = TABLETOP_ROOT / "outputs" / "stl" / f"{src.stem}.stl"
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    info = blender_cleanup.run(input_mesh=str(src), out_stl=str(out_stl),
                                 printer=printer, scale_mm=scale_mm)
    info["stl_path"] = str(out_stl)
    return info


async def tabletop_analyze_for_slicing(stl_path: str,
                                         printer: str = "resin") -> dict:
    return slicer_prep.analyze(stl_path, printer=printer)


async def tabletop_list_models() -> dict:
    return op.check_models()


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="tabletop_full_pipeline",
        description=("End-to-end: turn a prompt into a printable STL. "
                     "kind=mini|terrain|prop|scatter. engine=sparc3d (printable) "
                     "or hunyuan (preview only). printer=resin|fdm."),
        inputSchema={
            "type": "object",
            "properties": {
                "prompt":   {"type": "string"},
                "kind":     {"type": "string", "enum": ["mini","terrain","prop","scatter"], "default": "mini"},
                "engine":   {"type": "string", "enum": ["sparc3d","hunyuan","both"], "default": "sparc3d"},
                "printer":  {"type": "string", "enum": ["resin","fdm"], "default": "resin"},
                "scale_mm": {"type": "number"},
            },
            "required": ["prompt"],
        },
    ),
    Tool(
        name="tabletop_concept_image",
        description="Just the FLUX concept image step. Returns PNG path.",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "kind":   {"type": "string", "enum": ["mini","terrain","prop","scatter"], "default": "mini"},
                "seed":   {"type": "integer"},
            },
            "required": ["prompt"],
        },
    ),
    Tool(
        name="tabletop_generate_mesh",
        description="Image-to-3D step. Engine=sparc3d (printable) or hunyuan (preview).",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path":      {"type": "string"},
                "engine":          {"type": "string", "enum": ["sparc3d","hunyuan"], "default": "sparc3d"},
                "target_scale_mm": {"type": "number", "default": 32.0},
            },
            "required": ["image_path"],
        },
    ),
    Tool(
        name="tabletop_cleanup_for_print",
        description="Blender cleanup -> STL. printer=resin|fdm.",
        inputSchema={
            "type": "object",
            "properties": {
                "mesh_path": {"type": "string"},
                "printer":   {"type": "string", "enum": ["resin","fdm"], "default": "resin"},
                "scale_mm":  {"type": "number", "default": 32.0},
            },
            "required": ["mesh_path"],
        },
    ),
    Tool(
        name="tabletop_analyze_for_slicing",
        description="Analyze a finished STL and emit orientation + support advice.",
        inputSchema={
            "type": "object",
            "properties": {
                "stl_path": {"type": "string"},
                "printer":  {"type": "string", "enum": ["resin","fdm"], "default": "resin"},
            },
            "required": ["stl_path"],
        },
    ),
    Tool(
        name="tabletop_list_models",
        description="Report which generation engines + LoRAs are installed locally.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


DISPATCH = {
    "tabletop_full_pipeline":     tabletop_full_pipeline,
    "tabletop_concept_image":     tabletop_concept_image,
    "tabletop_generate_mesh":     tabletop_generate_mesh,
    "tabletop_cleanup_for_print": tabletop_cleanup_for_print,
    "tabletop_analyze_for_slicing": tabletop_analyze_for_slicing,
    "tabletop_list_models":       tabletop_list_models,
}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = Server("tabletop")


@server.list_tools()
async def _list_tools():
    return TOOLS


@server.call_tool()
async def _call_tool(name: str, arguments: dict):
    if name not in DISPATCH:
        raise ValueError(f"unknown tool: {name}")
    result = await DISPATCH[name](**(arguments or {}))
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
