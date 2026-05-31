"""comfy_client - drive a local ComfyUI instance via its HTTP API.

ComfyUI exposes a websocket + HTTP API on http://127.0.0.1:8188 by default.
We POST a workflow JSON to /prompt, poll /history/<id> until the run is
done, then download the generated PNG.

The workflow JSON lives in tabletop/workflows/ and contains placeholders
that we substitute with the user's prompt + seed before posting.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.parse
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
WORKFLOWS_DIR = TABLETOP_ROOT / "workflows"

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"


def _load_workflow(name: str) -> dict:
    path = WORKFLOWS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"workflow not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _substitute(workflow: dict, prompt: str, seed: int | None) -> dict:
    """Walk the workflow JSON and replace placeholders.

    Placeholders we look for in any node's `inputs`:
      "$PROMPT$"   -> user prompt
      "$NEG$"      -> a fixed negative prompt (low quality, etc.)
      "$SEED$"     -> the seed (int)
    """
    NEG = ("low quality, blurry, watermark, text, logo, signature, "
           "multiple subjects, background clutter")
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for k, v in list(inputs.items()):
            if isinstance(v, str):
                if "$PROMPT$" in v:
                    inputs[k] = v.replace("$PROMPT$", prompt)
                elif "$NEG$" in v:
                    inputs[k] = v.replace("$NEG$", NEG)
                elif v == "$SEED$":
                    inputs[k] = seed
    return workflow


def _post_prompt(workflow: dict) -> str:
    data = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(
        f"{COMFY_BASE}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(
            f"ComfyUI /prompt returned HTTP {e.code}.\n"
            f"Response body:\n{detail}"
        ) from None
    if "prompt_id" not in body:
        raise RuntimeError(f"ComfyUI rejected prompt: {body}")
    return body["prompt_id"]


def _poll_history(prompt_id: str, timeout_s: float = 600) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        url = f"{COMFY_BASE}/history/{prompt_id}"
        with urllib.request.urlopen(url) as resp:
            body = json.loads(resp.read().decode())
        if prompt_id in body and body[prompt_id].get("status", {}).get("completed"):
            return body[prompt_id]
        time.sleep(1.0)
    raise TimeoutError(f"ComfyUI timeout after {timeout_s}s for prompt {prompt_id}")


def _download_image(image_node_output: dict, out_path: str) -> None:
    images = image_node_output.get("images", [])
    if not images:
        raise RuntimeError(f"no images in node output: {image_node_output}")
    img = images[0]
    q = urllib.parse.urlencode({
        "filename": img["filename"],
        "subfolder": img.get("subfolder", ""),
        "type": img.get("type", "output"),
    })
    url = f"{COMFY_BASE}/view?{q}"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(data)


def generate(prompt: str, workflow: str = "flux_mini_concept.json",
             kind: str = "mini", seed: int | None = None,
             out_path: str = None) -> str:
    """Run a ComfyUI workflow with the given prompt; save the resulting
    image to out_path. Returns out_path."""
    wf = _load_workflow(workflow)
    wf = _substitute(wf, prompt, seed)
    # ComfyUI expects only node dicts — strip comment/metadata keys
    wf = {k: v for k, v in wf.items() if isinstance(v, dict)}
    pid = _post_prompt(wf)
    history = _poll_history(pid)

    # Find the output image node (usually a SaveImage)
    outputs = history.get("outputs", {})
    image_output = None
    for nid, node_out in outputs.items():
        if "images" in node_out:
            image_output = node_out
            break
    if not image_output:
        raise RuntimeError("no image output in ComfyUI history")

    if out_path is None:
        out_path = str(TABLETOP_ROOT / "outputs" / "concepts" / f"{pid}.png")
    _download_image(image_output, out_path)
    return out_path


def free_vram() -> None:
    """Tell ComfyUI to unload its models from VRAM.

    Call this after concept image generation and before handing the GPU to
    HY3D. Frees ~6.5 GB SDXL VRAM so HY3D has the full 16 GB budget,
    dropping diffusion step time from ~60 s to ~9 s.
    """
    data = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(
        f"{COMFY_BASE}/free",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
        print("  [comfyui] VRAM freed (models unloaded)", flush=True)
    except Exception as e:
        print(f"  [comfyui] /free not available ({e}); continuing", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python comfy_client.py 'your prompt here'", file=sys.stderr)
        sys.exit(2)
    out = generate(prompt=sys.argv[1])
    print(out)
