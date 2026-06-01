"""_trellis_infer - Microsoft TRELLIS.2 single-image-to-mesh.

Largest open image->3D model (4B, O-Voxel). Imports the upstream `trellis`
package from the cloned TRELLIS.2 repo via sys.path. Uses xformers attention
(not flash-attn) to dodge the Blackwell flash-attn build. Weights auto-download
from microsoft/TRELLIS.2-4B (or pass --model for a local dir).

NOTE: first-draft API modeled on TRELLIS-1 (run() -> outputs['mesh']); may need
a tweak once we see the real TRELLIS.2 pipeline — the streaming log will show it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _force_xformers_cutlass() -> None:
    """TRELLIS.2 SPARSE attention only supports xformers/flash_attn (no sdpa).
    We route it to xformers, but xformers' default dispatch picks the flash
    *Hopper* kernel (sm_90) which throws 'CUDA error: invalid argument' on
    Blackwell (sm_120). Force the CUTLASS op (broad arch coverage); fall back to
    auto-dispatch if CUTLASS rejects a specific mask/dtype."""
    try:
        import xformers.ops as xops  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return
    cutlass = getattr(xops, "MemoryEfficientAttentionCutlassOp", None)
    if cutlass is None or getattr(xops.memory_efficient_attention, "_cutlass_forced", False):
        return
    orig = xops.memory_efficient_attention

    def patched(*a, **k):
        if k.get("op") is None:
            k["op"] = cutlass
            try:
                return orig(*a, **k)
            except Exception:  # noqa: BLE001
                k.pop("op", None)
        return orig(*a, **k)

    patched._cutlass_forced = True  # noqa: SLF001
    xops.memory_efficient_attention = patched
    print("[trellis] forced xformers CUTLASS attention (Blackwell-safe)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--output", required=True, help="output GLB path")
    ap.add_argument("--lib",    required=True, help="TRELLIS.2 repo dir")
    ap.add_argument("--model",  default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--max-faces", type=int, default=800_000,
                    help="decimate above this (TRELLIS.2 emits ~4M faces which "
                         "hang the repair/solidify gauntlet)")
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    # DENSE attention -> pure-torch sdpa (no xformers/flash-attn needed). But
    # TRELLIS.2's SPARSE attention supports ONLY xformers/flash_attn (no sdpa),
    # so route THAT to xformers (installed) and force its CUTLASS op below to
    # avoid the sm_90-only flash-Hopper kernel that crashes on Blackwell.
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import numpy as np
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print(f"[trellis] torch {torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    print(f"[trellis] loading pipeline {args.model} ...", flush=True)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipe.cuda()

    img = Image.open(args.image).convert("RGBA")
    _force_xformers_cutlass()   # Blackwell-safe sparse attention
    print(f"[trellis] running ({args.image}, seed={args.seed})...", flush=True)
    m = pipe.run(img)[0]   # O-Voxel mesh result

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    mesh = trimesh.Trimesh(vertices=_np(m.vertices), faces=_np(m.faces))
    print(f"[trellis] raw mesh: {len(mesh.faces)} faces", flush=True)

    # TRELLIS.2 emits ~4M faces — the repair/orient/solidify gauntlet effectively
    # hangs on meshes that dense. Decimate to a printable target (quality is
    # unaffected at mini scale; this is what TRELLIS.2's own to_glb does on export).
    if len(mesh.faces) > args.max_faces:
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype("float64"),
                                       face_matrix=mesh.faces.astype("int32")))
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=int(args.max_faces), preservenormal=True,
                preservetopology=False, planarquadric=True)
            cm = ms.current_mesh()
            mesh = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                                   faces=cm.face_matrix())
            print(f"[trellis] decimated -> {len(mesh.faces)} faces "
                  f"(target {args.max_faces})", flush=True)
        except Exception as e:  # noqa: BLE001 — never block on decimation
            print(f"[trellis] WARN decimation failed ({e}); exporting raw mesh",
                  flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[trellis] saved {out.name} ({out.stat().st_size/1e6:.1f} MB, "
          f"faces={len(mesh.faces)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
