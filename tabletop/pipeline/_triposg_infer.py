"""_triposg_infer - CLI subprocess that runs TripoSG (VAST-AI/TripoSG).

Lives in the pixal3d_venv (torch 2.8.0+cu129). Imports the upstream TripoSG
package via sys.path. Uses use_flash_decoder=False so the hierarchical
skimage-marching-cubes path is used (avoids the diso CUDA build dep, which
needs MSVC on Windows). TripoSG's SDF representation guarantees watertight
output regardless of which extractor is used.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",       required=True)
    ap.add_argument("--output",      required=True, help="output GLB path")
    ap.add_argument("--lib",         required=True, help="TripoSG repo dir")
    ap.add_argument("--weights",     required=True, help="dir holding TripoSG + RMBG weights")
    ap.add_argument("--steps",       type=int,   default=50)
    ap.add_argument("--guidance",    type=float, default=7.0)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--faces",       type=int,   default=300_000,
                    help="decimate to N faces; -1 keeps native resolution")
    args = ap.parse_args()

    sys.path.insert(0, args.lib)
    sys.path.insert(0, str(Path(args.lib) / "scripts"))

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from image_process import prepare_image
    from briarmbg import BriaRMBG

    device = "cuda"
    dtype = torch.float16

    weights_root = Path(args.weights); weights_root.mkdir(parents=True, exist_ok=True)
    triposg_dir = weights_root / "TripoSG"
    rmbg_dir    = weights_root / "RMBG-1.4"
    if not (triposg_dir / "model_index.json").exists():
        print(f"[triposg] downloading VAST-AI/TripoSG -> {triposg_dir}", flush=True)
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=str(triposg_dir))
    if not (rmbg_dir / "config.json").exists() and not any(rmbg_dir.glob("*.bin")):
        print(f"[triposg] downloading briaai/RMBG-1.4 -> {rmbg_dir}", flush=True)
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=str(rmbg_dir))

    print(f"[triposg] loading RMBG-1.4", flush=True)
    rmbg_net = BriaRMBG.from_pretrained(str(rmbg_dir)).to(device); rmbg_net.eval()

    print(f"[triposg] loading TripoSG pipeline", flush=True)
    pipe = TripoSGPipeline.from_pretrained(str(triposg_dir)).to(device, dtype)

    print(f"[triposg] preprocessing image: {args.image}", flush=True)
    img = prepare_image(args.image, bg_color=np.array([1.0, 1.0, 1.0]),
                        rmbg_net=rmbg_net)

    print(f"[triposg] sampling ({args.steps} steps, guidance={args.guidance}, "
          f"seed={args.seed})", flush=True)
    out = pipe(
        image=img,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_flash_decoder=True,     # patched flash path uses skimage MC if diso absent
    ).samples[0]
    import trimesh
    mesh = trimesh.Trimesh(out[0].astype(np.float32),
                           np.ascontiguousarray(out[1]))

    # TripoSG's DMC output has a clean watertight main hull + a few hundred
    # tiny floating artifacts (typically <0.1% of total faces). Drop them so
    # the slicer doesn't choke on phantom geometry. The main piece is
    # genuinely watertight by construction.
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        main = max(parts, key=lambda p: len(p.faces))
        dropped = sum(len(p.faces) for p in parts if p is not main)
        print(f"[triposg] kept largest of {len(parts)} pieces "
              f"({len(main.faces)} faces, dropped {dropped} from "
              f"{len(parts)-1} floaters); watertight={main.is_watertight}",
              flush=True)
        mesh = main

    if args.faces > 0 and len(mesh.faces) > args.faces:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=args.faces)
        m = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
        print(f"[triposg] decimated to {len(mesh.faces)} faces", flush=True)

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"[triposg] saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB, "
          f"verts={len(mesh.vertices)}, faces={len(mesh.faces)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
