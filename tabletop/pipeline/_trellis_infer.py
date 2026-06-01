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


def _trellis_geometry_mesh(m, max_faces, remesh=True,
                           remesh_band=1.0, remesh_project=0.9, verbose=True):
    """Run ONLY the geometry half of o_voxel.postprocess.to_glb — the cumesh
    remesh + clean steps that build the clean watertight mesh — and read out
    verts/faces, SKIPPING the slow CPU xatlas UV unwrap + nvdiffrast texture
    bake (we print geometry, not color). Mirrors the upstream to_glb body so the
    coordinate convention matches the textured path (Y-up glTF). ~seconds vs the
    multi-minute xatlas tail."""
    import cumesh
    import numpy as np
    import torch
    import trimesh

    dev = m.coords.device
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        dtype=torch.float32, device=dev)
    vs = m.voxel_size
    if isinstance(vs, float):
        vs = [vs, vs, vs]
    if not torch.is_tensor(vs):
        vs = torch.tensor(np.array(vs), dtype=torch.float32, device=dev)
    grid_size = ((aabb[1] - aabb[0]) / vs.to(dev).float()).round().int()

    mesh = cumesh.CuMesh()
    mesh.init(m.vertices.cuda(), m.faces.cuda())
    mesh.fill_holes(max_hole_perimeter=3e-2)
    v, f = mesh.read()
    bvh = cumesh.cuBVH(v, f)

    if remesh:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            v, f, center=center,
            scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution, band=remesh_band,
            project_back=remesh_project, verbose=verbose, bvh=bvh))
        mesh.simplify(int(max_faces), verbose=verbose)
        # Upstream's remesh branch STOPS here, leaving non-manifold edges + holes
        # (fine for texturing, NOT for printing). Add the same cumesh cleaning
        # the non-remesh branch uses — all GPU, fast — so the mesh is manifold +
        # watertight without the slow CPU pymeshfix. fill_holes at 1.0 closes the
        # remaining boundaries (extent ~1 unit) so the slicer accepts it.
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=1.0)
        mesh.repair_non_manifold_edges()
        mesh.unify_face_orientations()
    else:
        mesh.simplify(int(max_faces) * 3, verbose=verbose)
        mesh.remove_duplicate_faces(); mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(int(max_faces), verbose=verbose)
        mesh.remove_duplicate_faces(); mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()

    ov, of = mesh.read()
    ov = ov.cpu().numpy().astype("float64")
    of = of.cpu().numpy()
    # Same coord convention as to_glb (swap Y/Z, invert Y -> glTF Y-up), done
    # safely (no in-place numpy view aliasing).
    conv = ov.copy()
    conv[:, 1] = ov[:, 2]
    conv[:, 2] = -ov[:, 1]
    return trimesh.Trimesh(vertices=conv, faces=of, process=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--output", required=True, help="output GLB path")
    ap.add_argument("--lib",    required=True, help="TRELLIS.2 repo dir")
    ap.add_argument("--model",  default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--max-faces", type=int,
                    default=int(os.environ.get("TRELLIS_MAX_FACES", 1_500_000)),
                    help="decimate above this (raw is ~4M; finish is fast now, "
                         "so we keep more faces = more surface detail)")
    # QUALITY knobs — these use your VRAM headroom (the 4B model itself only
    # needs ~4GB; the rest of your 32GB buys detail, not speed):
    #   max_num_tokens — TRELLIS.2's primary quality dial (default 49152 ≈ 4GB).
    #     Doubling to ~98k ≈ 8-12GB gives a denser/finer O-Voxel structure.
    #   steps — diffusion sampling steps per stage (was 12; 25 = cleaner shape).
    ap.add_argument("--max-tokens", type=int,
                    default=int(os.environ.get("TRELLIS_TOKENS", 131_072)))
    ap.add_argument("--steps", type=int,
                    default=int(os.environ.get("TRELLIS_STEPS", 30)))
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
    print(f"[trellis] running (seed={args.seed}, tokens={args.max_tokens}, "
          f"steps={args.steps})...", flush=True)
    # preprocess_image=False: we already removed the background upstream
    # (concept_gen rembg), so skip TRELLIS.2's internal RMBG re-crop.
    try:
        m = pipe.run(
            img,
            seed=args.seed,
            preprocess_image=False,
            max_num_tokens=int(args.max_tokens),
            sparse_structure_sampler_params={"steps": int(args.steps)},
            shape_slat_sampler_params={"steps": int(args.steps)},
            tex_slat_sampler_params={"steps": int(args.steps)},
        )[0]
    except TypeError as e:
        # Param-name drift across TRELLIS.2 versions — fall back to a plain run
        # so we still get a mesh (OOM is NOT caught here; lower TRELLIS_TOKENS).
        print(f"[trellis] WARN quality params rejected ({e}); plain run", flush=True)
        m = pipe.run(img, seed=args.seed, preprocess_image=False)[0]

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    # === Build the printable mesh ===
    # Both paths use TRELLIS.2's OFFICIAL o_voxel remesh+clean (cumesh) — that's
    # what turns the raw dual-grid voxel output into a CLEAN watertight mesh.
    # The difference is the texture half (xatlas UV unwrap + nvdiffrast bake),
    # which is a slow CPU step we DON'T need for printing:
    #   TRELLIS_TEXTURE=0 (default) -> geometry only, ~seconds (printing)
    #   TRELLIS_TEXTURE=1           -> full to_glb with baked texture (slow; the
    #                                  colored GLB for viewing, not printing)
    want_texture = os.environ.get("TRELLIS_TEXTURE", "0") == "1"
    mesh = None
    if not want_texture:
        try:
            print(f"[trellis] o_voxel GEOMETRY-only remesh (skip xatlas/texture; "
                  f"target {args.max_faces})...", flush=True)
            mesh = _trellis_geometry_mesh(m, int(args.max_faces), remesh=True,
                                          verbose=True)
            print(f"[trellis] geometry remesh OK (no texture): {len(mesh.faces)} "
                  f"faces, watertight={mesh.is_watertight}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[trellis] WARN geometry-only path failed ({e}); "
                  f"trying full to_glb", flush=True)
            mesh = None
    if mesh is None:
        try:
            import o_voxel
            print(f"[trellis] o_voxel.to_glb (with texture; target "
                  f"{args.max_faces})...", flush=True)
            glb = o_voxel.postprocess.to_glb(
                vertices=m.vertices, faces=m.faces,
                attr_volume=m.attrs, coords=m.coords, attr_layout=m.layout,
                voxel_size=m.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(args.max_faces),
                texture_size=1024, remesh=True, remesh_band=1.0, remesh_project=0.9,
                verbose=True)
            mesh = glb if isinstance(glb, trimesh.Trimesh) else glb.dump(concatenate=True)
            print(f"[trellis] to_glb OK: {len(mesh.faces)} faces, "
                  f"watertight={mesh.is_watertight}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[trellis] WARN o_voxel to_glb unavailable ({e}); "
                  f"raw + pymeshlab decimate fallback", flush=True)
            mesh = None

    if mesh is None:
        mesh = trimesh.Trimesh(vertices=_np(m.vertices), faces=_np(m.faces))
        if len(mesh.faces) > args.max_faces:
            try:
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices.astype("float64"),
                    face_matrix=mesh.faces.astype("int32")))
                for _ps in (
                        dict(targetfacenum=int(args.max_faces), preservenormal=True,
                             preservetopology=True, planarquadric=True,
                             qualitythr=0.35, optimalplacement=True),
                        dict(targetfacenum=int(args.max_faces), preservenormal=True,
                             preservetopology=True),
                        dict(targetfacenum=int(args.max_faces))):
                    try:
                        ms.meshing_decimation_quadric_edge_collapse(**_ps)
                        break
                    except Exception:  # noqa: BLE001
                        continue
                for filt in ("meshing_remove_null_faces",
                             "meshing_remove_duplicate_faces",
                             "meshing_remove_duplicate_vertices",
                             "meshing_remove_unreferenced_vertices"):
                    try:
                        getattr(ms, filt)()
                    except Exception:  # noqa: BLE001
                        pass
                cm = ms.current_mesh()
                mesh = trimesh.Trimesh(vertices=cm.vertex_matrix(),
                                       faces=cm.face_matrix())
                print(f"[trellis] decimated -> {len(mesh.faces)} faces", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[trellis] WARN decimation failed ({e}); raw mesh", flush=True)

    # Manifold cleanup (fast trimesh ops): cumesh emits doubled vertices at
    # seams which read as non-manifold edges. Merge coincident verts + drop
    # degenerate/duplicate faces + fix winding/normals — resolves most NM edges
    # without the slow CPU pymeshfix.
    try:
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception as e:  # noqa: BLE001
        print(f"[trellis] WARN manifold cleanup partial ({e})", flush=True)

    # Orientation: the o_voxel coord swap leaves the model glTF Y-up; our gauntlet
    # + slicer are Z-up. Rotate +90 deg about X so +Y (up) -> +Z (up) = upright.
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0]))
    print(f"[trellis] final mesh: {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight} (Z-up upright)", flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[trellis] saved {out.name} ({out.stat().st_size/1e6:.1f} MB, "
          f"faces={len(mesh.faces)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
