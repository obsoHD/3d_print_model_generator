"""analyze_stl - one STL in -> a design brief (dimensions + key features) out.

The output is a markdown design brief with an embedded JSON block, optimised
for handing back to the parametric-CAD design loop (build123d generators).

Usage:
  python analyze_stl.py <file.stl> [--out brief.md] [--axis-hint forearm:X]

Notes:
  * Reads STL via trimesh (mesh-level analysis only - no BREP).
  * Computes bbox, volume, surface area, principal inertia axes (to suggest
    a natural "long axis"), and clusters mesh faces by normal direction to
    surface the dominant planar groups (= candidate walls / faces).
  * Does NOT try to be clever about labelling features ("this is a screen,
    this is a dial"); leave that to the human (or to Claude reading the brief
    next to a reference photo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh


def _cluster_planar_groups(mesh: trimesh.Trimesh, k_directions: int = 12, area_cutoff_pct: float = 1.0):
    """Bucket the mesh's facets into groups whose face normals are colinear.

    Returns the top groups by total area, with the axis-aligned direction
    when applicable, otherwise the mean normal vector.
    """
    normals = mesh.face_normals
    areas = mesh.area_faces
    total_area = float(areas.sum()) or 1.0

    # snap normals to nearest axis if within 5 degrees, otherwise keep them
    snapped = []
    axis_vecs = [
        ("+X", np.array([1, 0, 0])),
        ("-X", np.array([-1, 0, 0])),
        ("+Y", np.array([0, 1, 0])),
        ("-Y", np.array([0, -1, 0])),
        ("+Z", np.array([0, 0, 1])),
        ("-Z", np.array([0, 0, -1])),
    ]
    threshold = np.cos(np.deg2rad(5.0))
    for n in normals:
        snap = None
        for name, v in axis_vecs:
            if float(np.dot(n, v)) >= threshold:
                snap = name
                break
        snapped.append(snap)

    # accumulate area per axis-aligned snap
    by_axis: dict[str, float] = Counter()
    for s, a in zip(snapped, areas):
        if s is not None:
            by_axis[s] += float(a)

    groups = [
        {
            "direction": name,
            "area_mm2": round(area, 1),
            "fraction_pct": round(area / total_area * 100.0, 1),
        }
        for name, area in sorted(by_axis.items(), key=lambda kv: -kv[1])
        if area / total_area * 100.0 >= area_cutoff_pct
    ]
    return groups, total_area


def _detect_cylindrical_features(mesh: trimesh.Trimesh, min_area_mm2: float = 30.0):
    """Heuristic: find faces whose normals form a smooth ring around an axis
    -> candidate cylindrical surfaces (sockets, bores, knob shafts)."""
    candidates = []
    # cluster faces with sufficient area and grouped by adjacency that share
    # angularly-varying normals around a single axis
    try:
        # Use facets (groups of coplanar faces) - very fast, picks up flat
        # cylinder ENDS plus walls
        from trimesh.proximity import closest_point
        graph = mesh.face_adjacency
        # Trimesh provides convenient detected `facets` for coplanar groups
        # but for curved surfaces use `face_adjacency_angles` clustering.
        # Simpler proxy: report Trimesh's built-in `facets` count and area.
        facet_areas = [float(mesh.area_faces[f].sum()) for f in mesh.facets]
        # Find groups whose normals span > 45 deg in a single connected
        # region (rough cylinder marker).
        normals = mesh.face_normals
        for face_group, area in zip(mesh.facets, facet_areas):
            if area < min_area_mm2:
                continue
            ns = normals[face_group]
            # if range of normals > 0.5 (i.e. surface curves significantly),
            # it's probably curved
            spread = float(ns.std(axis=0).sum())
            if spread > 0.3:
                # estimate axis as the normal direction with the LEAST variance
                u, s, vt = np.linalg.svd(ns - ns.mean(axis=0), full_matrices=False)
                axis_vec = vt[-1] / max(np.linalg.norm(vt[-1]), 1e-9)
                center = mesh.triangles_center[face_group].mean(axis=0)
                candidates.append({
                    "area_mm2": round(area, 1),
                    "axis_vector": [round(float(v), 3) for v in axis_vec],
                    "center_mm": [round(float(v), 3) for v in center],
                    "face_count": int(len(face_group)),
                })
    except Exception:
        pass
    candidates.sort(key=lambda c: -c["area_mm2"])
    return candidates[:12]


def analyze(stl_path: Path) -> dict:
    mesh: trimesh.Trimesh = trimesh.load_mesh(str(stl_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.dump()))

    bb_min, bb_max = mesh.bounds
    size = bb_max - bb_min
    extent_axis = "XYZ"[int(np.argmax(size))]
    aspect = size / float(size.max())

    watertight = bool(mesh.is_watertight)
    volume = float(mesh.volume) if watertight else None
    com = mesh.center_mass.tolist() if watertight else None

    # principal inertia (gives the "natural" long axis even when bbox is ambiguous)
    try:
        pia = mesh.principal_inertia_vectors  # 3x3 (rows are axes)
        principal_axes = pia.tolist()
    except Exception:
        principal_axes = None

    # Oriented Bounding Box (the natural box of the part regardless of world axes)
    try:
        obb = mesh.bounding_box_oriented
        obb_extents = obb.primitive.extents.tolist()
        obb_transform = obb.primitive.transform.tolist()
    except Exception:
        obb_extents = None
        obb_transform = None

    # Connected components (a quick way to know if the file is multiple parts)
    try:
        components = mesh.split(only_watertight=False)
        n_components = len(components)
    except Exception:
        n_components = 1

    groups, total_area = _cluster_planar_groups(mesh)
    cyl_features = _detect_cylindrical_features(mesh)

    return {
        "file": str(stl_path),
        "size_mm": [round(float(v), 3) for v in size],
        "bbox_min_mm": [round(float(v), 3) for v in bb_min],
        "bbox_max_mm": [round(float(v), 3) for v in bb_max],
        "obb_extents_mm": [round(float(v), 3) for v in obb_extents] if obb_extents else None,
        "obb_transform": obb_transform,
        "extent_axis": extent_axis,
        "aspect_normalised": [round(float(v), 3) for v in aspect],
        "face_count": int(len(mesh.faces)),
        "vertex_count": int(len(mesh.vertices)),
        "surface_area_mm2": round(float(total_area), 1),
        "is_watertight": watertight,
        "volume_mm3": round(volume, 1) if volume is not None else None,
        "center_of_mass_mm": [round(float(v), 3) for v in com] if com else None,
        "principal_inertia_axes": principal_axes,
        "dominant_planar_groups": groups,
        "n_connected_components": int(n_components),
        "curved_feature_candidates": cyl_features,
    }


def to_markdown(report: dict, axis_hints: dict | None = None) -> str:
    sz = report["size_mm"]
    bb_min = report["bbox_min_mm"]
    bb_max = report["bbox_max_mm"]

    lines = []
    lines.append(f"# Design brief — `{Path(report['file']).name}`")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Bounding box: **{sz[0]} x {sz[1]} x {sz[2]} mm**")
    lines.append(f"- Long axis: **{report['extent_axis']}** (aspect {report['aspect_normalised']})")
    lines.append(f"- Faces: {report['face_count']:,}  vertices: {report['vertex_count']:,}")
    lines.append(f"- Surface area: {report['surface_area_mm2']:,} mm²")
    if report["volume_mm3"] is not None:
        lines.append(f"- Volume: {report['volume_mm3']:,} mm³ (watertight ✓)")
    else:
        lines.append("- Volume: n/a (mesh is not watertight)")
    if report["center_of_mass_mm"]:
        com = report["center_of_mass_mm"]
        lines.append(f"- Center of mass: ({com[0]}, {com[1]}, {com[2]}) mm")
    lines.append("")

    lines.append("## Dominant planar groups (face-normal axis-aligned)")
    lines.append("These are large flat regions; each is a candidate 'wall'/'face' of the part.")
    lines.append("")
    lines.append("| Direction | Area (mm²) | % of total surface |")
    lines.append("|-----------|-----------:|-------------------:|")
    for g in report["dominant_planar_groups"]:
        lines.append(f"| {g['direction']} | {g['area_mm2']:,} | {g['fraction_pct']}% |")
    lines.append("")

    if axis_hints:
        lines.append("## Axis interpretation (user-provided hints)")
        for k, v in axis_hints.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if report.get("obb_extents_mm"):
        lines.append("## Oriented bounding box (natural axes of the part)")
        oe = report["obb_extents_mm"]
        lines.append(f"- Extents: **{oe[0]} x {oe[1]} x {oe[2]} mm** (sorted by the part's own principal axes)")
        lines.append("- If this differs from the axis-aligned bbox, the part is rotated relative to world axes.")
        lines.append("")

    if report.get("curved_feature_candidates"):
        lines.append("## Curved-feature candidates (likely sockets / bores / cylinder walls)")
        lines.append("Heuristically detected from face-normal clustering. Eyeball-check before trusting.")
        lines.append("")
        lines.append("| # | Area (mm²) | Axis ≈ | Center (X, Y, Z) mm |")
        lines.append("|---|-----------:|--------|--------------------:|")
        for i, c in enumerate(report["curved_feature_candidates"][:8], 1):
            ax = c["axis_vector"]
            ct = c["center_mm"]
            lines.append(f"| {i} | {c['area_mm2']} | ({ax[0]}, {ax[1]}, {ax[2]}) | ({ct[0]}, {ct[1]}, {ct[2]}) |")
        lines.append("")

    nc = report.get("n_connected_components", 1)
    if nc > 1:
        lines.append(f"⚠ This file contains **{nc} disconnected components** — likely an assembly, not a single part.")
        lines.append("")

    # --- build123d skeleton (paste-and-go starter) ---
    name = Path(report["file"]).stem.replace("-", "_").replace(" ", "_")
    sz_x, sz_y, sz_z = sz
    lines.append("## build123d generator skeleton")
    lines.append("Paste into your generator to start with the reference dimensions:")
    lines.append("")
    lines.append("```python")
    lines.append(f'"""Generator inspired by {Path(report["file"]).name}.')
    lines.append("")
    lines.append(f"Reference bbox: {sz_x} x {sz_y} x {sz_z} mm, long axis {report['extent_axis']}.")
    lines.append('"""')
    lines.append("")
    lines.append("from build123d import *")
    lines.append("")
    lines.append(f"# Reference target dimensions (from {Path(report['file']).name})")
    lines.append(f"REF_X = {sz_x}")
    lines.append(f"REF_Y = {sz_y}")
    lines.append(f"REF_Z = {sz_z}")
    lines.append("")
    lines.append("# Working scale - start equal to reference; reduce if too big")
    lines.append("SX, SY, SZ = REF_X, REF_Y, REF_Z")
    lines.append("")
    lines.append("def gen_step():")
    lines.append("    body = Pos(0, 0, SZ / 2) * Box(SX, SY, SZ)")
    lines.append("    # TODO: add real geometry — cuff cavity, screen well, sockets, etc.")
    lines.append(f'    body.label = "{name}_v1"')
    lines.append("    return body")
    lines.append("```")
    lines.append("")

    lines.append("## Raw JSON")
    lines.append("```json")
    lines.append(json.dumps(report, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stl", help="STL file to analyze")
    ap.add_argument("--out", default=None,
                    help="Output markdown path (default: prints to stdout)")
    ap.add_argument("--axis-hint", action="append", default=[],
                    help="key:value hints (e.g. forearm:X). Repeatable.")
    args = ap.parse_args(argv)

    stl_path = Path(args.stl)
    if not stl_path.exists():
        print(f"ERROR: file not found: {stl_path}", file=sys.stderr)
        return 2

    report = analyze(stl_path)

    hints = {}
    for h in args.axis_hint:
        if ":" in h:
            k, v = h.split(":", 1)
            hints[k.strip()] = v.strip()
    md = to_markdown(report, hints if hints else None)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"saved: {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
