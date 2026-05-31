# Example 1 — goblin warrior (mini, resin, 32 mm)

The canonical first run. Demonstrates the full pipeline end-to-end.

## 1. The user prompt

You say:
> "I need a goblin warrior with a curved dagger and a small shield, mid-stride, for a fantasy skirmish game. 32mm."

## 2. Claude → FLUX prompt (concept_director)

Claude calls `tabletop_full_pipeline` with:
```json
{
  "prompt": "goblin warrior with curved dagger and small wooden shield, mid-stride combat pose, leather armor with scrap-metal plates, full body, isolated on plain white background, even studio lighting, sharp focus, single subject, tabletop wargame miniature reference, 32mm scale",
  "kind": "mini",
  "printer": "resin",
  "scale_mm": 32
}
```

(The Claude-side concept_director prompt also adds negative-prompt extras
like "no background props, no multiple subjects, no blur".)

## 3. ComfyUI / FLUX-dev runs

`outputs/concepts/2026XXXX_HHMMSS_goblin_warrior_<hash>.png` appears.
Time on RTX 5080: ~12 s. Time on RTX 5090: ~6 s.

Expected output: a clean isolated goblin on white, full body, weapons
clearly visible.

## 4. Sparc3D produces the printable mesh

`outputs/meshes/2026XXXX_..._goblin_warrior_<hash>.glb`.
Time on RTX 5080: ~25 s.

Mesh stats you'll see in the run log:
```
triangles: ~120k
watertight: true (Sparc3D guarantees)
dims_mm: ~32 × 14 × 28
```

## 5. Blender cleanup → STL

`outputs/stl/2026XXXX_..._goblin_warrior_<hash>.stl`.
Time: ~3 s.

`<id>.cleanup.json` will say:
```json
{
  "ok": true,
  "non_manifold_before": 0,
  "non_manifold_after": 0,
  "triangles": 118432,
  "dims_mm": [32.0, 14.2, 28.1]
}
```

## 6. Slicer advisory

`<id>.print.json` looks like:
```json
{
  "watertight": true,
  "volume_cm3": 2.4,
  "suggested_orientation": {
    "method": "tilt-back-30-45",
    "tilt_deg": 35,
    "axis": "X"
  },
  "overhang_area_cm2_estimate": 1.8,
  "support_intensity_hint": "medium",
  "resin_volume_estimate_ml_solid": 2.4,
  "resin_volume_estimate_ml_hollowed": 0.7
}
```

Claude then writes a short advisory:

> Orient the goblin tilted ~35° back around X so the dagger and shield
> face upward. Auto-supports at medium density should land mostly under
> the base and behind the cape. Hollow at 2 mm wall with two 1.5 mm
> drainage holes on the underside of the base — saves ~70% resin (down
> to ~0.7 ml from 2.4). Sub-30° overhangs under the dagger pommel and
> the shield rim — bump those to manual support points.

## 7. You import into Lychee Slicer

Drag the STL in, apply the suggested orientation, accept auto-supports
with a manual pass for the dagger/shield, hollow + drains, slice.

## 8. Print result

Your turn. Add a photo here after the first successful print and update
the `printer_settings.md` next to this file with your printer brand /
profile / cure time that worked. That makes it the reference for future
minis.
