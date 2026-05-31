# Hunyuan3D — Two Generation Paths

The pipeline offers **two image→3D engines**, both from Tencent's Hunyuan3D
family. They share the same VAE/decoder family and the same downstream
finishing (sharp pymeshfix finish → statue scale → slicer validation), and
differ only in how many views of the subject they consume.

| | **Single-view** | **Multi-view** |
|---|---|---|
| Model | `Hunyuan3D-2.1` (shape DiT) | `Hunyuan3D-2mv` (multi-view DiT) |
| HF repo | local `Hunyuan3D-2.1/` clone | `tencent/Hunyuan3D-2mv`, subfolder `hunyuan3d-dit-v2-mv` |
| VAE | Hunyuan3D-2.1 VAE | `tencent/Hunyuan3D-2` `hunyuan3d-vae-v2-0` |
| Input | **1 image** (ideally the clean grey render) | **3 views**: front + left + back |
| Engine name | `hunyuan21` | `hunyuan2mv` |
| Best for | quick gen, front-facing display pieces | accurate in-the-round minis, correct backs |
| Octree | 384 (high-detail default) | 380 |
| Steps | 75 | 30 (mv model is finetuned for fewer) |

---

## When to use which

**Single-view (`hunyuan21`)** — give it ONE image and it generates a full 3D
model, *inferring* everything it can't see. The front is faithful; the back
and occluded regions (behind a wall, the far side of a coat) are the model's
best guess — they come out plausible but smooth/blank. Perfect when:
- You only have one reference image (a concept, a photo, a grey render).
- The piece is viewed mainly from the front (diorama, display base).
- You want speed.

**Multi-view (`hunyuan2mv`)** — give it the **front, left, and back** and it
fuses them into one mesh, so the occluded sides are *real data* instead of a
guess. The back is accurate, the silhouette is correct from every angle. Use
when:
- You have a turntable/spin GIF or multiple stills of the same sculpt.
- You need the model correct in the round (tabletop mini seen from all sides).
- Consistency matters more than convenience.

> **The three views must be the same subject, consistent lighting, clean
> background.** A painted-mini turntable GIF is ideal input. A few handheld
> phone angles work if the lighting is even and the poses match.

---

## Input formats

### Single-view
- One image, square-ish, subject centered, plain/dark background.
- **The clean grey resin render is the best input** — no paint, no scene
  clutter, unambiguous geometry. A painted mini also works; a concept image
  generated from a prompt works.

### Multi-view
- **Three stills**: `front`, `left`, `back` (the exact keys the model wants).
- **OR a spinning GIF / turntable video** → we extract evenly-spaced frames
  and map them to front/left/back (e.g. frames at 0°, 90°, 180°).
- Each view: same sculpt, same scale, clean background. The UI asks for each
  specific view in its own box ("front goes here", "left goes here", ...).

---

## Models we actually use

| Model | Role | Why |
|---|---|---|
| **Z-Image-Turbo** (Tongyi-MAI) | text→concept image | fast, high-quality concept at 1280px when prompting first |
| **FLUX.1-Kontext-dev** | view-fix / identity edits | repair broken side views (legacy MV path) |
| **Hunyuan3D-2.1** | single-view shape | best open single-image mesh; clean topology |
| **Hunyuan3D-2mv** | multi-view shape | accurate in-the-round from front/left/back |
| **build123d** (parametric CAD) | structural terrain | watertight by construction (towers, walls, tiles, crates, barrels) |
| **pymeshfix / Tweaker-3 / manifold3d** | finishing | sharp topology repair (no resampling), orient, base |
| **Bambu Studio + PrusaSlicer** | validation oracle | real-slicer PASS/FAIL gate |

---

## Models installed but NOT used (and why)

These are on disk from earlier experiments. Kept for reference / fallback, but
**not** in the active path:

| Model | Status | Why not used |
|---|---|---|
| **Pixal3D** (TencentARC) | installed, working | Beautiful PBR-textured GLB, but its O-Voxel surface extraction produces thin-shell, multi-nested-shell meshes that are unprintable even after heavy repair (~0.3–7% solid). Great for renders, bad for printing. Kept for textured preview only. |
| **TripoSG** (VAST-AI) | installed, working | SDF + DiffDMC, watertight-ish, but softer detail than Hunyuan and needs the same gauntlet. Hunyuan3D 2.1 produces cleaner topology (12 splits vs 312) and better aspect, so it won. Kept as a fast alternative. |
| **TRELLIS.2** | downloaded | Same SDF/MC family; "raw meshes may have small holes." 24 GB VRAM min — tight on 16 GB. Superseded by Hunyuan for our use. |
| **Sparc3D** | researched, NOT installed | Claims watertight 1024³ but **no public code/weights released** — only a HF Space demo. Not installable. |
| **Direct3D-S2** | researched, NOT installed | Acknowledged duplicate-inner-shell bug (issue #55) + no Blackwell sm_120 support. Worse than what we have. |
| **Step1X-3D / Hi3DGen** | researched, NOT installed | Promising (TSDF / normal-bridged) but Linux/older-torch targeted, no Blackwell support merged; install is a 2–4h source battle for marginal gain over Hunyuan. |
| **Zero123++ / stable-zero123 / MV-Adapter** | installed | Multi-view *synthesizers* (one image → fake side views) for the legacy HY3D-mv path. Superseded: real multi-view input (Hunyuan3D-2mv) beats synthesized views. Kept as the single-image→multiview bridge if no real views exist. |
| **Sparc3D / sparc3d venv** | partial | Never finished install; Hunyuan covered the need. |

### Why Hunyuan won overall
- Single-image: cleanest topology of the open models, best aspect ratio,
  most production-mature (Tencent maintains it actively).
- Multi-view: a dedicated finetune (`-2mv`) that takes real views — the
  honest fix for the "blank inferred back" problem, instead of synthesizing
  fake views and hoping they're consistent.
- Both run on a 16 GB Blackwell RTX 5080 with CPU offload + expandable
  segments + FlashVDM off.

---

## Hardware / runtime notes
- RTX 5080, 16 GB VRAM, sm_120 Blackwell, Windows.
- venv: `pixal3d_venv` (py3.12, torch 2.8.0+cu129).
- FlashVDM is **disabled** — it corrupts CUDA state when combined with
  diffusers' model_cpu_offload ("unknown error" in the VAE decoder).
- octree 384 single-view ≈ 6–9 min; mv ≈ similar.
- Output is **statue scale** (~160 mm) so fine grooves are physically deep;
  downsize in the slicer for a tabletop mini.
