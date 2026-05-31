# Models & LoRAs to download

The pipeline references several model weights. Download them into
`tabletop/models/` after the install scripts finish.

Use the HuggingFace CLI for everything — fastest, resumes downloads,
parallel chunks:

```bash
pip install huggingface_hub
huggingface-cli login   # only needed for gated models
```

## Tier 1 — required for any printable output

### Sparc3D weights  (PRIMARY printable engine)

```bash
huggingface-cli download ilcve21/Sparc3D \
    --local-dir tabletop/models/sparc3d
```

Size: ~4 GB. Required.

### FLUX.1-dev or SDXL  (concept image step)

FLUX-dev is the better quality model but is **gated** — accept the license
on the HuggingFace page first, then:

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev \
    --local-dir tabletop/models/flux \
    --include "flux1-dev.safetensors" \
              "ae.safetensors" \
              "text_encoder*/*" \
              "tokenizer*/*"
```

Size: ~24 GB. Required for terrain & high-quality minis.

Alternative if you'd rather start smaller: SDXL base 1.0

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    --local-dir tabletop/models/sdxl \
    --include "sd_xl_base_1.0.safetensors"
```

Size: ~6.5 GB.

## Tier 2 — strongly recommended

### D&D / character LoRAs

```bash
# General D&D style
huggingface-cli download weasley24/dnd-SDXL-LoRA \
    --local-dir tabletop/models/loras/dnd-sdxl

# Or browse civitai.com for character-pose, dynamic-pose, T-pose LoRAs
# and drop the .safetensors files into tabletop/models/loras/
```

### Terrain / architecture LoRAs

Pick from civitai.com — search "ruins", "fantasy architecture",
"wargames terrain". Save to `tabletop/models/loras/`.

## Tier 3 — optional (preview engine)

### Hunyuan3D 2.1 weights

```bash
huggingface-cli download tencent/Hunyuan3D-2.1 \
    --local-dir tabletop/models/hunyuan3d
```

Size: ~13 GB. Only needed if you want textured previews.

## Where files should end up

```
tabletop/models/
├── flux/
│   ├── flux1-dev.safetensors
│   ├── ae.safetensors
│   ├── text_encoder/...
│   └── tokenizer/...
├── sdxl/
│   └── sd_xl_base_1.0.safetensors
├── sparc3d/
│   ├── <checkpoint files>
│   └── config files
├── hunyuan3d/        (optional)
│   ├── shape/
│   └── texture/
└── loras/
    ├── dnd-sdxl/
    ├── miniature_style.safetensors
    └── ...
```

## Sanity check after downloads

```bash
python tabletop/pipeline/orchestrate.py --check-models
```

This will list which engines + LoRAs are detected and which are missing.
