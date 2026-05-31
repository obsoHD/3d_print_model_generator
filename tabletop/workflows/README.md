# workflows/  —  ComfyUI workflow JSONs

The pipeline drives ComfyUI by POSTing a workflow JSON to its HTTP API.
Each file here is a workflow with three placeholders that `comfy_client.py`
substitutes at runtime:

| Token | Replaced with |
|---|---|
| `$PROMPT$` | the user's prompt (kind-specific styling pre-/suffixed by Claude) |
| `$NEG$` | a fixed negative-prompt string |
| `$SEED$` | the seed integer |

## Provided workflows

| File | For | Notes |
|---|---|---|
| `flux_mini_concept.json` | minis | FLUX-dev, clean isolated character on plain background, T-pose-ish, even lighting — ideal Sparc3D input. |
| `flux_terrain_concept.json` | terrain | FLUX-dev, orthographic-ish hero shot of architecture/scatter, plain background. |

You can drop in more workflows (e.g. `sdxl_dnd_lora_mini.json` if you
prefer SDXL + a D&D LoRA). Just keep the three placeholder tokens.

## Editing in ComfyUI's GUI

1. Open ComfyUI in a browser (`http://127.0.0.1:8188`)
2. Load the JSON: `Workflow → Load → pick the file from this directory`
3. Edit nodes, reroute, swap models, tweak sampler steps
4. `Workflow → Save (API format)` — saves a workflow that the HTTP API
   accepts. Replace the file here.

**Don't save in "developer" format** — the API format is what the HTTP
endpoint expects. The two look almost identical; the API format omits the
GUI-only metadata.

## Building a new workflow from scratch

1. Open ComfyUI fresh.
2. Build the graph that produces your image (Load Checkpoint, CLIP Text
   Encode for positive, another for negative, KSampler, VAE Decode, Save
   Image).
3. Replace your prompt text with the literal `$PROMPT$`, your negative
   with `$NEG$`, and your seed with the string `$SEED$` (in the seed
   node's `inputs.seed` field).
4. Save in API format.
5. Reference it by name in `pipeline/orchestrate.py`'s `KIND_DEFAULTS`.
