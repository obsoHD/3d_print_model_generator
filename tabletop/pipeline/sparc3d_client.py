"""sparc3d_client - run Sparc3D locally to produce a printable mesh.

Sparc3D lives in its own venv at tabletop/sparc3d/venv/. We invoke its
inference script via subprocess so we don't have to install its (large,
strict) dependency tree into the orchestrator's interpreter.

The exact CLI invocation depends on the upstream repo's layout; this
client assumes a standard `python -m sparc3d.run_image2mesh` interface
with `--image` and `--output` flags. If the upstream changes, override
SPARC3D_CLI_TEMPLATE via env var.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLETOP_ROOT = HERE.parent
SPARC_DIR = TABLETOP_ROOT / "sparc3d"
SPARC_WEIGHTS = TABLETOP_ROOT / "models" / "sparc3d"

# Default invocation. The upstream repo's actual entrypoint may differ;
# override SPARC3D_CLI to point at the right script.
# One shared interpreter on the Linux workstation (override with GEN3D_PY).
DEFAULT_CLI = (
    os.environ.get("GEN3D_PY") or sys.executable,
    "-m", "sparc3d.run_image2mesh",
    "--image",  "{image}",
    "--output", "{output}",
    "--weights", str(SPARC_WEIGHTS),
    "--resolution", "1024",
)


def generate(image_path: str, out_path: str,
             target_scale_mm: float = 32.0,
             resolution: int = 1024) -> str:
    """Sparc3D image -> mesh GLB. Returns out_path."""
    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(f"input image not found: {img}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cli_template = os.environ.get("SPARC3D_CLI")
    if cli_template:
        # User supplied an override, expected to contain {image} {output} {resolution}
        cmd = cli_template.format(image=img, output=out_path,
                                   resolution=resolution).split()
    else:
        cmd = [
            str(DEFAULT_CLI[0]),
            *DEFAULT_CLI[1:4],
            str(img),                # {image}
            DEFAULT_CLI[5],
            str(out_path),           # {output}
            DEFAULT_CLI[7],
            str(SPARC_WEIGHTS),
            DEFAULT_CLI[9],
            str(resolution),
        ]

    # Sparc3D's exact CLI may differ from our default. If the run fails
    # because of unknown args, the user can set SPARC3D_CLI in env.
    print(f"  [sparc3d] {' '.join(cmd[:3])} ...")
    proc = subprocess.run(cmd, cwd=str(SPARC_DIR), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Sparc3D failed (exit {proc.returncode}).\n"
            f"stderr tail:\n{proc.stderr[-1500:]}\n"
            "If the CLI shape is wrong for your Sparc3D version, set the\n"
            "SPARC3D_CLI environment variable to the correct command with\n"
            "{image} and {output} placeholders."
        )

    if not Path(out_path).exists():
        raise RuntimeError(
            f"Sparc3D returned 0 but no output produced at {out_path}.\n"
            f"stdout tail:\n{proc.stdout[-1500:]}"
        )

    # Sparc3D's outputs are already watertight/manifold by design — no
    # post-processing here, that's Blender's job.
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python sparc3d_client.py <image.png> <out.glb>", file=sys.stderr)
        sys.exit(2)
    out = generate(image_path=sys.argv[1], out_path=sys.argv[2])
    print(out)
