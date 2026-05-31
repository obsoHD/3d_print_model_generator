"""_render_preview - headless PNG render of an STL via pyvista (VTK offscreen)."""
import argparse, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--res", type=int, default=700)
    ap.add_argument("--azimuth", type=float, default=30)
    ap.add_argument("--elevation", type=float, default=-10)
    args = ap.parse_args()
    import pyvista as pv
    pv.OFF_SCREEN = True
    mesh = pv.read(args.input)
    pl = pv.Plotter(off_screen=True, window_size=(args.res, args.res))
    pl.add_mesh(mesh, color="#9fb4c4", smooth_shading=False,
                show_edges=False, specular=0.3)
    pl.add_axes()
    pl.camera_position = "iso"
    pl.camera.azimuth = args.azimuth
    pl.camera.elevation = args.elevation
    pl.set_background("#1a1f28")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(args.output)
    print(f"wrote {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
