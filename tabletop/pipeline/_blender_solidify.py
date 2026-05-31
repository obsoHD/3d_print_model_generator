"""_blender_solidify - turn a thin-shell mesh into a printable solid by
voxel-remeshing at a coarse cell size. Subprocess from mesh_gauntlet.

Strategy: voxel-remesh treats the input as an implicit occupancy field
and reconstructs a closed isosurface. Unlike Solidify-modifier (which
duplicates surfaces and tangles on dense MC output), voxel-remesh is
guaranteed watertight and manifold by construction.

Pass `--voxel-size 1.0` for a 1mm grid — fuses thin walls into proper
solid material while keeping ~1mm feature detail.

Invoked as:
    blender --background --python _blender_solidify.py -- \\
        --input mesh.glb --output mesh.solid.glb --voxel-size 1.0
"""
import argparse
import sys
import bpy


def _argv():
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     required=True)
    ap.add_argument("--output",    required=True)
    ap.add_argument("--voxel-size", type=float, default=1.0,
                    help="voxel-remesh cell size in mesh units (mm). "
                         "Smaller = more detail but may keep thin walls; "
                         "larger = sturdier solid but loses fine detail.")
    args = ap.parse_args(_argv())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Auto-detect format from extension — STL preserves mm units, GLB doesn't
    if args.input.lower().endswith(".stl"):
        try: bpy.ops.wm.stl_import(filepath=args.input)
        except Exception: bpy.ops.import_mesh.stl(filepath=args.input)
    else:
        bpy.ops.import_scene.gltf(filepath=args.input)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        print(f"[solidify] no mesh in {args.input}", flush=True)
        return 1
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    print(f"[solidify] loaded {len(obj.data.vertices)} verts, "
          f"{len(obj.data.polygons)} faces", flush=True)

    # Voxel remesh: implicit-field reconstruction. The cell size determines
    # the minimum feature size; anything thinner than `voxel_size` gets
    # fused into solid material — which is exactly what we want for
    # collapsing TripoSG/Pixal3D's parallel inner+outer shells into one
    # printable solid wall.
    mod = obj.modifiers.new(name="VoxelSolidify", type="REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = args.voxel_size
    mod.use_smooth_shade = False      # sharp — preserve corners
    mod.adaptivity = 0.0
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"[solidify] after voxel remesh ({args.voxel_size} mm): "
          f"{len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces",
          flush=True)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    if args.output.lower().endswith(".stl"):
        try:
            bpy.ops.wm.stl_export(filepath=args.output,
                                  export_selected_objects=True,
                                  ascii_format=False)
        except Exception:
            bpy.ops.export_mesh.stl(filepath=args.output, use_selection=True)
    else:
        try:
            bpy.ops.export_scene.gltf(filepath=args.output,
                                      use_selection=True,
                                      export_format="GLB")
        except Exception:
            bpy.ops.wm.gltf_export(filepath=args.output, use_selection=True,
                                   export_format="GLB")
    print(f"[solidify] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
