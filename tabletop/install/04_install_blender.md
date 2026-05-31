# Blender install (manual)

The pipeline calls Blender in headless mode to clean meshes for printing.
We don't auto-install Blender — install the app yourself.

## Windows

1. Download **Blender 4.2 LTS** (or newer) from
   <https://www.blender.org/download/lts/>.
2. Install. Default location is fine.
3. Add Blender to PATH so `blender --version` works in PowerShell:
   - Add `C:\Program Files\Blender Foundation\Blender 4.2` to your `PATH`
     environment variable.
4. Verify:
   ```powershell
   blender --version
   blender --background --python-expr "import bpy; print('ok')"
   ```

## Linux

```bash
# Easiest: snap install
sudo snap install blender --classic

# Or download Linux x64 from blender.org and put it on PATH:
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar xf blender-4.2.0-linux-x64.tar.xz
sudo mv blender-4.2.0-linux-x64 /opt/blender
sudo ln -sf /opt/blender/blender /usr/local/bin/blender
```

Verify:
```bash
blender --version
blender --background --python-expr "import bpy; print('ok')"
```

## Required addons

The cleanup script uses Blender's bundled **3D-Print Toolbox** addon (ships
with Blender 4.x, just needs to be enabled). The pipeline enables it
programmatically per-run — you don't need to enable it manually.
