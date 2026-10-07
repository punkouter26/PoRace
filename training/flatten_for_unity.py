"""Emit Assets/MuJoCo/go2_unity.xml: scene_porace.xml with includes resolved, for the Unity importer
(which cannot read <include>, <keyframe> or <contact> sensors). Meshes are copied next to it.
Run: uv run --with mujoco python flatten_for_unity.py
"""
import shutil
import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).parent))
from go2 import constants as C  # noqa: E402

out_dir = C.ROOT_PATH.parent / "Assets" / "MuJoCo"
out_dir.mkdir(parents=True, exist_ok=True)
spec = mujoco.MjSpec.from_file(str(C.SCENE_XML))
spec.compile()
xml = spec.to_xml()
# MjSpec omits MuJoCo defaults; make the contract values explicit for the Unity importer.
xml = xml.replace("<option ", '<option integrator="Euler" cone="pyramidal" ', 1)
assert '<flag eulerdamp="disable"/>' in xml
(out_dir / "go2_unity.xml").write_text(xml)
assets_out = out_dir / "assets"
assets_out.mkdir(exist_ok=True)
for obj in (C.ASSET_PATH / "assets").glob("*.obj"):
    shutil.copy(obj, assets_out / obj.name)
# Round-trip check: the flat file must compile to the same sizes and options as the scene.
flat = mujoco.MjModel.from_xml_path(str(out_dir / "go2_unity.xml"))
src = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
assert (flat.nq, flat.nv, flat.nu, flat.ngeom, flat.nsensor) == (src.nq, src.nv, src.nu, src.ngeom, src.nsensor)
assert flat.opt.timestep == src.opt.timestep and flat.opt.integrator == src.opt.integrator
assert abs(flat.body_mass.sum() - src.body_mass.sum()) < 1e-9
print(f"OK wrote {out_dir / 'go2_unity.xml'} ({len(xml)} chars), {len(list(assets_out.glob('*.obj')))} meshes")
