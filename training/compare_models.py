"""Diff the Unity-regenerated model against the trainer model, element by element (by name). B.3."""
import sys
from pathlib import Path
import mujoco, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from go2 import constants as C
A = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
B = mujoco.MjModel.from_xml_path(sys.argv[1] if len(sys.argv) > 1 else str(C.ROOT_PATH.parent / "docs/parity/unity_generated.xml"))
OBJ = mujoco.mjtObj
def names(m, t, n): return [mujoco.mj_id2name(m, t, i) for i in range(n)]
def qnorm(q): q=np.asarray(q,float); return q if q[0] >= 0 else -q
def cmp(label, va, vb, tol=1e-6):
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    if va.shape != vb.shape or np.abs(va - vb).max() > tol: print(f"DIFF {label}: train={va.tolist()} unity={vb.tolist()}")
for f in ["timestep","impratio","iterations","ls_iterations","integrator","cone","jacobian","solver","tolerance","ls_tolerance","noslip_iterations","noslip_tolerance","ccd_iterations","ccd_tolerance","disableflags","enableflags","density","viscosity","o_margin"]:
    cmp("opt."+f, getattr(A.opt,f), getattr(B.opt,f))
cmp("opt.gravity", A.opt.gravity, B.opt.gravity); cmp("opt.o_solref", A.opt.o_solref, B.opt.o_solref); cmp("opt.o_solimp", A.opt.o_solimp, B.opt.o_solimp)
print("sizes", (A.nq,A.nv,A.nu,A.nbody,A.ngeom,A.njnt,A.nsensor), (B.nq,B.nv,B.nu,B.nbody,B.ngeom,B.njnt,B.nsensor))
# bodies
for n in names(A, OBJ.mjOBJ_BODY, A.nbody)[1:]:
    a, b = mujoco.mj_name2id(A, OBJ.mjOBJ_BODY, n), mujoco.mj_name2id(B, OBJ.mjOBJ_BODY, n)
    if b < 0: print("MISSING body", n); continue
    for f in ["body_mass","body_inertia","body_ipos","body_iquat","body_pos","body_quat","body_gravcomp"]:
        cmp(f"{f}[{n}]", qnorm(getattr(A,f)[a]) if "quat" in f else getattr(A,f)[a], qnorm(getattr(B,f)[b]) if "quat" in f else getattr(B,f)[b])
# joints
for n in names(A, OBJ.mjOBJ_JOINT, A.njnt):
    if n is None: continue
    a, b = mujoco.mj_name2id(A, OBJ.mjOBJ_JOINT, n), mujoco.mj_name2id(B, OBJ.mjOBJ_JOINT, n)
    if b < 0: print("MISSING joint", n); continue
    for f in ["jnt_type","jnt_range","jnt_axis","jnt_pos","jnt_stiffness","jnt_solref","jnt_solimp","jnt_margin","jnt_limited","jnt_actfrclimited","jnt_actfrcrange"]:
        cmp(f"{f}[{n}]", getattr(A,f)[a], getattr(B,f)[b])
    da, db = A.jnt_dofadr[a], B.jnt_dofadr[b]
    for f in ["dof_damping","dof_armature","dof_frictionloss","dof_solref","dof_solimp"]:
        cmp(f"{f}[{n}]", getattr(A,f)[da], getattr(B,f)[db])
# geoms: match by body + type + nearest pos (most geoms are unnamed; quaternion sign flips are equivalent)
def qnorm(q): q=np.asarray(q,float); return q if q[0] >= 0 else -q
for g in range(A.ngeom):
    bn, t = mujoco.mj_id2name(A, OBJ.mjOBJ_BODY, A.geom_bodyid[g]), A.geom_type[g]
    cands = [h for h in range(B.ngeom) if mujoco.mj_id2name(B, OBJ.mjOBJ_BODY, B.geom_bodyid[h]) == bn and B.geom_type[h] == t]
    if not cands: print("MISSING geom", bn, t, A.geom_size[g]); continue
    h = min(cands, key=lambda h: np.linalg.norm(A.geom_pos[g] - B.geom_pos[h]))
    k = f"{bn}:type{t}:{np.round(A.geom_size[g],4).tolist()}"
    cmp(f"geom_pos[{k}]", A.geom_pos[g], B.geom_pos[h], 1e-5); cmp(f"geom_size[{k}]", A.geom_size[g], B.geom_size[h], 1e-5)
    cmp(f"geom_quat[{k}]", qnorm(A.geom_quat[g]), qnorm(B.geom_quat[h]), 1e-5)
    for f in ["geom_contype","geom_conaffinity","geom_condim","geom_priority","geom_friction","geom_solref","geom_solimp","geom_margin","geom_gap","geom_solmix","geom_group","geom_rbound"]:
        cmp(f"{f}[{k}]", getattr(A,f)[g], getattr(B,f)[h], 1e-5)
# actuators
for n in C.ACTUATOR_NAMES:
    a, b = mujoco.mj_name2id(A, OBJ.mjOBJ_ACTUATOR, n), mujoco.mj_name2id(B, OBJ.mjOBJ_ACTUATOR, n)
    for f in ["actuator_gainprm","actuator_biasprm","actuator_ctrlrange","actuator_forcerange","actuator_dyntype","actuator_gaintype","actuator_biastype","actuator_gear","actuator_ctrllimited","actuator_forcelimited","actuator_actlimited"]:
        cmp(f"{f}[{n}]", getattr(A,f)[a], getattr(B,f)[b])
# sites
for n in ["imu", *C.FEET_SITES]:
    a, b = mujoco.mj_name2id(A, OBJ.mjOBJ_SITE, n), mujoco.mj_name2id(B, OBJ.mjOBJ_SITE, n)
    cmp(f"site_pos[{n}]", A.site_pos[a], B.site_pos[b]); cmp(f"site_quat[{n}]", A.site_quat[a], B.site_quat[b])
print("done")
