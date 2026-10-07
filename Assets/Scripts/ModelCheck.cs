using System;
using Mujoco;
using UnityEngine;

namespace PoRace {

/// <summary>B.3 readback: the model Unity regenerated from its component tree must equal the trainer's.
/// Values come from docs/model_summary.json and training/go2/constants.py. Throws on mismatch.</summary>
public static unsafe class ModelCheck {
  const double Eps = 1e-6;

  public static void Assert(MujocoLib.mjModel_* m, Go2Model M) {
    Check(m->nq == 47 && m->nv == 42 && m->nu == 12, $"sizes nq={m->nq} nv={m->nv} nu={m->nu}, want 47/42/12");
    Check(Math.Abs(m->opt.timestep - 0.004) < 1e-9, $"timestep {m->opt.timestep} != 0.004 (set Time.fixedDeltaTime)");
    Check(m->opt.integrator == (int)MujocoLib.mjtIntegrator.mjINT_EULER, "integrator != Euler");
    Check(m->opt.cone == (int)MujocoLib.mjtCone.mjCONE_PYRAMIDAL, "cone != pyramidal");
    Check(Math.Abs(m->opt.impratio - 100) < Eps, $"impratio {m->opt.impratio}");
    Check(m->opt.iterations == 1 && m->opt.ls_iterations == 5, $"iterations {m->opt.iterations} ls {m->opt.ls_iterations}");
    Check((m->opt.disableflags & (int)MujocoLib.mjtDisableBit.mjDSBL_EULERDAMP) != 0, "eulerdamp not disabled");
    Check(Math.Abs(m->opt.gravity[2] + 9.81) < 1e-6 && Math.Abs(m->opt.gravity[0]) < Eps && Math.Abs(m->opt.gravity[1]) < Eps,
          $"gravity ({m->opt.gravity[0]},{m->opt.gravity[1]},{m->opt.gravity[2]})");

    double mass = 0;
    for (int b = 0; b < (int)m->nbody; b++) if (m->body_rootid[b] == M.BaseBody) mass += m->body_mass[b];
    Check(Math.Abs(mass - 15.206408) < 1e-4, $"robot mass {mass}");
    for (int k = 0; k < Go2Model.NumCubes; k++) Check(Math.Abs(m->body_mass[M.CubeBody[k]] - 1.0) < Eps, $"cube{k} mass");

    for (int i = 0; i < 12; i++) {
      int a = M.ActId[i];
      Check(Math.Abs(m->actuator_gainprm[a * 10] - 35) < Eps, $"gainprm[{i}]");
      Check(Math.Abs(m->actuator_biasprm[a * 10 + 1] + 35) < Eps && Math.Abs(m->actuator_biasprm[a * 10 + 2] + 0.5) < Eps, $"biasprm[{i}]");
      Check(Math.Abs(m->actuator_forcerange[a * 2] + 24) < Eps && Math.Abs(m->actuator_forcerange[a * 2 + 1] - 24) < Eps, $"forcerange[{i}]");
      int dof = M.JointDof[i];
      Check(Math.Abs(m->dof_damping[dof] - 0.5) < Eps && Math.Abs(m->dof_armature[dof] - 0.01) < Eps, $"dof damping/armature[{i}]");
    }
    foreach (var foot in Go2Model.Legs) {
      int g = Go2Model.Id(m, MujocoLib.mjtObj.mjOBJ_GEOM, foot);
      Check(Math.Abs(m->geom_friction[g * 3] - 0.8) < Eps && m->geom_condim[g] == 6 && m->geom_priority[g] == 1, $"foot {foot} contact params");
    }
    int floor = Go2Model.Id(m, MujocoLib.mjtObj.mjOBJ_GEOM, "floor");
    Check(Math.Abs(m->geom_friction[floor * 3] - 0.6) < Eps && m->geom_condim[floor] == 3, "floor contact params");
    Debug.Log($"[ModelCheck] OK nq={m->nq} nv={m->nv} nu={m->nu} mass={mass:F4} dt={m->opt.timestep} baseQpos={M.BaseQpos} jointQpos0={M.JointQpos[0]} cube0Qpos={M.CubeQpos[0]}");
  }

  static void Check(bool ok, string msg) {
    if (!ok) throw new InvalidOperationException("[ModelCheck] " + msg);
  }
}

}
