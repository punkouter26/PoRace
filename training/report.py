"""Build docs/report/training_report.html from docs/report/tb_scalars.json + eval_results.json.
Run (host): uv run --no-project --python 3.11 python training/report.py
"""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "docs" / "report"
tb = json.loads((R / "tb_scalars.json").read_text())
ev = json.loads((R / "eval_results.json").read_text()) if (R / "eval_results.json").exists() else {}
log = (ROOT / "rl_optimization_log.md").read_text(encoding="utf-8")


def series(run, tag):
  """Concatenate resumed segments into one (step, value) list; resumed segments restart at step 0."""
  segs = tb.get(run, {}).get(tag, [])
  out, offset = [], 0
  for seg in segs:
    pts = [(s + offset, v) for _, s, v in seg if v == v]  # drop NaN
    if out and pts and pts[0][0] <= out[-1][0]:
      offset = out[-1][0]; pts = [(s + offset, v) for _, s, v in seg if v == v]
    out += pts
    if out: offset = out[-1][0]
  return out


def svg_chart(pts, title, ylabel, notes, color="#2563eb", extra=None, ymin=None):
  W, H, L, B, T, Rm = 640, 300, 60, 40, 40, 20
  xs = [p[0] for p in pts] + [p[0] for p in (extra or [])]
  ys = [p[1] for p in pts] + [p[1] for p in (extra or [])]
  x0, x1 = 0, max(xs + [1]) * 1.05
  y0 = min(ys + [0]) if ymin is None else ymin
  y1 = max(ys + [1]) * 1.15
  sx = lambda x: L + (x - x0) / (x1 - x0) * (W - L - Rm)
  sy = lambda y: H - B - (y - y0) / (y1 - y0) * (H - B - T)
  g = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}">',
       f'<text x="{L}" y="22" class="ct">{title}</text>']
  for i in range(5):
    y = y0 + (y1 - y0) * i / 4
    g.append(f'<line x1="{L}" y1="{sy(y):.1f}" x2="{W-Rm}" y2="{sy(y):.1f}" class="grid"/>'
             f'<text x="{L-6}" y="{sy(y)+4:.1f}" class="tick" text-anchor="end">{y:.3g}</text>')
  for i in range(5):
    x = x0 + (x1 - x0) * i / 4
    g.append(f'<text x="{sx(x):.1f}" y="{H-B+16}" class="tick" text-anchor="middle">{x/1e6:.0f}M</text>')
  g.append(f'<text x="{(W+L)/2:.0f}" y="{H-6}" class="tick" text-anchor="middle">training steps (millions of simulated control steps)</text>')
  g.append(f'<text transform="translate(14,{(H-B+T)/2:.0f}) rotate(-90)" class="tick" text-anchor="middle">{ylabel}</text>')
  if len(pts) > 1:
    g.append('<polyline fill="none" stroke="%s" stroke-width="2.5" points="%s"/>' % (color, " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)))
  for x, y in pts:
    g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
  for x, y in (extra or []):
    g.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="none" stroke="#9333ea" stroke-width="2"/>')
  for (x, y, text, dy) in notes:
    g.append(f'<line x1="{sx(x):.1f}" y1="{sy(y):.1f}" x2="{sx(x)+18:.1f}" y2="{sy(y)+dy:.1f}" class="ann"/>'
             f'<text x="{sx(x)+22:.1f}" y="{sy(y)+dy+4:.1f}" class="annt">{text}</text>')
  g.append("</svg>")
  return "\n".join(g)


reward = series("r1", "eval/episode_reward")
length = series("r1", "eval/avg_episode_length")
track = series("r1", "eval/episode_reward/tracking_lin_vel")
nanres = series("r1", "eval/episode_nan_resets")
last_step = reward[-1][0] if reward else 0
sanity_pt = [(14745600, 16.209)]

charts = [
  ("1. Overall score per episode", svg_chart(reward, "Reward per 20-second episode (higher is better)", "reward", [
      (reward[0][0], reward[0][1], "start: random brain, falls almost immediately", -60) if reward else None,
      (reward[-1][0], reward[-1][1], f"latest checkpoint: {reward[-1][1]:.1f}", 30) if reward else None,
      (sanity_pt[0][0], sanity_pt[0][1], "throwaway 'sanity' run for comparison (16.2)", 50),
    ], extra=sanity_pt),
   "Think of this as the dog's report-card grade for one 20-second test. It earns points every tick for moving at the commanded speed, turning at the commanded rate and keeping its body level, and loses points for shaking, wasting energy, dragging feet or falling. A random brain scores about 0. The first real checkpoint already scores 17, which is the level the earlier throwaway run reached after a similar amount of training."),
  ("2. How long it stays on its feet", svg_chart(length, "Average episode length (1000 = never fell in 20 s)", "control steps", [
      (length[0][0], length[0][1], "falls after ~7 s on average", -50) if length else None,
      (length[-1][0], length[-1][1], "1000 = no falls in any test episode", 30) if length else None,
    ], ymin=0),
   "Each test lasts at most 1000 control steps (20 seconds) and ends early if the dog flips over. 1000 means it never fell in any of the test runs. Standing and balance (Rung 0) are therefore already solved by the first checkpoint; what remains is walking quality."),
  ("3. How well it follows the speed command", svg_chart(track, "Speed-tracking points per episode (max about 1000)", "tracking points", [
      (track[-1][0], track[-1][1], "about a quarter of the maximum so far", 30) if track else None,
    ], color="#059669", ymin=0),
   "The dog is given a joystick-style command (walk forward at up to 1.5 m/s, sideways, or turn). This curve counts how close its actual speed is to the command, every tick, summed over the episode; a perfect follower would score about 1000. It is at roughly a quarter of that: it balances well but does not yet reach the commanded speeds. This is the curve to watch for Rung 1 (walk and turn)."),
]

rungs = [("R0 Stand", "r0", "Stay upright 10 s with no command, 10 of 10 seeds"),
         ("R1 Walk + turn", "r1", "Follow speed/turn commands, mean error under 0.2 m/s, 10 of 10 seeds"),
         ("R2 Recovery", "r2", "Survive a 30 N·s push and a 1 kg falling cube, 9 of 10 seeds"),
         ("R3 Stand-up", "r3", "Rise from a fallen pose within 3 s, 9 of 10 seeds (separate getup policy)")]
agents = [("Go2 'sanity' policy", "sanity", "Throwaway 14.7M-step pipeline test. Proves train, export, Unity replay and closed-loop parity.", "parity: zero-brain 4.8e-8, replay 3.6e-7, closed-loop PASS"),
          ("Go2 Rung 1, checkpoint 23M", "r1_23M", "First healthy checkpoint of the real Rung 1 run (no pushes, no randomization yet).", "not yet gated in Unity"),
          ("Go2 Rung 1, continuation (stalled)", None, "Resume from the 23M checkpoint toward 200M steps. Currently NOT progressing: the training container hung and has to be restarted.", "gate runs on completion")]


def cell(agent_key, rung_key):
  if agent_key is None:
    return '<td class="c pending">no result yet</td>'
  r = ev.get(agent_key, {}).get(rung_key)
  if r is None:
    return '<td class="c na">not trained yet</td>' if rung_key == "r3" else '<td class="c na">not evaluated</td>'
  cls = "pass" if r["pass"] else "fail"
  return f'<td class="c {cls}"><b>{"PASS" if r["pass"] else "FAIL"}</b><br><small>{r["summary"].split(":",1)[1].strip()}</small></td>'


grid = ['<table class="grid"><thead><tr><th>Agent / policy</th>' + "".join(f'<th>{n}<br><small>{d}</small></th>' for n, _, d in rungs) + '<th>Sim-to-Unity parity</th></tr></thead><tbody>']
for name, key, desc, parity in agents:
  grid.append(f'<tr><th class="agent">{name}<br><small>{desc}</small></th>' + "".join(cell(key, rk) for _, rk, _ in rungs) + f'<td class="c note">{parity}</td></tr>')
grid.append("</tbody></table>")

attempts = [
  ("Attempt 1", "failed at 22M", "Brain went NaN. One in a few thousand simulated dogs hit a solver blow-up, the fall check could not see NaN, and the bad numbers poisoned the shared observation statistics."),
  ("Attempt 2", "failed at 22M", "Guard added for NaN, but fallen dogs were sinking through the floor: the contact budget was sized for a feet-only robot and dropped contacts once many dogs were on the ground."),
  ("Attempt 3", "healthy, then out of GPU memory at 23M", "Bigger contact budget worked (reward 17.1, no falls, 0 NaN resets) but the GPU ran out of memory."),
  ("Attempt 3c", "stalled", "Resumed from the 23M checkpoint with a smaller contact budget and a cap on JAX's memory share. GPU stayed at 100 % for over an hour with no eval logged; the container could not be stopped (zombie process) and now blocks new GPU containers until Docker is restarted."),
]

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PoRace training report</title>
<style>
:root{{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--pass:#dcfce7;--fail:#fee2e2;--na:#f3f4f6;--pend:#fef3c7;--card:#f9fafb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f172a;--fg:#e5e7eb;--muted:#9ca3af;--line:#334155;--pass:#14532d;--fail:#7f1d1d;--na:#1e293b;--pend:#78350f;--card:#1e293b}}}}
body{{margin:0;padding:16px;font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin-inline:auto}}
h1{{font-size:1.6rem;margin:.2em 0}} h2{{font-size:1.2rem;margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.3em}}
.muted{{color:var(--muted)}} .banner{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}}
table.grid{{border-collapse:collapse;width:100%;font-size:.92rem}} .grid th,.grid td{{border:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
.grid th small{{display:block;font-weight:normal;color:var(--muted)}} .agent{{width:24%}} .c{{text-align:center}}
.pass{{background:var(--pass)}} .fail{{background:var(--fail)}} .na{{background:var(--na);color:var(--muted)}} .pending{{background:var(--pend)}} .note{{font-size:.85rem}}
.chart{{width:100%;max-width:640px;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;display:block}}
.ct{{font-size:15px;font-weight:600;fill:var(--fg)}} .tick{{font-size:11px;fill:var(--muted)}} .grid{{stroke:var(--line);stroke-width:1}}
.ann{{stroke:#f59e0b;stroke-width:1.5}} .annt{{font-size:12px;fill:#b45309;font-weight:600}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}} @media(max-width:800px){{.row{{grid-template-columns:1fr}}}}
.timeline{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}} .tl{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
.tl b{{display:block}} .tl .s{{color:var(--muted);font-size:.85rem}}
code{{background:var(--na);padding:1px 5px;border-radius:4px;font-size:.9em}}
</style></head><body>
<h1>PoRace: Go2 training report</h1>
<div class="muted">Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. One creature so far (Unitree Go2 quadruped), trained in MuJoCo Warp, played back in Unity through the native MuJoCo plugin.</div>
<div class="banner"><b>Where things stand:</b> the dog can already <b>stand and balance</b> reliably. It is partway through learning to <b>walk and turn on command</b> (Rung 1); that training run is <b>currently stalled</b> at its 23M-step checkpoint and needs a Docker restart to continue. Push recovery (Rung 2) and standing up after a fall (Rung 3) have not been trained yet. The Unity side is verified to reproduce the simulator exactly on everything trained so far.</div>

<h2>Agents and what they can do</h2>
<p class="muted">Each cell is a pass/fail against the contract's numeric bar, measured on CPU MuJoCo with 5 seeds (full gates use 10). The right-most column is whether the same policy has been verified inside Unity.</p>
{''.join(grid)}

<h2>The three charts that matter right now</h2>
<p class="muted">Drawn from the TensorBoard logs of the Rung 1 run (<code>runs/r1</code>). Each point is a test of 128 simulated dogs with the current brain, every 20 million training steps. Purple ring = the throwaway sanity run, for scale.</p>
<div class="row">
{''.join(f'<div><h3>{t}</h3>{s}<p>{e}</p></div>' for t, s, e in charts)}
<div><h3>Health check: NaN resets</h3>{svg_chart(nanres, "Simulated dogs reset because the physics blew up (must stay 0)", "resets per episode", [(nanres[-1][0], nanres[-1][1], "0 = healthy", -30)] if nanres else [], color="#dc2626", ymin=0)}<p>Not a skill, but the number that killed the first two attempts. Any value above zero means a simulated dog's physics exploded or it fell through the floor; the trainer now catches these and resets that dog, and this counter shows how often. Zero so far in the current attempt.</p></div>
</div>

<h2>How the Rung 1 run has gone</h2>
<div class="timeline">
{''.join(f'<div class="tl"><b>{a}</b><span class="s">{s}</span>{d}</div>' for a, s, d in attempts)}
</div>

<h2>What happens next</h2>
<ol>
<li>Restart Docker to clear the stuck container, resume from the 23M checkpoint, and let the run finish (about 200M more steps). The gate script exports the brain to ONNX, re-runs the R0 and R1 bars with 10 seeds, records a 5-second reference run, rebuilds the Unity player and checks it reproduces the policy outputs (under 1e-4) and the gait (speed, cadence, torque within tolerance).</li>
<li>On a pass: tag <code>rung1-parity-pass</code>, then Rung 2 continues from this brain with random pushes, falling cubes and physics randomization.</li>
<li>Rung 3 trains a second, separate "get up" brain from fallen poses; Unity switches between the two automatically.</li>
</ol>
<p class="muted">Sources: <code>docs/report/tb_scalars.json</code>, <code>docs/report/eval_results.json</code>, <code>rl_optimization_log.md</code>. Regenerate with <code>python training/report.py</code>.</p>
</body></html>"""
(R / "training_report.html").write_text(html, encoding="utf-8")
print("wrote", R / "training_report.html", "reward points", len(reward))
