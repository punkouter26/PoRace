"""Build docs/report/training_report.html from docs/report/tb_final.json (TensorBoard dump) + the measured results below.
Charts are drawn from the TensorBoard scalars (same numbers TensorBoard plots) so they can carry annotations.
Run (host): python training/report.py
"""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "docs" / "report"
tb = json.loads((R / "tb_final.json").read_text())
BLUE, AMBER, RED, GREEN, GREY, PURPLE = "#2563eb", "#d97706", "#dc2626", "#059669", "#6b7280", "#7c3aed"

PHASES = [("r1", "start", BLUE), ("r1b", "learn to walk", BLUE), ("r2", "pushes, cubes, random physics", AMBER),
          ("r2b", "more of the same", AMBER), ("r2c", "pushes doubled", RED)]


def pts_of(run, tag):
  p = sorted(set((s, v) for s, v in tb.get(run, {}).get(tag, []) if v == v))
  if run == "r1": p = [q for q in p if q[0] > 0 or q[1] < 5]   # drop the duplicate step-0 eval of a hung resume
  return p


def chain(tag):
  out, bounds, off = [], [], 0
  for run, label, color in PHASES:
    p = pts_of(run, tag)
    if not p: continue
    seg = [(s + off, v) for s, v in p]
    out.append((run, color, seg)); bounds.append((off, label, color)); off = seg[-1][0]
  return out, bounds, off


def svg(segs, bounds, xmax, title, ylabel, notes=(), ymin=0, ymax=None, xlabel="cumulative training steps (millions)", hline=None):
  W, H, L, B, T, Rm = 680, 320, 62, 42, 46, 20
  ys = [v for _, _, s in segs for _, v in s]
  y0, y1 = ymin, (ymax if ymax else max(ys) * 1.18)
  x1 = xmax * 1.03
  sx = lambda x: L + x / x1 * (W - L - Rm)
  sy = lambda y: H - B - (y - y0) / (y1 - y0) * (H - B - T)
  g = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}"><text x="{L}" y="20" class="ct">{title}</text>']
  for i, (bx, label, color) in enumerate(bounds):
    ex = bounds[i + 1][0] if i + 1 < len(bounds) else xmax
    g.append(f'<rect x="{sx(bx):.1f}" y="{T}" width="{sx(ex)-sx(bx):.1f}" height="{H-B-T}" fill="{color}" opacity="{0.06 if i % 2 else 0.11}"/>')
  for i in range(5):
    y = y0 + (y1 - y0) * i / 4
    g.append(f'<line x1="{L}" y1="{sy(y):.1f}" x2="{W-Rm}" y2="{sy(y):.1f}" class="gl"/><text x="{L-6}" y="{sy(y)+4:.1f}" class="tick" text-anchor="end">{y:.3g}</text>')
  for i in range(6):
    x = x1 * i / 5
    g.append(f'<text x="{sx(x):.1f}" y="{H-B+16}" class="tick" text-anchor="middle">{x/1e6:.0f}M</text>')
  g.append(f'<text x="{(W+L)/2:.0f}" y="{H-6}" class="tick" text-anchor="middle">{xlabel}</text>')
  g.append(f'<text transform="translate(14,{(H-B+T)/2:.0f}) rotate(-90)" class="tick" text-anchor="middle">{ylabel}</text>')
  if hline: g.append(f'<line x1="{L}" y1="{sy(hline[0]):.1f}" x2="{W-Rm}" y2="{sy(hline[0]):.1f}" class="bar"/><text x="{W-Rm-4}" y="{sy(hline[0])-5:.1f}" class="annt" text-anchor="end">{hline[1]}</text>')
  for _, color, seg in segs:
    if len(seg) > 1: g.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in seg)}"/>')
    g += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>' for x, y in seg]
  for x, y, text, dx, dy in notes:
    g.append(f'<line x1="{sx(x):.1f}" y1="{sy(y):.1f}" x2="{sx(x)+dx:.1f}" y2="{sy(y)+dy:.1f}" class="ann"/><text x="{sx(x)+dx+(4 if dx >= 0 else -4):.1f}" y="{sy(y)+dy+4:.1f}" class="annt" text-anchor="{"start" if dx >= 0 else "end"}">{text}</text>')
  g.append("</svg>"); return "\n".join(g)


def bars(groups, series, title, ymax, ylabel):
  """groups: [label]; series: [(name, color, [values])] -> grouped bar chart with an 'asked for' diagonal marker."""
  W, H, L, B, T, Rm = 680, 320, 62, 52, 46, 20
  gw = (W - L - Rm) / len(groups); bw = gw * 0.8 / len(series)
  sy = lambda y: H - B - y / ymax * (H - B - T)
  g = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{title}"><text x="{L}" y="20" class="ct">{title}</text>']
  for i in range(5):
    y = ymax * i / 4
    g.append(f'<line x1="{L}" y1="{sy(y):.1f}" x2="{W-Rm}" y2="{sy(y):.1f}" class="gl"/><text x="{L-6}" y="{sy(y)+4:.1f}" class="tick" text-anchor="end">{y:.2g}</text>')
  for gi, lab in enumerate(groups):
    x0 = L + gi * gw + gw * 0.1
    asked = float(lab)
    g.append(f'<line x1="{x0-3:.1f}" y1="{sy(asked):.1f}" x2="{x0+gw*0.8+3:.1f}" y2="{sy(asked):.1f}" class="bar"/>')
    for si, (name, color, vals) in enumerate(series):
      v = vals[gi]
      if v is None: continue
      h = max(1.5, (H - B) - sy(max(v, 0)))
      g.append(f'<rect x="{x0+si*bw:.1f}" y="{sy(max(v,0)):.1f}" width="{bw-2:.1f}" height="{h:.1f}" fill="{color}" rx="2"/>'
               f'<text x="{x0+si*bw+bw/2-1:.1f}" y="{sy(max(v,0))-4:.1f}" class="tick" text-anchor="middle">{v:.2f}</text>')
    g.append(f'<text x="{x0+gw*0.4:.1f}" y="{H-B+16}" class="tick" text-anchor="middle">asked {lab} m/s</text>')
  lx = L
  for name, color, _ in series:
    g.append(f'<rect x="{lx}" y="{H-16}" width="11" height="11" fill="{color}" rx="2"/><text x="{lx+15}" y="{H-6}" class="tick">{name}</text>'); lx += 22 + 6.4 * len(name)
  g.append(f'<line x1="{lx}" y1="{H-10}" x2="{lx+18}" y2="{H-10}" class="bar"/><text x="{lx+22}" y="{H-6}" class="tick">speed asked for</text>')
  g.append(f'<text transform="translate(14,{(H-B+T)/2:.0f}) rotate(-90)" class="tick" text-anchor="middle">{ylabel}</text></svg>')
  return "\n".join(g)


rew, bounds, xmax = chain("eval/episode_reward")
trk, _, _ = chain("eval/episode_reward/tracking_lin_vel")
allr = [p for _, _, s in rew for p in s]
walk_pt = next(p for p in allr if p[1] > 22)
r2_start = next(s[0] for r, _, s in rew if r == "r2")
final_pt = next(s for r, _, s in rew if r == "r2c")[-1]
getup = pts_of("r3", "eval/episode_reward")
fail = pts_of("r4_fixed_sigma_failed", "eval/episode_reward")
new = pts_of("r4", "eval/episode_reward")

c1 = svg(rew, bounds, xmax, "Walking brain: score per 20-second test", "reward", [
    (allr[0][0], allr[0][1], "random brain: falls at once", 14, -26), (walk_pt[0], walk_pt[1], "starts walking", -16, -34),
    (r2_start[0], r2_start[1], "harder test begins: score drops", -14, 38), (final_pt[0], final_pt[1], "brain in the game today", -12, -40)], ymax=30)
c2 = svg(trk, bounds, xmax, "How closely it follows the speed command (about 1000 = perfect)", "tracking points", [
    (trk[1][2][0][0], trk[1][2][0][1], "balancing only, not walking yet", 14, 34)], ymax=1000)
fx = max([p[0] for p in fail + new] + [1])
c3 = svg([("fail", RED, fail), ("new", PURPLE, new)], [(0, "", GREY)], fx, "Teaching it to run: two attempts", "reward", [
    (fail[-1][0], fail[-1][1], "attempt 1: score rises, but it learned to REFUSE fast commands", -10, 46),
    (new[-1][0], new[-1][1], "attempt 2 (fixed scoring): just started" if len(new) < 2 else f"attempt 2: {new[-1][1]:.1f}", 14, -30)],
    ymax=22, xlabel="training steps in this attempt (millions)")
c4 = svg([("r3", GREEN, getup)], [(0, "", GREEN)], getup[-1][0], "Stand-up brain: score per 6-second test (about 20 = perfect)", "reward", [
    (getup[2][0], getup[2][1], "half-trained: only some starts recover", 12, 30), (getup[-1][0], getup[-1][1], "20 of 20 stand up within 3 s", -14, -26)],
    ymax=20, xlabel="training steps (millions)")
speed = bars(["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"], [
    ("brain in the game", BLUE, [0.42, 0.92, 1.32, 1.52, 0.00, 0.00]),
    ("run attempt 1 (failed)", RED, [0.43, 0.93, 0.00, 0.00, 0.00, 0.00])],
    "Speed asked for vs. speed actually reached (from a standing start)", 3.2, "speed reached (m/s)")

P, F, N, T = "pass", "fail", "na", "pend"
AGENTS = [
  ("Go2 walking brain", "In the game now. 389M training steps.",
   [("10 / 10", P), ("10 / 10<small>error 0.10&ndash;0.12 m/s</small>", P), ("40 / 40<small>Unity 20 / 20</small>", P), ("uses stand-up brain", N), ("1.3&ndash;1.5 m/s<small>refuses 2.5+</small>", F), ("matches Unity<small>0.000001 rad</small>", P)]),
  ("Go2 stand-up brain", "In the game now. 52M steps. Takes over when the dog is down.",
   [("&ndash;", N), ("&ndash;", N), ("19 / 20<small>shoved while rising</small>", P), ("20 / 20<small>Unity 10 / 10, slowest 2.2 s</small>", P), ("&ndash;", N), ("Unity 10 / 10", P)]),
  ("Go2 + both brains together", "The actual racer: walk, fall, get up, carry on.",
   [("10 / 10", P), ("10 / 10", P), ("40 / 40", P), ("20 / 20<small>back walking in ~1.5 s</small>", P), ("1.3&ndash;1.5 m/s", F), ("Unity 10 / 10", P)]),
  ("Go2 run brain, attempt 1", "Stopped at 46M steps and discarded.",
   [("yes", P), ("only up to 1.0 m/s", F), ("not tested", N), ("&ndash;", N), ("refuses 1.5 and above<small>worse than before</small>", F), ("not gated", N)]),
  ("Go2 run brain, attempt 2", "Training now with corrected scoring. Target 2.5 m/s.",
   [("training", T), ("training", T), ("training", T), ("&ndash;", N), ("no result yet", T), ("not gated", N)]),
  ("G1 humanoid", "Second creature. Body verified in Unity; no brain trained yet.",
   [("no brain yet", N), ("no brain yet", N), ("no brain yet", N), ("not planned yet", N), ("no brain yet", N), ("body matches Unity<small>0.0000009 rad, no brain</small>", P)]),
]
COLS = ["Stand", "Walk and turn<small>random commands, error under 0.2 m/s</small>", "Take a hit<small>30 N·s shove + falling 1 kg cube</small>",
        "Stand up<small>from fallen, within 3 s</small>", "Run fast<small>above 1.5 m/s</small>", "Same in Unity as in training"]
grid = "".join(f'<tr><th class="agent">{n}<small>{d}</small></th>' + "".join(f'<td class="c {k}">{t}</td>' for t, k in cells) + "</tr>" for n, d, cells in AGENTS)

RACES = [("Sprint, 100 m, 4 dogs", "79.3 / 79.4 / 79.5 / 80.5 s", "267 bumps, 0 respawns"),
         ("Oval, 2 laps (195 m), 4 dogs", "150.6 / 150.7 / 153.2 / 155.2 s", "419 bumps, 0 respawns"),
         ("Sprint, leader flipped mid-pack, 5 races", "all 20 finishes", "0 respawns; un-stick rule fired twice"),
         ("Cost for 4 dogs", "2.0 ms per control step", "budget 5.0 ms")]

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PoRace training report</title>
<style>
:root{{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--pass:#dcfce7;--fail:#fee2e2;--na:#f3f4f6;--pend:#fef3c7;--card:#f9fafb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f172a;--fg:#e5e7eb;--muted:#9ca3af;--line:#334155;--pass:#14532d;--fail:#7f1d1d;--na:#1e293b;--pend:#78350f;--card:#1e293b}}}}
body{{margin:0;padding:16px;font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:1180px;margin-inline:auto}}
h1{{font-size:1.6rem;margin:.2em 0}} h2{{font-size:1.2rem;margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.3em}} h3{{font-size:1rem;margin:.2em 0 .4em}}
.muted{{color:var(--muted)}} .banner{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}} .warn{{background:var(--pend)}}
.scroll{{overflow-x:auto}} table.grid{{border-collapse:collapse;width:100%;min-width:900px;font-size:.9rem}} table.grid th,table.grid td{{border:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
table.grid small{{display:block;font-weight:normal;color:var(--muted)}} .agent{{width:21%}} .c{{text-align:center}}
.pass{{background:var(--pass)}} .fail{{background:var(--fail)}} .na{{background:var(--na);color:var(--muted)}} .pend{{background:var(--pend)}}
.key span{{display:inline-block;padding:2px 10px;border-radius:6px;margin-right:8px;font-size:.85rem;border:1px solid var(--line)}}
.chart{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;display:block}}
.ct{{font-size:14px;font-weight:600;fill:var(--fg)}} .tick{{font-size:11px;fill:var(--muted)}} .gl{{stroke:var(--line);stroke-width:1}}
.ann{{stroke:#f59e0b;stroke-width:1.5}} .annt{{font-size:11.5px;fill:#b45309;font-weight:600}} .bar{{stroke:#16a34a;stroke-width:2;stroke-dasharray:5 4}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}} @media(max-width:820px){{.row{{grid-template-columns:1fr}}}}
table.small{{border-collapse:collapse;width:100%;font-size:.92rem}} table.small td,table.small th{{border:1px solid var(--line);padding:7px 9px;text-align:left}}
.legend span{{display:inline-block;margin-right:14px;font-size:.85rem}} .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}}
code{{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em}}
</style></head><body>
<h1>PoRace: training report</h1>
<div class="muted">Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. Two creatures: the Unitree Go2 robot dog (racing now) and the Unitree G1 humanoid (body ready, not yet trained). Trained in MuJoCo Warp, played in Unity through the native MuJoCo plugin.</div>
<div class="banner"><b>In one paragraph.</b> The dog has two brains that work: one to stand, walk, turn and take hits, one to get back up. Together they race four-at-a-time on two tracks. What it cannot do yet is <b>run</b>: it tops out around 1.3 to 1.5 m/s. The first attempt to teach running backfired and was thrown away; a corrected second attempt is training now. The humanoid's body is in Unity and matches the training simulator, but it has no brain yet.</div>

<h2>Agents and what each can do</h2>
<div class="key"><span class="pass">passes its bar</span><span class="fail">does not</span><span class="pend">in training</span><span class="na">not applicable / not trained</span></div>
<div class="scroll"><table class="grid"><thead><tr><th>Agent</th>{''.join(f'<th>{c}</th>' for c in COLS)}</tr></thead><tbody>{grid}</tbody></table></div>
<p class="muted">Scores are "passes out of attempts" in the training simulator unless marked Unity. "Same in Unity" means the Unity game reproduces the simulator: for the walking brain a 5-second run at 1.5 m/s stays within a millionth of a radian in every joint.</p>

<h2>The three training charts that matter most</h2>
<p class="muted">These are the TensorBoard curves (<code>eval/episode_reward</code> and <code>eval/episode_reward/tracking_lin_vel</code>), redrawn from the same logged numbers so they can carry notes. Each dot is a test of 128 simulated dogs. Open the live versions with TensorBoard on <code>training/runs</code>.</p>
<div class="legend muted"><span><i class="sw" style="background:{BLUE}"></i>learning to walk</span><span><i class="sw" style="background:{AMBER}"></i>pushes, cubes, random physics</span><span><i class="sw" style="background:{RED}"></i>pushes doubled / failed run attempt</span><span><i class="sw" style="background:{PURPLE}"></i>run attempt 2</span></div>
<div class="row">
<div><h3>1. Overall score of the walking brain</h3>{c1}<p>Think of it as a report-card grade for one 20-second test: points for moving at the speed asked and staying level, minus points for shaking, wasting energy or falling. It sat near 18 while the dog only balanced, climbed to 26 once it learned to walk, then <b>dropped on purpose</b> each time the test got harder (shoves and cubes, then shoves twice as strong). A lower score on a harder test is not a worse dog; the last red dot is the brain racing today.</p></div>
<div><h3>2. Does it go the speed it is told?</h3>{c2}<p>How close the dog's real speed is to the joystick command, added up over the test. For the first 70 million steps it stayed low because the dog had found a lazy answer: stand still and only turn. Then walking "clicked", and this is the curve that shows it. It dips when it also has to survive being shoved.</p></div>
<div><h3>3. Teaching it to run</h3>{c3}<p><b>Red, attempt 1:</b> the score went up, which looked like progress, but the dog was getting worse: it learned to stand still whenever asked to go fast. The scoring gave almost no credit for "partly fast", so refusing was the cheapest answer. <b>Purple, attempt 2:</b> scoring fixed so partial speed earns partial credit; it has only just started. This chart is the one to watch next.</p></div>
<div><h3>Bonus: the stand-up brain</h3>{c4}<p>A separate brain that starts on its back, side or upside down. Halfway through it could right itself only from some positions; by the end it stands up from every random start in under 2.2 seconds.</p></div>
</div>

<h2>Why "run fast" is red: asked vs. reached</h2>
<div class="row"><div>{speed}</div><div><p>Each group is one speed command from a standstill. The green dashed line is what was asked; the bar is what the dog actually did.</p>
<ul><li><b>Blue, today's brain:</b> follows commands well up to 1.5, manages 1.5 when asked for 2.0, and <b>stands still</b> when asked for 2.5 or more.</li>
<li><b>Red, run attempt 1:</b> after more training it stood still at 1.5 as well. That is why it was discarded.</li></ul>
<p>A bar at zero does not mean the dog fell. It stays upright and braces. In races this is why a dog bumped to a halt used to stay stuck; the race now restarts its speed command gently to get it moving again.</p></div></div>

<h2>How the working brains race</h2>
<table class="small"><thead><tr><th>Race (Unity, no graphics, repeatable)</th><th>Result</th><th>Notes</th></tr></thead><tbody>
{''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>' for a, b, c in RACES)}</tbody></table>

<h2>Honest caveats</h2>
<div class="banner warn"><ul style="margin:0;padding-left:1.2em">
<li><b>No running yet.</b> Everything in the "Run fast" column is either failing or unproven. Attempt 2 may also fail.</li>
<li><b>The humanoid is only a body.</b> Its green cell means the physics match between Unity and the simulator with no brain attached. Its training environment and tests are written but have never been run.</li>
<li><b>Fast starts rely on a helper.</b> The game ramps speed commands over a quarter second and un-sticks stalled racers; without those the current brain freezes on sudden fast commands.</li>
<li><b>One parity run drifts.</b> At 0.5 m/s the joints of the Unity and simulator runs wander apart by 5 seconds while speed, step rhythm and effort still agree. Not yet explained.</li>
<li><b>Charts 1 to 3 have few dots</b> in their latest sections because checkpoints are 23 million steps apart.</li>
</ul></div>
<p class="muted">Sources: <code>docs/report/tb_final.json</code> (TensorBoard scalars), <code>training/speed_probe.py</code>, <code>rl_optimization_log.md</code>. Regenerate with <code>python training/report.py</code>.</p>
</body></html>"""
(R / "training_report.html").write_text(html, encoding="utf-8")
print("wrote", R / "training_report.html", len(html), "chars; attempt-2 points:", len(new))
