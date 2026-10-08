"""Build docs/report/training_report.html from docs/report/tb_final.json (TensorBoard dump) + the measured results below.
Run (host): python training/report.py
"""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "docs" / "report"
tb = json.loads((R / "tb_final.json").read_text())

# Locomotion training chain in order, with what changed in each phase.
PHASES = [("r1", "Rung 1 start", "#2563eb"), ("r1b", "Rung 1: learn to walk", "#2563eb"),
          ("r2", "Rung 2: pushes, cubes, random physics", "#d97706"), ("r2b", "more of the same", "#d97706"),
          ("r2c", "pushes doubled to ~34 N·s", "#dc2626")]


def chain(tag):
  """Concatenate runs on one step axis. Returns [(run, color, [(step, value)])] and phase boundaries."""
  out, bounds, off = [], [], 0
  for run, label, color in PHASES:
    pts = sorted(set((s, v) for s, v in tb.get(run, {}).get(tag, []) if v == v))
    if run == "r1": pts = [p for p in pts if p[0] > 0 or p[1] < 5]  # drop the duplicate step-0 eval of the hung resume
    if not pts: continue
    seg = [(s + off, v) for s, v in pts]
    out.append((run, color, seg)); bounds.append((off, label, color))
    off = seg[-1][0]
  return out, bounds, off


def svg(segs, bounds, xmax, title, ylabel, notes=(), ymin=0, ymax=None, hline=None):
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
  g.append(f'<text x="{(W+L)/2:.0f}" y="{H-6}" class="tick" text-anchor="middle">cumulative training steps (millions)</text>')
  g.append(f'<text transform="translate(14,{(H-B+T)/2:.0f}) rotate(-90)" class="tick" text-anchor="middle">{ylabel}</text>')
  if hline is not None:
    g.append(f'<line x1="{L}" y1="{sy(hline[0]):.1f}" x2="{W-Rm}" y2="{sy(hline[0]):.1f}" class="bar"/><text x="{W-Rm-4}" y="{sy(hline[0])-5:.1f}" class="annt" text-anchor="end">{hline[1]}</text>')
  for _, color, seg in segs:
    if len(seg) > 1:
      g.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in seg)}"/>')
    g += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>' for x, y in seg]
  for x, y, text, dx, dy in notes:
    g.append(f'<line x1="{sx(x):.1f}" y1="{sy(y):.1f}" x2="{sx(x)+dx:.1f}" y2="{sy(y)+dy:.1f}" class="ann"/><text x="{sx(x)+dx+(4 if dx >= 0 else -4):.1f}" y="{sy(y)+dy+4:.1f}" class="annt" text-anchor="{"start" if dx >= 0 else "end"}">{text}</text>')
  g.append("</svg>")
  return "\n".join(g)


rew, bounds, xmax = chain("eval/episode_reward")
trk, _, _ = chain("eval/episode_reward/tracking_lin_vel")
ln, _, _ = chain("eval/avg_episode_length")
allr = [p for _, _, s in rew for p in s]
walk_pt = next(p for p in allr if p[1] > 22)
r2_start = next(s[0] for r, _, s in rew if r == "r2")
r2c_seg = next(s for r, _, s in rew if r == "r2c")
getup = sorted(set((s, v) for s, v in tb["r3"]["eval/episode_reward"]))

c1 = svg(rew, bounds, xmax, "Locomotion brain: score per 20-second test episode", "reward", [
    (allr[0][0], allr[0][1], "random brain: falls at once", 14, -26),
    (walk_pt[0], walk_pt[1], "starts walking", -16, -34),
    (r2_start[0], r2_start[1], "harder test begins: score drops", -14, 38),
    (r2c_seg[-1][0], r2c_seg[-1][1], "final brain", -12, -40)], ymax=30)
c2 = svg(trk, bounds, xmax, "How closely it follows the speed command (about 1000 = perfect)", "tracking points", [
    (trk[1][2][0][0], trk[1][2][0][1], "balancing only, not walking yet", 14, 34)], ymax=1000)
c3 = svg([("r3", "#059669", getup)], [(0, "Stand-up brain (separate network)", "#059669")], getup[-1][0],
         "Stand-up brain: score per 6-second episode (about 20 = perfect)", "reward", [
    (getup[2][0], getup[2][1], "half-trained: only some starts recover", 12, 30),
    (getup[-1][0], getup[-1][1], "20 of 20 stand up within 3 s", -14, -26)], ymax=20)

GRID = [
  ("Stand still", "Upright 10 s, no command", "10 / 10", "pass", "covered by closed-loop runs", "pass"),
  ("Walk and turn", "Random speed and turn commands incl. 1.5 m/s from a stop; mean error under 0.2 m/s", "10 / 10<br><small>error 0.10&ndash;0.12 m/s</small>", "pass", "matches trainer<br><small>1.2588 vs 1.2588 m/s at the 1.5 command</small>", "pass"),
  ("Take a hit", "30 N·s shove while walking, then a 1 kg cube dropped from 1 m; stay on its feet 90 % of the time", "40 / 40", "pass", "20 / 20", "pass"),
  ("Stand up", "From a random fallen pose, standing within 3 s, 90 % of the time", "20 / 20<br><small>slowest 2.2 s</small>", "pass", "10 / 10", "pass"),
  ("Fall, get up, carry on", "Flipped onto its back mid-walk; switches to the stand-up brain, then back, and is walking again", "20 / 20", "pass", "10 / 10<br><small>walking again after about 1.5 s</small>", "pass"),
]
grid = "".join(f'<tr><th class="agent">{n}<small>{d}</small></th><td class="c {tc}">{t}</td><td class="c {uc}">{u}</td></tr>' for n, d, t, tc, u, uc in GRID)

STORY = [
  ("Attempts 1 and 2", "failed", "The brain turned to NaN after 22M steps. First cause: rare physics blow-ups were never reset. Second cause: fallen dogs sank through the floor because the contact budget was sized for a feet-only robot."),
  ("Attempt 3", "stalled", "Healthy but ran the GPU out of memory, then hung so badly that Docker and finally the PC had to be restarted."),
  ("Rung 1", "206M steps, 2 h 10 min", "Resumed with memory sized per simulated world. Stood for the first 70M steps, then learned to walk."),
  ("Rung 2", "161M steps", "Pushes, falling cubes and randomized mass, friction and motor gains. Passed only after the training pushes were doubled: they had peaked at half of what the test demands."),
  ("Rung 3", "52M steps, 28 min", "A second brain trained from random fallen poses. Unity switches between the two automatically."),
]

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PoRace training report</title>
<style>
:root{{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--pass:#dcfce7;--card:#f9fafb;--warn:#fef3c7}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f172a;--fg:#e5e7eb;--muted:#9ca3af;--line:#334155;--pass:#14532d;--card:#1e293b;--warn:#78350f}}}}
body{{margin:0;padding:16px;font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin-inline:auto}}
h1{{font-size:1.6rem;margin:.2em 0}} h2{{font-size:1.2rem;margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.3em}} h3{{font-size:1rem;margin:.2em 0 .4em}}
.muted{{color:var(--muted)}} .banner{{background:var(--pass);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}} .warn{{background:var(--warn)}}
table.grid{{border-collapse:collapse;width:100%;font-size:.95rem}} table.grid th,table.grid td{{border:1px solid var(--line);padding:9px;vertical-align:top;text-align:left}}
table.grid small{{display:block;font-weight:normal;color:var(--muted)}} .agent{{width:44%}} .c{{text-align:center}} .pass{{background:var(--pass)}}
.chart{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;display:block}}
.ct{{font-size:14px;font-weight:600;fill:var(--fg)}} .tick{{font-size:11px;fill:var(--muted)}} .gl{{stroke:var(--line);stroke-width:1}}
.ann{{stroke:#f59e0b;stroke-width:1.5}} .annt{{font-size:11.5px;fill:#b45309;font-weight:600}} .bar{{stroke:#16a34a;stroke-dasharray:5 4}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}} @media(max-width:820px){{.row{{grid-template-columns:1fr}}}}
.tlw{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}} .tl{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:.92rem}}
.tl b{{display:block}} .tl .s{{color:var(--muted);font-size:.85rem;display:block}} .legend span{{display:inline-block;margin-right:14px;font-size:.85rem}} .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}}
code{{background:var(--card);padding:1px 5px;border-radius:4px;font-size:.9em}}
</style></head><body>
<h1>PoRace: Go2 training report</h1>
<div class="muted">Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. One creature: the Unitree Go2 robot dog, trained in MuJoCo Warp and played back in Unity through the native MuJoCo plugin (no Unity physics).</div>
<div class="banner"><b>All behaviours are achieved.</b> The dog stands, walks and turns on command up to 1.5 m/s, survives hard shoves and falling cubes, and gets back up and carries on after being flipped over. Every result below was measured in the training simulator and again inside Unity.</div>

<h2>What the dog can do</h2>
<table class="grid"><thead><tr><th>Behaviour and pass bar</th><th>Training simulator</th><th>Inside Unity</th></tr></thead><tbody>{grid}</tbody></table>
<p class="muted">Two brains: a locomotion brain (stand, walk, turn, take hits) and a stand-up brain. Unity runs physics at 250 Hz and the brain at 50 Hz, the same as training. Cost per racer: about 0.9 ms per control step, so roughly 3.6 ms estimated for four racers against a 5 ms budget.</p>

<h2>The three charts that tell the story</h2>
<div class="legend muted"><span><i class="sw" style="background:#2563eb"></i>learning to walk</span><span><i class="sw" style="background:#d97706"></i>pushes, cubes, random physics</span><span><i class="sw" style="background:#dc2626"></i>pushes doubled</span><span><i class="sw" style="background:#059669"></i>stand-up brain</span></div>
<div class="row">
<div><h3>1. Overall score</h3>{c1}<p>The dog's report-card grade for one 20-second test: points for moving at the commanded speed and staying level, minus points for shaking, wasting energy or falling. It sat near 18 while it only balanced, climbed to 26 once it learned to walk, then <b>dropped on purpose</b> each time the test was made harder (shoves and cubes, then shoves twice as strong). A lower score on a harder test is not a worse dog.</p></div>
<div><h3>2. Following the speed command</h3>{c2}<p>How close the dog's real speed is to the joystick command, added up over the episode. For the first 70 million steps it stayed low: the dog had found a lazy answer, standing still and only turning. Then walking "clicked" and this is the curve that shows it. It dips again when it also has to cope with being shoved.</p></div>
<div><h3>3. Learning to stand up</h3>{c3}<p>A separate brain, started on its back, side or upside down. Halfway through it could only right itself from some positions. By the end it stands up from all 20 random starts in under 2.2 seconds.</p></div>
<div><h3>Staying on its feet</h3>{svg(ln, bounds, xmax, "Average test length (1000 = never fell in 20 s)", "control steps", [], ymin=0, ymax=1100)}<p>Each test ends early if the dog flips over. It reached 1000 (no falls) almost immediately, which is why "stand" was solved long before "walk". It stays near 1000 even with the hardest pushes.</p></div>
</div>

<h2>How it got here</h2>
<div class="tlw">{''.join(f'<div class="tl"><b>{a}</b><span class="s">{s}</span>{d}</div>' for a, s, d in STORY)}</div>

<h2>Honest caveats</h2>
<div class="banner warn"><ul style="margin:0;padding-left:1.2em">
<li><b>Top speed from a standstill relies on a command ramp.</b> Hit with an instant jump to 1.3 m/s or more, the brain braces and does not move. The controller therefore ramps the command (reaching 1.5 m/s takes a quarter second), in training tests and in Unity alike.</li>
<li><b>One parity run drifts.</b> At 0.5 m/s Unity and the simulator agree on speed, step rhythm and motor effort, but their joint angles wander apart by 5 seconds. At 1.5 m/s they stay within 0.00001. The drift is believed to be numerical, not yet proven.</li>
<li><b>Four-racer cost is an estimate</b> from one racer; no multi-dog scene exists yet.</li>
<li><b>Not built yet:</b> tracks, the creature and map registries, the pre-race menu, rough terrain, other creatures, Android.</li>
</ul></div>
<p class="muted">Sources: <code>docs/report/tb_final.json</code> (TensorBoard), <code>rl_optimization_log.md</code>. Regenerate with <code>python training/report.py</code>.</p>
</body></html>"""
(R / "training_report.html").write_text(html, encoding="utf-8")
print("wrote", R / "training_report.html", len(html), "chars; locomotion points", len(allr), "xmax", xmax)
