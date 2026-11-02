"""tests_ube.py — checks of the literature baseline before any pilot (DESIGN.md §10). Writes results/tests_ube.txt.
  i   vectorized completion = scalar completion() on random cases (realized and sampled wind)
  ii  perfect information (the realized wind as the only sample, no time limit): UBE realizes the PI-FREE optimum
      on the exact-check instances (6-12 vessels)
  iii non-anticipation: commits at periods <= t* are unchanged when the realized wind after t* is replaced"""
import numpy as np
from common import BASE, CHECK_SEEDS, VALID_SEEDS, POINTS, inst_of, wind_of, wind_seed, RESULTS
from lsps_core.model import completion, Wind, RHO
from lsps_core.exact import solve_cpsat
from lsps_core.ube import completion_vec, sample_paths, run_ube

out = []; ok_all = True
def rep(s, ok):
    global ok_all
    ok_all &= ok; out.append(("PASS " if ok else "FAIL ") + s); print(out[-1], flush=True)

# i
rng = np.random.default_rng(1); bad = 0; nchk = 0
for pt in ("V20", "mu20", "sig8.0"):
    cfg = POINTS[pt]
    for s in VALID_SEEDS[:3]:
        inst = inst_of(pt, s); real = wind_of(pt, inst, wind_seed(s, 0))
        for w in [real] + sample_paths(real, 50, cfg, inst.beta, 3, [s, 0, 50, 77]):
            for b in range(inst.m):
                r = rng.integers(1, 700, size=200); v = rng.integers(0, inst.n, size=200)
                need = np.array([inst.h[x] / (inst.q[x] * RHO) for x in v])
                cv = completion_vec(w, b, r, need)
                cs = np.array([completion(w, b, int(rr), inst.h[x], inst.q[x])[1] for rr, x in zip(r, v)])
                bad += int((cv != cs).sum()); nchk += len(r)
rep(f"i   vectorized completion: {nchk} cases, {bad} mismatches", bad == 0)

# ii
for n, m in [(6, 2), (8, 2), (10, 3), (12, 3)]:
    cfg = dict(BASE, n_vessels=n, n_berths=m); okn = True; msg = []
    for s in CHECK_SEEDS:
        inst = inst_of(cfg, s); real = wind_of(cfg, inst, wind_seed(s, 0))
        cf = solve_cpsat(inst, real, "free", time_limit=600, workers=1)
        r = run_ube(inst, real, cfg, rho=0.95, cap0=None, cap=None, samples_fn=lambda t, real=real: [real])
        ok = cf["status"] == "OPTIMAL" and abs(r["Z"] - cf["obj"]) < 1e-9
        okn &= ok; msg.append(f"{s}:{r['Z']:.0f}/{cf['obj']:.0f}")
    rep(f"ii  PI UBE = PI-FREE optimum {n}x{m}: " + " ".join(msg), okn)

# iii
for pt, s, tstar in (("V20", VALID_SEEDS[0], 30), ("V30", VALID_SEEDS[1], 40)):
    cfg = POINTS[pt]; inst = inst_of(pt, s); real = wind_of(pt, inst, wind_seed(s, 0))
    W2 = list(real.W); g = np.random.default_rng(9)
    for k in range(tstar, len(W2)): W2[k] = float(max(0.0, 25.0 * g.random()))
    real2 = Wind(W2, inst.beta, real.alpha)
    r1 = run_ube(inst, real, cfg, S=20, cap0=20, cap=5); r2 = run_ube(inst, real2, cfg, S=20, cap0=20, cap=5)
    # commit period = the berthing decision; recover it from both runs by re-simulating would need internals, so compare
    # berths and completions of vessels completed (realized) before t*: those depend only on commits before t*
    early = [v for v in range(inst.n) if r1["C"][v] is not None and r1["C"][v] <= tstar]
    same = all(r1["berth"][v] == r2["berth"][v] and r1["C"][v] == r2["C"][v] for v in early)
    rep(f"iii non-anticipation {pt} instance {s}, t*={tstar}: {len(early)} vessels completed by t*, identical: {same}", same and len(early) > 0)

out.append("ALL PASS" if ok_all else "SOME CHECKS FAILED"); print(out[-1])
(RESULTS / "tests_ube.txt").write_text("\n".join(out) + "\n")
