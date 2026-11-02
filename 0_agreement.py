"""0_agreement.py — exactness and consistency checks (DESIGN.md §6), on the exact-check pool. Writes results/agreement.txt.
  A  plain DP (IC_TORS) = DP with normalization+dominance+bounding = CP-SAT-AO = simulator replay of the DP plan
  B  CP-SAT-FREE optimum is reproduced by the simulator replaying its berth assignment and order
  C  FIFO: completion is non-decreasing in the ready period for every vessel and berth (Proposition)
  D  RollDP with perfect information reproduces PI-AO exactly (principle of optimality of the online harness)
Usage: python 0_agreement.py [--workers 8]"""
import argparse
from common import *
from lsps_core.model import evaluate_ao, evaluate_sequence, fifo_check
from lsps_core.dp import solve_ao
from lsps_core.exact import solve_cpsat
import lsps_core.online as O

ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
lines = []; ok_all = True
for n, m in [(6, 2), (8, 2), (10, 3), (12, 3)]:
    cfg = dict(BASE, n_vessels=n, n_berths=m)
    for s in CHECK_SEEDS:
        inst = inst_of(cfg, s); real = wind_of(cfg, inst, wind_seed(s, 0))
        d = solve_ao(inst, real); p = solve_ao(inst, real, dominance=False, normalize=False, bound=False)
        ca = solve_cpsat(inst, real, "ao", time_limit=600, workers=a.workers)
        cf = solve_cpsat(inst, real, "free", time_limit=600, workers=a.workers)
        rep = evaluate_ao(inst, real, d.plan)["Z"]
        rf = evaluate_sequence(inst, real, cf["berth"], cf["order"])["Z"] if cf["berth"] else float("nan")
        fifo = fifo_check(inst, real)
        orig = O.forecast; O.forecast = lambda r, t, c, b: r
        rp = O.run_online(inst, real, cfg, O.RollDP(time_cap=None))["Z"]; O.forecast = orig
        okA = d.cost == p.cost == ca["obj"] == rep and ca["status"] == "OPTIMAL"
        okB = cf["status"] == "OPTIMAL" and abs(rf - cf["obj"]) < 1e-9 and cf["obj"] <= d.cost
        okD = rp == d.cost
        ok = okA and okB and fifo and okD; ok_all &= ok
        lines.append(f"{'AGREE' if ok else 'DISAGREE'} {n}x{m} seed {s}: DP {d.cost} plain {p.cost} CP-AO {ca['obj']} ({ca['status']}) "
                     f"replay {rep} | CP-FREE {cf['obj']} ({cf['status']}) replay {rf} | FIFO {fifo} | RollDP-PI {rp}")
        print(lines[-1], flush=True)
lines.append("ALL AGREE" if ok_all else "SOME CHECKS FAILED")
(RESULTS / "agreement.txt").write_text("\n".join(lines) + "\n"); print(lines[-1])
