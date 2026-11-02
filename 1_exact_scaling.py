"""1_exact_scaling.py — exact methods on the perfect-information problem (tab:exact): per main size, the 20 test
instances x realization 0: DP-AO (labels, time, status) and CP-SAT-AO solved cold, without the DP hint, so that the
two exact methods are compared independently (status, objective, bound, time). CP-SAT-FREE on
the same instance and realization is the PI-FREE row of runs.csv (3_benchmark.py --methods PI), not solved twice.
Resumable (results/exact.csv). Usage: python 1_exact_scaling.py --point V30 [--seeds 1000-1004] [--dp-cap 1800]
[--cp-limit 600] [--workers 1]"""
import argparse, csv, time
from common import *
from lsps_core.dp import solve_ao
from lsps_core.exact import solve_cpsat
ap = argparse.ArgumentParser(); ap.add_argument("--point", required=True); ap.add_argument("--dp-cap", type=float, default=1800)
ap.add_argument("--cp-limit", type=float, default=600); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seeds", default=None); a = ap.parse_args()
def parse_seeds(x):
    if x is None: return TEST_SEEDS
    out = []
    for part in x.split(","):
        lo, _, hi = part.partition("-"); out += list(range(int(lo), int(hi or lo) + 1))
    return out
OUT = RESULTS / "exact.csv"; F = ["point", "instance", "dp_status", "dp_obj", "dp_labels", "dp_peak", "dp_time",
      "ao_status", "ao_obj", "ao_bound", "ao_time"]
done = {(r["point"], int(r["instance"])) for r in csv.DictReader(open(OUT))} if OUT.exists() else set()
for s in parse_seeds(a.seeds):
    if (a.point, s) in done: continue
    inst = inst_of(a.point, s); real = wind_of(a.point, inst, wind_seed(s, 0))
    d = solve_ao(inst, real, time_cap=a.dp_cap)
    hint = d.plan if d.plan else None
    ca = solve_cpsat(inst, real, "ao", time_limit=a.cp_limit, workers=a.workers, hint_plan=None)   # cold: independent of the DP
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=F); (w.writeheader() if new else None)
        w.writerow(dict(point=a.point, instance=s, dp_status=d.status, dp_obj=d.cost, dp_labels=d.labels, dp_peak=d.peak,
                        dp_time=f"{d.time_s:.2f}", ao_status=ca["status"], ao_obj=ca["obj"], ao_bound=ca["bound"],
                        ao_time=f"{ca['time_s']:.1f}"))
    print(f"{a.point} {s}: DP {d.cost} [{d.status}, {d.labels} labels, {d.time_s:.1f}s] | CP-AO {ca['obj']} {ca['status']} {ca['time_s']:.0f}s", flush=True)
