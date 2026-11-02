"""7_ube_pilot.py — validation pilot of the literature baseline (DESIGN.md §10): UBE with rho in {0.5, 0.95} at one
point on the VALIDATION pool (10 instances, realization 0), with MyopicC and LookDP-R (K=5) as references.
Usage: python3 7_ube_pilot.py --point V30 [--rho 0.5,0.95] [--n-valid 10]   -> results/ube_pilot.csv (appends; resumable)"""
import argparse, csv, time
from common import POINTS, inst_of, wind_of, wind_seed, VALID_SEEDS, RESULTS
from lsps_core import online as O
from lsps_core.lookahead import LookDP
from lsps_core.ube import run_ube

ap = argparse.ArgumentParser(); ap.add_argument("--point", default="V30"); ap.add_argument("--rho", default="0.5,0.95")
ap.add_argument("--n-valid", type=int, default=10); ap.add_argument("--refs", type=int, default=1); a = ap.parse_args()
pt = a.point; cfg = POINTS[pt]; out = RESULTS / "ube_pilot.csv"; new = not out.exists()
done = set()
if out.exists():
    for row in csv.DictReader(open(out)):
        if row["point"] == pt: done.add((row["method"], int(row["instance"])))
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["point", "method", "instance", "realization", "Z", "time_s", "events", "hint_used"])
    for s in VALID_SEEDS[:a.n_valid]:
        inst = inst_of(pt, s); real = wind_of(pt, inst, wind_seed(s, 0))
        if a.refs:
            for name, pol in (("MyopicC", O.pol_myopic), ("LookDP-R", LookDP(5, "rollout"))):
                if (name, s) in done: continue
                t0 = time.perf_counter(); Z = O.run_online(inst, real, cfg, pol)["Z"]
                w.writerow([pt, name, s, 0, Z, f"{time.perf_counter() - t0:.2f}", "", ""]); f.flush()
        for rho in [float(x) for x in a.rho.split(",")]:
            if (f"UBE-{rho}", s) in done: continue
            r = run_ube(inst, real, cfg, rho=rho, S=100, cap0=600, cap=60, real_idx=0)
            w.writerow([pt, f"UBE-{rho}", s, 0, r["Z"], f"{r['time_s']:.1f}", r["events"], r["hint_used"]]); f.flush()
        print(f"  {pt} validation instance {s}: done", flush=True)
