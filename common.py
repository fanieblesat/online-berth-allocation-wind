"""common.py — points of the campaign, seed pools and the single results logger (DESIGN.md §5-6)."""
from __future__ import annotations
import csv, os
from pathlib import Path
from lsps_core.model import make_instance, make_wind, BETA

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
RUNS = RESULTS / "runs.csv"
FIELDS = ["point", "method", "seed", "instance", "realization", "Z", "status", "time_s", "extra"]

WIND_REF = dict(mu_w=10.0, sigma_w=3.5, theta_w=0.12, alpha=0.5)
BASE = dict(WIND_REF, load=0.8, kappa=1.0)

def size(n, m=None):
    return dict(BASE, n_vessels=n, n_berths=m or n // 10)

POINTS = {
    # main grid (10:1)
    "V20": size(20), "V30": size(30), "V40": size(40), "V50": size(50),
    # sensitivity at the reference V30 (3 berths)
    **{f"mu{m}": dict(size(30), mu_w=float(m)) for m in (5, 15, 20)},
    **{f"sig{s}": dict(size(30), sigma_w=float(s)) for s in (1.5, 5.5, 8.0)},
    **{f"theta{t}": dict(size(30), theta_w=float(t)) for t in (0.06, 0.24)},
    **{f"alpha{a}": dict(size(30), alpha=float(a)) for a in (0.25, 0.75)},
    **{f"B{m}": size(30, m) for m in (4, 5)},                      # ratio 7.5 and 6
    **{f"load{l}": dict(size(30), load=float(l)) for l in (0.6, 0.95)},
    **{f"kappa{k}": dict(size(30), kappa=float(k)) for k in (0.5, 2.0)},
}
MAIN = ["V20", "V30", "V40", "V50"]
SENS = [p for p in POINTS if p not in MAIN]

TEST_SEEDS = list(range(1000, 1020))      # 20 test instances per point
VALID_SEEDS = list(range(4000, 4010))     # validation pool (checkpoint selection)
TRAIN_SEEDS = list(range(5000, 5200))     # training pool
CHECK_SEEDS = list(range(3000, 3010))     # exact-check pool
N_REAL = 3
TRAIN_RNG_SEEDS = [0, 1, 2, 3, 4]

def wind_seed(inst_seed, real): return 10_000_000 + 100 * inst_seed + real

def inst_of(point, seed): return make_instance(seed, POINTS[point] if isinstance(point, str) else point)

def wind_of(point, inst, real_seed):
    cfg = POINTS[point] if isinstance(point, str) else point
    return make_wind(real_seed, cfg, inst.beta)

def log_run(point, method, seed, instance, realization, Z, status, time_s, extra=""):
    new = not RUNS.exists()
    with open(RUNS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(dict(point=point, method=method, seed=seed, instance=instance, realization=realization,
                        Z=Z, status=status, time_s=f"{time_s:.4f}", extra=extra))

def done_keys(point, method, seed):
    if not RUNS.exists(): return set()
    out = set()
    for r in csv.DictReader(open(RUNS)):
        try:
            if r["point"] == point and r["method"] == method and int(r["seed"]) == seed:
                out.add((int(r["instance"]), int(r["realization"])))
        except (TypeError, ValueError):
            continue          # a row another job is writing at this moment
    return out

def policy_path(m, seed, episodes=4000):
    return RESULTS / (f"policy_dqn_B{m}_seed{seed}" + ("" if episodes == 4000 else f"_ep{episodes}") + ".pkl")
