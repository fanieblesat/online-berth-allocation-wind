"""ext_numbers.py — every number of the extension (DESIGN.md §9) that the text quotes and no table shows, from the record.
Usage: python3 ext_numbers.py [--results results] > results/ext_numbers.txt   (added 25 Oct 2026)"""
import argparse, math
import numpy as np, pandas as pd
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()
df = pd.read_csv(f"{a.results}/runs.csv")
ALL = ["V20", "V30", "V40", "V50", "mu5", "mu15", "mu20", "sig1.5", "sig5.5", "sig8.0", "theta0.06", "theta0.24", "alpha0.25",
       "alpha0.75", "B4", "B5", "load0.6", "load0.95", "kappa0.5", "kappa2.0"]
SENS = [p for p in ALL if not p.startswith("V")] + ["V30"]
def pi(pt, m): return df[(df.point == pt) & (df.method == m)].groupby("instance").Z.mean()
def gap(x, y): return (x.mean() - y.mean()) / y.mean() * 100
def wil(x, y): return stats.wilcoxon(x, y).pvalue if not np.allclose(x, y) else 1.0
def tost(x, y, frac=0.01):
    d = (x - y).values; n = len(d); mar = frac * float(y.mean()); se = d.std(ddof=1) / math.sqrt(n)
    return (1 - stats.t.cdf((d.mean() + mar) / se, n - 1)) < .05 and stats.t.cdf((d.mean() - mar) / se, n - 1) < .05
def exact(pt):
    d = df[(df.point == pt) & (df.method == "PI-AO")]; ok = d.groupby("instance").status.apply(lambda s: (s == "optimal").all())
    return d[d.instance.isin(ok[ok].index)].groupby("instance").Z.mean()
def best_rule(pt): return min(("FirstFree", "MyopicC", "LowExposure"), key=lambda m: pi(pt, m).mean())

print("== LookDP-R vs best rule, all 20 points (all instances)")
g = [(pt, gap(pi(pt, "LookDP-R"), pi(pt, best_rule(pt))), wil(pi(pt, "LookDP-R"), pi(pt, best_rule(pt)))) for pt in ALL]
print(f"  range {min(x[1] for x in g):.2f} .. {max(x[1] for x in g):.2f} %; max p {max(x[2] for x in g):.2g}; all below: {all(x[1] < 0 for x in g)}")
gs = [x for x in g if x[0] in SENS]; print(f"  sensitivity+V30: range {min(x[1] for x in gs):.2f} .. {max(x[1] for x in gs):.2f}")
print("== LookDP-R vs every rule and DQN at every point: worst gap and max p")
w = [(pt, m, gap(pi(pt, "LookDP-R"), pi(pt, m)), wil(pi(pt, "LookDP-R"), pi(pt, m))) for pt in ALL for m in ("FirstFree", "MyopicC", "LowExposure", "DQN")]
print(f"  worst {max(x[2] for x in w):.2f} %, max p {max(x[3] for x in w):.2g}")
print("== LookDP-R vs RollDP on all instances, sensitivity points + V30")
r = [(pt, gap(pi(pt, "LookDP-R"), pi(pt, "RollDP")), wil(pi(pt, "LookDP-R"), pi(pt, "RollDP")), tost(pi(pt, "LookDP-R"), pi(pt, "RollDP"))) for pt in SENS]
for x in r: print(f"  {x[0]:10} {x[1]:+.2f} p={x[2]:.3g} eq={x[3]}")
print(f"  range {min(x[1] for x in r):.2f} .. {max(x[1] for x in r):.2f}; p<0.05: {[x[0] for x in r if x[2] < .05]}; eq: {[x[0] for x in r if x[3]]}")
print("== LookDP-R vs RollDP on the decomposition instances (Delta_pi), sensitivity points + V30")
r = []
for pt in SENS:
    za = exact(pt); L = pi(pt, "LookDP-R")[za.index]; R = pi(pt, "RollDP")[za.index]; r.append((pt, gap(L, R), wil(L, R)))
print(f"  range {min(x[1] for x in r):.2f} .. {max(x[1] for x in r):.2f}; p<0.05: {[(x[0], round(x[1], 2), round(x[2], 3)) for x in r if x[2] < .05]}")
print("== LookDP-R vs LookDP-0 (value of the rollout terminal cost), all 20 points")
r = [(pt, gap(pi(pt, "LookDP-R"), pi(pt, "LookDP-0")), wil(pi(pt, "LookDP-R"), pi(pt, "LookDP-0"))) for pt in ALL]
print(f"  lower mean at {sum(x[1] < 0 for x in r)}/20; p<0.05 at {sum(x[2] < .05 and x[1] < 0 for x in r)}/20; range {min(x[1] for x in r):.2f} .. {max(x[1] for x in r):.2f}")
print("  sensitivity:", {x[0]: round(x[1], 2) for x in r if x[0] in SENS})
print("== LookDP-RV vs LookDP-R (value of the learned correction), all 20 points")
r = [(pt, gap(pi(pt, "LookDP-RV"), pi(pt, "LookDP-R")), wil(pi(pt, "LookDP-RV"), pi(pt, "LookDP-R"))) for pt in ALL]
print(f"  higher mean at {sum(x[1] > 0 for x in r)}/20; sig. higher (p<0.05) at {[x[0] for x in r if x[1] > 0 and x[2] < .05]}; sig. lower at {[x[0] for x in r if x[1] < 0 and x[2] < .05]}; range {min(x[1] for x in r):.2f} .. {max(x[1] for x in r):.2f}")
print("== RollDP-T vs RollDP, all 20 points (all instances)")
r = [(pt, gap(pi(pt, "RollDP-T"), pi(pt, "RollDP")), wil(pi(pt, "RollDP-T"), pi(pt, "RollDP"))) for pt in ALL]
print(f"  lower mean at {sum(x[1] < 0 for x in r)}/20; sig lower {[(x[0], round(x[1], 2), round(x[2], 3)) for x in r if x[1] < 0 and x[2] < .05]}; sig higher {[x[0] for x in r if x[1] > 0 and x[2] < .05]}; range {min(x[1] for x in r):.2f} .. {max(x[1] for x in r):.2f}")
print("== lowest-mean online policy per point (all instances)")
M = ["RollDP", "RollDP-T", "LookDP-0", "LookDP-R", "LookDP-RV", "Static", "FirstFree", "MyopicC", "LowExposure", "DQN"]
for pt in ALL: print(f"  {pt:10} {min(M, key=lambda m: pi(pt, m).mean())}")
print("== smallest Delta_info over RollDP, RollDP-T, LookDP-R (decomposition instances)")
for pt in ALL:
    za = exact(pt); v = {m: (pi(pt, m)[za.index].mean() - za.mean()) / za.mean() * 100 for m in ("RollDP", "RollDP-T", "LookDP-R")}
    b = min(v, key=v.get); print(f"  {pt:10} n={len(za):2} RollDP {v['RollDP']:.2f}  min {v[b]:.2f} ({b})  reduction {v['RollDP'] - v[b]:.2f} pts")
print("== LookDP-R at V40/V50 split by decomposition instances (RollDP few fallbacks) and the others")
for pt in ("V40", "V50"):
    za = exact(pt); L, R = pi(pt, "LookDP-R"), pi(pt, "RollDP"); o = R.index.difference(za.index)
    d = df[(df.point == pt) & (df.method == "RollDP")]; d = d.assign(fb=d.extra.str.extract(r"fallbacks=([0-9]+)")[0].astype(float))
    fb = d.groupby("instance").fb.mean() / int(pt[1:]) * 100
    print(f"  {pt}: decomp n={len(za)} gap {gap(L[za.index], R[za.index]):+.2f} (RollDP fallback {fb[za.index].mean():.1f}%); others n={len(o)} gap {gap(L[o], R[o]):+.2f} (fallback {fb[o].mean():.1f}%)")
print("== correction networks (results/correction_r2.csv)")
try:
    c = pd.read_csv(f"{a.results}/correction_r2.csv")
    print(c.groupby("m").agg(r2_V=("r2_V", "mean"), r2_RV=("r2_RV", "mean"), yr_mean=("yr_mean_days", "first"), yr_sd=("yr_sd_days", "first")).round(3).to_string())
    print(f"  R2 RV range {c.r2_RV.min():.3f}..{c.r2_RV.max():.3f}; R2 V range {c.r2_V.min():.3f}..{c.r2_V.max():.3f}")
except FileNotFoundError: print("  (run 6_correction_r2.py)")
lk = pd.read_csv(f"{a.results}/look_train.csv", header=None, names=["B", "seed", "mse_v", "mse_rv", "samples", "time"])
print("== training samples per m:", lk.groupby("B").samples.first().to_dict(), " time s range", lk.time.min(), lk.time.max())
