"""paper_rows.py — every table row of the results section from results/runs.csv (and exact.csv). Written before the
campaign (DESIGN.md rule); run it on partial records at any time.
Conventions: per-instance values = mean over the 3 realizations (and over the 5 seeds for DQN/QL); gaps in % of the
reference's mean; p = Wilcoxon signed-rank on per-instance means; TOST +-1% for 'no difference'.
Decomposition (DESIGN §1): Z(pi) - Z*_free = [Z*_AO - Z*_free] + [Z(RollDP) - Z*_AO] + [Z(pi) - Z(RollDP)];
Only instances whose PI-AO is optimal on all realizations enter the decomposition (the count is printed), policy gaps included. The cost of
arrival order is an interval: low = Z*_AO - Z_FREE (CP-SAT incumbent, an upper bound on Z*_free), high = Z*_AO -
max(CP-SAT bound, no-wait bound); the two coincide where CP-SAT proves optimality.
Usage: python paper_rows.py results/runs.csv > results/paper_rows.tex"""
import sys, math
import numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/runs.csv")
POL = ["Static", "FirstFree", "MyopicC", "LowExposure", "QL", "DQN", "RollDP"]

def per_inst(pt, m):
    d = df[(df.point == pt) & (df.method == m)]
    return d.groupby("instance").Z.mean() if len(d) else None

def tost(a, b, frac=0.01):
    d = (a - b).values; n = len(d); mar = frac * float(b.mean())
    if n < 3 or d.std(ddof=1) == 0: return False
    se = d.std(ddof=1) / math.sqrt(n)
    return (1 - stats.t.cdf((d.mean() + mar) / se, n - 1)) < .05 and stats.t.cdf((d.mean() - mar) / se, n - 1) < .05

def fp(p): return "---" if p != p else (f"{p:.3f}" if p >= 1e-3 else f"${p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}$")

def extra_val(d, key):
    return d.extra.fillna("").str.extract(key + r"=([-0-9.eE+naninf]+)")[0].astype(float)

print("%% ---- tab:decomp  (point & n [AO exact] & Z*AO & AO cost % [low, high] (FREE proven) & RollDP & VoI % & per policy: loss vs RollDP %, p)")
for pt in sorted(df.point.unique()):
    dA = df[(df.point == pt) & (df.method == "PI-AO")]
    if dA.empty or per_inst(pt, "RollDP") is None: print(f"%% {pt}: missing PI-AO or RollDP"); continue
    ok = dA.groupby("instance").status.apply(lambda x: (x == "optimal").all()); ok = ok[ok].index
    za = dA[dA.instance.isin(ok)].groupby("instance").Z.mean(); zr = per_inst(pt, "RollDP")
    idx = za.index.intersection(zr.index)
    dF = df[(df.point == pt) & (df.method == "PI-FREE") & df.instance.isin(idx)]
    if len(dF):
        lb = np.maximum(extra_val(dF, "bound").values, extra_val(dF, "nowait_lb").values)
        dF = dF.assign(LB=lb); zf = dF.groupby("instance").Z.mean(); zl = dF.groupby("instance").LB.mean()
        j = idx.intersection(zf.index); proven = int(dF.status.eq("OPTIMAL").sum())
        lo = (za[j].mean() - zf[j].mean()) / za[j].mean() * 100; hi = (za[j].mean() - zl[j].mean()) / za[j].mean() * 100
        aoc = f"[{lo:.2f}, {hi:.2f}]\\% ({proven}/{len(dF)})"
    else:
        aoc = "---"
    voi = (zr[idx].mean() - za[idx].mean()) / za[idx].mean() * 100
    cells = []
    for m in POL[:-1]:
        zp = per_inst(pt, m)
        if zp is None: cells.append("--- & ---"); continue
        j = zp.index.intersection(idx); g = (zp[j].mean() - zr[j].mean()) / zr[j].mean() * 100   # decomposition instances only
        p = stats.wilcoxon(zp[j], zr[j]).pvalue if not np.allclose(zp[j], zr[j]) else float("nan")
        cells.append(f"{g:+.2f}\\% & {fp(p)}{' eq.' if tost(zp[j], zr[j]) else ''}")
    print(f"{pt} & {len(idx)}/{dA.instance.nunique()} & {za[idx].mean():.1f} & {aoc} & {zr[idx].mean():.1f} & {voi:+.2f}\\% & "
          + " & ".join(cells) + " \\\\")
for m, key in (("Static", "fallback"), ("RollDP", "fallbacks")):
    d = df[df.method == m]
    if len(d): print(f"%% {m}: runs with {key} > 0: {int((extra_val(d, key) > 0).sum())} of {len(d)}")

print("\n%% ---- tab:seeds  (point & DQN per-seed gap to RollDP %)")
for pt in sorted(df.point.unique()):
    zr = per_inst(pt, "RollDP"); d = df[(df.point == pt) & (df.method == "DQN")]
    if zr is None or d.empty: continue
    g = [(s.groupby("instance").Z.mean().mean() - zr.mean()) / zr.mean() * 100 for _, s in d.groupby("seed")]
    print(f"{pt} & " + " & ".join(f"{x:+.2f}\\%" for x in g) + f" & {np.mean(g):+.2f} $\\pm$ {np.std(g, ddof=1) if len(g) > 1 else 0:.2f} \\\\")

print("\n%% ---- tab:time  (point & method & mean s per run)")
for pt in sorted(df.point.unique()):
    for m in ["PI-AO", "PI-FREE", "RollDP", "DQN", "MyopicC"]:
        d = df[(df.point == pt) & (df.method == m)]
        if len(d): print(f"{pt} & {m} & {d.time_s.mean():.2f} \\\\")

# ---- figures quoted only in the text (Section 6), from the record (added 22 Oct 2026)
print("\n%% ---- textfigs")
for pt in ["V20", "V30", "V40", "V50"]:
    for x_, y_ in [("DQN", "FirstFree"), ("DQN", "MyopicC"), ("FirstFree", "MyopicC")]:
        a_, b_ = per_inst(pt, x_), per_inst(pt, y_); j = a_.index.intersection(b_.index)
        print(f"%% {pt} {x_} vs {y_}: {(a_[j].mean() - b_[j].mean()) / b_[j].mean() * 100:+.2f} p={stats.wilcoxon(a_[j], b_[j]).pvalue:.2g} wins {int((a_[j] < b_[j]).sum())}/{len(j)} eq={tost(a_[j], b_[j])}")
ao = df[df.method == "PI-AO"]
for pt in ["V40", "V50", "B4", "B5"]:
    d = ao[ao.point == pt]; ok = d.groupby("instance").status.apply(lambda s: (s == "optimal").all()); ok = ok[ok].index
    r = df[(df.point == pt) & (df.method == "RollDP")]; fb = extra_val(r, "fallbacks"); n = {"B4": 30, "B5": 30}.get(pt, int(pt[1:]) if pt[1:].isdigit() else 30)
    rs = r[r.instance.isin(ok)]; fbs = extra_val(rs, "fallbacks")
    print(f"%% {pt}: AO proven on all 3 realizations for {len(ok)} instances; RollDP fallback share all {fb.sum() / (len(r) * n) * 100:.1f}%, on those instances {fbs.sum() / max(1, len(rs) * n) * 100:.1f}%")
f = df[df.method == "PI-FREE"].merge(ao, on=["point", "instance", "realization"], suffixes=("_f", "_a"))
for pt, g in f.groupby("point"):
    print(f"%% {pt}: FREE incumbent < AO value in {int((g.Z_f < g.Z_a).sum())}/{len(g)} runs (all runs; the AO value is the DP incumbent where the DP was capped); FREE at horizon {int(g.extra_f.str.contains('at_horizon=1').sum())}")
d = df[(df.point == "mu20") & (df.method == "DQN")]; r = per_inst("mu20", "RollDP")
print("%% mu20 DQN per seed vs RollDP: " + ", ".join(f"{s}: {(x.groupby('instance').Z.mean().mean() - r.mean()) / r.mean() * 100:+.1f}" for s, x in d.groupby("seed")))
# further text figures (added 22 Oct 2026, audit draft 3)
ex = pd.read_csv(sys.argv[1].replace("runs.csv", "exact.csv")) if len(sys.argv) > 1 else pd.read_csv("results/exact.csv")
for pt in ["V40", "V50"]:
    e = ex[ex.point == pt]; print(f"%% {pt}: DP proven within 600 s (realization 0): {int(((e.dp_status == 'optimal') & (e.dp_time <= 600)).sum())} of {len(e)}")
    d = ao[ao.point == pt].copy(); d["lab"] = extra_val(d, "labels"); ok = d.groupby("instance").status.apply(lambda s: (s == "optimal").all())
    li = d.groupby("instance").lab.mean(); print(f"%% {pt}: mean DP labels per run, decomposition instances {li[ok[ok].index].mean():.0f}, others {li[ok[~ok].index].mean():.0f}")
for pt in ["B4", "B5"]:
    s_ = df[(df.point == pt) & (df.method == "Static")]; print(f"%% {pt}: Static fallback runs {int((extra_val(s_, 'fallback') > 0).sum())} of {len(s_)}")
d = ao[ao.point == "B5"]; ok = d.groupby("instance").status.apply(lambda s: (s == "optimal").all()); ok = ok[ok].index
zr = per_inst("B5", "RollDP")[ok]
for m in ["FirstFree", "MyopicC"]:
    z = per_inst("B5", m)[ok]; print(f"%% B5 decomposition instances ({len(ok)}): {m} vs RollDP {(z.mean() - zr.mean()) / zr.mean() * 100:+.2f} p={stats.wilcoxon(z, zr).pvalue:.2f} below on {int((z < zr).sum())}")
for pt in ["V20", "V50"]:
    a_, b_ = per_inst(pt, "DQN"), per_inst(pt, "LowExposure"); print(f"%% {pt} DQN vs LowExposure: {(a_.mean() - b_.mean()) / b_.mean() * 100:+.2f} p={stats.wilcoxon(a_, b_).pvalue:.3f}")
for pt in ["V40", "V50"]:
    g = f[f.point == pt]; g = g[g.status_a == "optimal"]; print(f"%% {pt}: FREE < AO in {int((g.Z_f < g.Z_a).sum())} of {len(g)} runs with proven AO optimum")
# post-hoc pilot: RollDP with a 300-s decision limit at V50 (CHANGELOG 22 Oct 2026)
p_ = df[df.method == "RollDP-cap300"]
if len(p_):
    m_ = p_.merge(df[(df.point == "V50") & (df.method == "RollDP")], on=["point", "instance", "realization"], suffixes=("_300", "_60"))
    for c in ("_300", "_60"): m_["fb" + c] = extra_val(m_.rename(columns={"extra" + c: "extra"}), "fallbacks").values
    print(f"%% pilot: {len(m_)} runs on instances {[int(x) for x in sorted(m_.instance.unique())]}")
    for i, g in m_.groupby("instance"):
        print(f"%%   {i}: fallbacks/run 60s {g.fb_60.mean():.1f} -> 300s {g.fb_300.mean():.1f}; Z change % {[round(x, 2) for x in ((g.Z_300 - g.Z_60) / g.Z_60 * 100)]}; time ratio {[round(x, 1) for x in g.time_s_300 / g.time_s_60]}")
    w = m_[m_.fb_60 > 0]
    print(f"%%   runs with fallbacks at 60 s: {len(w)}; fallback share {w.fb_60.sum() / (50 * len(w)) * 100:.1f}% -> {w.fb_300.sum() / (50 * len(w)) * 100:.1f}%; "
          f"Z change per run {((w.Z_300 - w.Z_60) / w.Z_60 * 100).min():+.2f}% to {((w.Z_300 - w.Z_60) / w.Z_60 * 100).max():+.2f}%, total {(w.Z_300.sum() - w.Z_60.sum()) / w.Z_60.sum() * 100:+.2f}%")
