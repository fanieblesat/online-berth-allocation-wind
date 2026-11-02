"""4_figures.py — the two figures of Section 6 from the record (results/runs.csv). Added 22 Oct 2026.
fig_decomp.pdf : loss of every online policy relative to the AO optimum, per main size, on the decomposition instances
                 (PI-AO proven on all three realizations), mean with 95% t-interval over instances.
fig_voi.pdf    : Delta_info (RollDP vs the AO optimum) with 95% t-interval against the wind parameters and alpha (V30).
fig_exact.pdf  : time of the two exact AO methods per test instance (realization 0) against size, with their limits.
fig_train.pdf  : validation turnaround of the five DQN training seeds against the episode, per berth count.
fig_sens.pdf   : policy loss vs RollDP against mu_w, sigma_w and the load (V30), per policy.
fig_scale.pdf  : (extension, 25 Oct 2026) seconds per decision per size, and turnaround vs the best rule on all instances.
Usage: python3 4_figures.py [--results results] [--out ../overleaf]"""
import argparse, math, os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="../overleaf")
a = ap.parse_args()
df = pd.read_csv(os.path.join(a.results, "runs.csv"))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"          # RollDP, DQN, neutral (reference palette)
COL = {"RollDP": BLUE, "DQN": ORANGE, "Static": "#1baf7a", "FirstFree": "#eda100", "MyopicC": "#e87ba4", "LowExposure": "#6250d6",
       "RollDP-T": "#1c5cab", "LookDP-R": INK, "LookDP-0": GRAY, "LookDP-RV": GRAY, "UBE-0.95": "#b8312f"}       # extension: LookDP-R in ink, ablations neutral
MRK = {"RollDP": "o", "DQN": "D", "Static": "s", "FirstFree": "^", "MyopicC": "v", "LowExposure": "P",
       "RollDP-T": "o", "LookDP-R": "*", "LookDP-0": "x", "LookDP-RV": "d", "UBE-0.95": "h"}
LS = {"RollDP-T": (0, (3, 2)), "LookDP-0": (0, (1, 1.5)), "LookDP-RV": (0, (4, 1.5, 1, 1.5))}
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})

def inst(pt, m): return df[(df.point == pt) & (df.method == m)].groupby("instance").Z.mean()
def exact_inst(pt):
    d = df[(df.point == pt) & (df.method == "PI-AO")]
    ok = d.groupby("instance").status.apply(lambda s: (s == "optimal").all()); ok = ok[ok].index
    return d[d.instance.isin(ok)].groupby("instance").Z.mean()
def rel_ci(z, za):
    """mean and 95% t-interval of (z - za) in % of mean(za), paired over instances."""
    j = z.index.intersection(za.index); d = (z[j] - za[j]).values; base = za[j].mean(); n = len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return d.mean() / base * 100, h / base * 100, n

# ---------------------------------------------------------------- fig_decomp
POL = [(m, m, COL[m]) for m in ("LookDP-R", "RollDP-T", "RollDP", "DQN", "FirstFree", "MyopicC", "LowExposure", "Static")]
sizes = ["V20", "V30", "V40"]
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
for ax, pt in zip(axs, sizes):
    za = exact_inst(pt); n = len(za)
    for k, (m, lab, col) in enumerate(POL):
        mu, h, _ = rel_ci(inst(pt, m), za); y = len(POL) - 1 - k
        ax.plot([mu - h, mu + h], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(mu, y, "o", ms=5.5, color=col, mec="white", mew=1.0, zorder=3)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_title(f"{pt} ($n={n}$)", fontsize=8.5, color=INK, pad=4)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axs[0].set_yticks(range(len(POL))); axs[0].set_yticklabels([p[1] for p in POL][::-1])
axs[1].set_xlabel(r"Turnaround above the AO optimum $Z^\ast_{\mathrm{AO}}$ (%)")
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(a.out, "fig_decomp.pdf")); plt.close(fig)

# ---------------------------------------------------------------- fig_voi
FAC = [(r"$\mu_w$ (m/s)", ["mu5", "V30", "mu15", "mu20"], [5, 10, 15, 20]),
       (r"$\sigma_w$", ["sig1.5", "V30", "sig5.5", "sig8.0"], [1.5, 3.5, 5.5, 8.0]),
       (r"$\theta_w$ (1/h)", ["theta0.06", "V30", "theta0.24"], [0.06, 0.12, 0.24]),
       (r"$\alpha$", ["alpha0.25", "V30", "alpha0.75"], [0.25, 0.5, 0.75])]
fig, axs = plt.subplots(1, 4, figsize=(6.3, 2.2), sharey=True)
for ax, (lab, pts, vals) in zip(axs, FAC):
    mus, hs = [], []
    for pt in pts:
        mu, h, _ = rel_ci(inst(pt, "RollDP"), exact_inst(pt)); mus.append(mu); hs.append(h)
    mus, hs = np.array(mus), np.array(hs)
    ax.fill_between(vals, mus - hs, mus + hs, color=BLUE, alpha=0.15, lw=0)
    ax.plot(vals, mus, color=BLUE, lw=2)
    ax.plot(vals, mus, "o", ms=5, color=BLUE, mec="white", mew=1.0, zorder=3)
    ref = pts.index("V30"); ax.plot(vals[ref], mus[ref], "o", ms=5, color="white", mec=BLUE, mew=1.4, zorder=4)
    for m in ("RollDP-T", "LookDP-R"):
        y = [rel_ci(inst(pt, m), exact_inst(pt))[0] for pt in pts]
        ax.plot(vals, y, color=COL[m], lw=1.2, ls=LS.get(m, "-"), marker=MRK[m], ms=4.5 if m != "LookDP-R" else 6, mec="white", mew=0.6, zorder=3,
                label=m if ax is axs[0] else None)
    if ax is axs[0]: ax.plot([], [], color=BLUE, lw=2, marker="o", ms=5, mec="white", label="RollDP (95% band)")
    ax.set_xlabel(lab); ax.set_xticks(vals)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel(r"$\Delta_{\mathrm{info}}$ (% of $Z^\ast_{\mathrm{AO}}$)")
axs[0].set_ylim(bottom=0)
h, l = axs[0].get_legend_handles_labels(); o = [l.index(x) for x in ("RollDP (95% band)", "RollDP-T", "LookDP-R")]
fig.legend([h[i] for i in o], [l[i] for i in o], loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
fig.savefig(os.path.join(a.out, "fig_voi.pdf")); plt.close(fig)

# ---------------------------------------------------------------- fig_exact
ex = pd.read_csv(os.path.join(a.results, "exact.csv"))
fig, ax = plt.subplots(figsize=(4.2, 2.6))
sz = {"V20": 20, "V30": 30, "V40": 40, "V50": 50}
rng = np.random.default_rng(0)
for meth, tcol, scol, ok, lim, col, dx, lab in (("DP", "dp_time", "dp_status", "optimal", 1800, BLUE, -1.3, "DP (AO)"),
                                                 ("CP", "ao_time", "ao_status", "OPTIMAL", 600, ORANGE, 1.3, "CP-SAT (AO)")):
    for pt, n in sz.items():
        e = ex[ex.point == pt]; x = n + dx + rng.uniform(-0.6, 0.6, len(e))
        pr = (e[scol] == ok).values; t = np.maximum(e[tcol].values, 1e-3)
        ax.scatter(x[pr], t[pr], s=14, color=col, edgecolor="white", linewidth=0.5, zorder=3, label=lab if pt == "V20" else None)
        ax.scatter(x[~pr], np.full((~pr).sum(), lim), s=14, facecolor="white", edgecolor=col, linewidth=1.0, zorder=3)
    ax.axhline(lim, color=col, lw=0.6, ls=(0, (3, 3)))
    ax.text(16.6, lim * 1.15, f"{lab}: {lim} s limit", color=INK2, va="bottom", fontsize=6.5)
ax.set_yscale("log"); ax.set_xticks(list(sz.values())); ax.set_xticklabels(list(sz.keys())); ax.set_xlim(16, 54); ax.set_ylim(5e-4, 8e3)
ax.set_ylabel("Time per instance (s)"); ax.set_xlabel("Size (vessels, berths = vessels/10)")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_exact.pdf")); plt.close(fig)

# ---------------------------------------------------------------- fig_train
tc = pd.read_csv(os.path.join(a.results, "train_curves.csv"), header=None, names=["tag", "m", "seed", "ep", "val", "eps", "time"])
tc = tc[tc.tag == "dqn-ep4000"]
fig, axs = plt.subplots(1, 4, figsize=(6.3, 1.9))
for ax, m in zip(axs, (2, 3, 4, 5)):
    for sd, g in tc[tc.m == m].groupby("seed"):
        g = g.sort_values("ep"); ax.plot(g.ep, g.val, color=BLUE, lw=1.0, alpha=0.55)
        b = g.loc[g.val.idxmin()]; ax.plot(b.ep, b.val, "o", ms=4.5, color=BLUE, mec="white", mew=0.8, zorder=3)
    ax.set_title(f"{m} berths, {10 * m} vessels", fontsize=8, color=INK, pad=3)
    ax.set_xticks([0, 2000, 4000]); ax.set_xticklabels(["0", "2k", "4k"]); ax.set_xlabel("Episode")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axs[0].set_ylabel("Validation turnaround (h)")
fig.tight_layout(w_pad=0.6); fig.savefig(os.path.join(a.out, "fig_train.pdf")); plt.close(fig)

# ---------------------------------------------------------------- fig_sens
FAC2 = [(r"$\mu_w$ (m/s)", ["mu5", "V30", "mu15", "mu20"], [5, 10, 15, 20]),
        (r"$\sigma_w$", ["sig1.5", "V30", "sig5.5", "sig8.0"], [1.5, 3.5, 5.5, 8.0]),
        (r"load $\lambda$", ["load0.6", "V30", "load0.95"], [0.6, 0.8, 0.95])]
PS = ["Static", "FirstFree", "MyopicC", "LowExposure", "DQN", "LookDP-R"]
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
for ax, (lab, pts, vals) in zip(axs, FAC2):
    for m in PS:
        y = []
        for pt in pts:
            za = exact_inst(pt); zr = inst(pt, "RollDP")[za.index]; zp = inst(pt, m)
            j = zp.index.intersection(za.index); y.append((zp[j].mean() - zr[j].mean()) / zr[j].mean() * 100)
        ax.plot(vals, y, color=COL[m], lw=1.6, marker=MRK[m], ms=6 if m == "LookDP-R" else 4.5, mec="white", mew=0.6, label=m)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xlabel(lab); ax.set_xticks(vals)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
axs[0].set_ylabel(r"Loss vs \textsc{RollDP} (%)".replace(r"\textsc{RollDP}", "RollDP"))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8); fig.savefig(os.path.join(a.out, "fig_sens.pdf")); plt.close(fig)
# ---------------------------------------------------------------- fig_scale (extension, 25 Oct 2026)
# (a) mean seconds per decision per main size; (b) mean turnaround relative to the best dispatching rule on all 20 test
# instances, mean with 95% t-interval of the paired per-instance difference.
def xv(d, k): return d.extra.fillna("").astype(str).str.extract(k + r"=([-0-9.eE+naninf]+)")[0].astype(float)
NS = [20, 30, 40, 50]; PTS = ["V20", "V30", "V40", "V50"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.6))
for m in ("RollDP", "LookDP-0", "LookDP-R", "DQN"):
    y = []
    for pt, n in zip(PTS, NS):
        d = df[(df.point == pt) & (df.method == m)]; y.append((xv(d, "dec_s") / n).mean() if m != "LookDP-0" else (d.time_s / n).mean())
    a1.plot(NS, y, color=COL[m], lw=1.6, ls=LS.get(m, "-"), marker=MRK[m], ms=6 if m == "LookDP-R" else 4.5, mec="white", mew=0.6, label=m)
a1.axhline(60, color=INK2, lw=0.6, ls=(0, (3, 3))); a1.text(20.5, 75, "60 s limit per decision", color=INK2, fontsize=6.5)
a1.set_yscale("log"); a1.set_ylim(1e-4, 300); a1.set_xticks(NS); a1.set_xticklabels(PTS); a1.set_ylabel("Seconds per decision")
a1.set_title("(a) decision time", fontsize=8.5, color=INK, pad=4)
def best_rule(pt): return min(("FirstFree", "MyopicC", "LowExposure"), key=lambda m: inst(pt, m).mean())
for k, m in enumerate(("RollDP", "LookDP-0", "LookDP-R", "LookDP-RV", "DQN", "UBE-0.95")):
    mu, lo, hi = [], [], []
    for pt in PTS:
        g, h, _ = rel_ci(inst(pt, m), inst(pt, best_rule(pt))); mu.append(g); lo.append(g - h); hi.append(g + h)
    x = np.array(NS) + (k - 2.5) * 0.8
    a2.vlines(x, lo, hi, color=COL[m], lw=1.0, alpha=0.7)
    a2.plot(x, mu, color=COL[m], lw=1.4, ls=LS.get(m, "-"), marker=MRK[m], ms=6 if m == "LookDP-R" else 4.5, mec="white", mew=0.6, label="UBE" if m == "UBE-0.95" else m)
a2.axhline(0, color=INK2, lw=0.6); a2.set_xticks(NS); a2.set_xticklabels(PTS)
a2.set_ylabel("Turnaround vs best rule (%)"); a2.set_title("(b) all 20 test instances", fontsize=8.5, color=INK, pad=4)
for ax in (a1, a2):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = a2.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.5); fig.savefig(os.path.join(a.out, "fig_scale.pdf")); plt.close(fig)
print("wrote fig_decomp.pdf, fig_voi.pdf, fig_exact.pdf, fig_train.pdf, fig_sens.pdf, fig_scale.pdf to", a.out)
