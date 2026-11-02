"""6_correction_r2.py — post-hoc diagnostic of the learned correction of LookDP-RV (added 25 Oct 2026, after the test runs).
For each berth count m and network seed, the share of variance (R^2) of the two learning targets of 5_lookahead.py
that the trained networks explain on the 10 validation instances (realization 0, the early-stopping set): y (realized
turnaround per remaining vessel of MyopicC; network 'net', LookDP-V) and yr (realized minus forecast-rollout turnaround
per remaining vessel; network 'rnet', LookDP-RV). Also the mean and standard deviation of yr (days). Descriptive only.
Usage: python3 6_correction_r2.py [--results results]   -> results/correction_r2.csv"""
import argparse, csv, pickle
import numpy as np
from common import size, inst_of, wind_of, wind_seed, VALID_SEEDS
from lsps_core.lookahead import sample_states

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()
rows = []
for m in (2, 3, 4, 5):
    cfg = size(10 * m, m); X, y, yr = [], [], []
    for s in VALID_SEEDS:
        inst = inst_of(cfg, s); x_, y_, r_ = sample_states(inst, wind_of(cfg, inst, wind_seed(s, 0)), cfg, 6)
        X.append(x_); y.append(y_); yr.append(r_)
    X, y, yr = np.concatenate(X), np.concatenate(y), np.concatenate(yr)
    for sd in range(5):
        P = pickle.load(open(f"{a.results}/vnet_B{m}_seed{sd}.pkl", "rb"))
        r2v = 1 - np.mean((P["net"].predict(X) - y) ** 2) / np.var(y)
        r2r = 1 - np.mean((P["rnet"].predict(X) - yr) ** 2) / np.var(yr)
        rows.append([m, sd, len(y), f"{r2v:.4f}", f"{r2r:.4f}", f"{yr.mean():.4f}", f"{yr.std():.4f}"])
        print(*rows[-1], flush=True)
with open(f"{a.results}/correction_r2.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["m", "seed", "samples", "r2_V", "r2_RV", "yr_mean_days", "yr_sd_days"]); w.writerows(rows)
