"""5_lookahead.py — the rolling DP with bounded lookahead and learned cost-to-go (DESIGN.md §9, post hoc, 23 Oct 2026).

  --train  --berths m --seed s [--kmax 6]     fit the cost-to-go network g for m berths (training pool, reference
                                              conditions, n = 10 m, as the DQN); early stopping on the validation pool
  --pilot  --point P --K 2,3,4 [--seeds valid] the three variants on the VALIDATION pool (never the test pool) to fix K
Writes results/vnet_B{m}_seed{s}.pkl, results/look_pilot.csv. Test-pool runs go through 3_benchmark.py --methods LookDP-*."""
import argparse, csv, pickle, time
import numpy as np
from common import POINTS, size, inst_of, wind_of, wind_seed, TRAIN_SEEDS, VALID_SEEDS, RESULTS
from lsps_core import online as O
from lsps_core.lookahead import LookDP, VNet, sample_states, n_vfeatures

ap = argparse.ArgumentParser()
ap.add_argument("--train", action="store_true"); ap.add_argument("--pilot", action="store_true")
ap.add_argument("--berths", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kmax", type=int, default=6); ap.add_argument("--reals", type=int, default=2)
ap.add_argument("--point", default="V30"); ap.add_argument("--K", default="2,3,4,5")
ap.add_argument("--variants", default="0,R,V,RV"); ap.add_argument("--n-valid", type=int, default=10)
a = ap.parse_args()

def vnet_path(m, s): return RESULTS / f"vnet_B{m}_seed{s}.pkl"

if a.train:
    m = a.berths; cfg = size(10 * m, m); t0 = time.time(); Xs, ys, yrs = [], [], []
    for k, s in enumerate(TRAIN_SEEDS):
        inst = inst_of(cfg, s)
        for r in range(a.reals):
            X, y, yr = sample_states(inst, wind_of(cfg, inst, wind_seed(s, 50 + r)), cfg, a.kmax); Xs.append(X); ys.append(y); yrs.append(yr)
        if (k + 1) % 50 == 0: print(f"  B{m}: {k + 1}/{len(TRAIN_SEEDS)} training instances, {sum(len(y) for y in ys)} samples, {time.time() - t0:.0f}s", flush=True)
    Xv, yv, yrv = [], [], []
    for s in VALID_SEEDS:
        inst = inst_of(cfg, s); X, y, yr = sample_states(inst, wind_of(cfg, inst, wind_seed(s, 0)), cfg, a.kmax); Xv.append(X); yv.append(y); yrv.append(yr)
    X, Xv = np.concatenate(Xs), np.concatenate(Xv)
    net = VNet(n_vfeatures(m), seed=a.seed); mse = net.fit(X, np.concatenate(ys), Xv, np.concatenate(yv))
    rnet = VNet(n_vfeatures(m), seed=a.seed); rmse = rnet.fit(X, np.concatenate(yrs), Xv, np.concatenate(yrv))
    pickle.dump(dict(net=net, rnet=rnet, m=m, seed=a.seed, kmax=a.kmax, val_mse=mse, val_mse_res=rmse,
                     train_time_s=time.time() - t0, samples=int(len(X))), open(vnet_path(m, a.seed), "wb"))
    with open(RESULTS / "look_train.csv", "a") as f: f.write(f"B{m},{a.seed},{mse:.6f},{rmse:.6f},{len(X)},{time.time() - t0:.0f}\n")
    print(f"saved {vnet_path(m, a.seed)}  val MSE {mse:.5f} (V), {rmse:.5f} (RV)  {time.time() - t0:.0f}s")

if a.pilot:
    pt = a.point; cfg = POINTS[pt]; m = cfg["n_berths"]
    out = RESULTS / "look_pilot.csv"; new = not out.exists()
    P = pickle.load(open(vnet_path(m, 0), "rb")) if vnet_path(m, 0).exists() else None
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["point", "method", "K", "instance", "realization", "Z", "time_s", "labels_per_dec"])
        for s in VALID_SEEDS[:a.n_valid]:
            inst = inst_of(pt, s); real = wind_of(pt, inst, wind_seed(s, 0))
            t = time.perf_counter(); Z = O.run_online(inst, real, cfg, O.pol_myopic)["Z"]
            w.writerow([pt, "MyopicC", 0, s, 0, Z, f"{time.perf_counter() - t:.3f}", 0]); f.flush()
            for K in [int(k) for k in a.K.split(",")]:
                for var in a.variants.split(","):
                    if var in ("V", "RV") and P is None: continue
                    pol = LookDP(K, {"0": "zero", "R": "rollout", "V": "learned", "RV": "residual"}[var],
                                 None if P is None else (P["rnet"] if var == "RV" else P["net"]))
                    t = time.perf_counter(); Z = O.run_online(inst, real, cfg, pol)["Z"]
                    w.writerow([pt, f"LookDP-{var}", K, s, 0, Z, f"{time.perf_counter() - t:.3f}", f"{pol.labels / inst.n:.0f}"]); f.flush()
            print(f"  {pt} validation instance {s}: done", flush=True)
