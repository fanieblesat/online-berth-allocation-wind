"""2_train.py — DQN (per number of berths, five seeds; checkpointed and resumable) and tabular QL (V20 only).
Training instances: the training pool at the reference wind and load, |V| = 10 |B| (the main-grid size for that |B|).
Usage: python 2_train.py --dqn --berths 3 --seed 0 [--episodes 4000]
       python 2_train.py --ql --seed 0 [--episodes 200000]"""
import argparse, pickle
import torch
from common import *
from lsps_core.learners import train_dqn, train_ql
ap = argparse.ArgumentParser(); ap.add_argument("--dqn", action="store_true"); ap.add_argument("--ql", action="store_true")
ap.add_argument("--berths", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=None); ap.add_argument("--threads", type=int, default=1); a = ap.parse_args()
torch.set_num_threads(a.threads)
if a.dqn:
    cfg = size(10 * a.berths, a.berths); ep = a.episodes or 4000
    pth = policy_path(a.berths, a.seed, ep)
    if pth.exists(): print(f"{pth} exists"); raise SystemExit
    mk = lambda s: inst_of(cfg, s); mw = lambda s, ws, inst: wind_of(cfg, inst, ws)
    valid = [(mk(s), mw(s, wind_seed(s, 0), mk(s))) for s in VALID_SEEDS]
    pol = train_dqn(a.seed, cfg, mk, mw, TRAIN_SEEDS, valid, episodes=ep, ckpt_every=max(10, ep // 20),
                    resume_path=str(RESULTS / f"ckpt_dqn_B{a.berths}_seed{a.seed}_ep{ep}.pt"),
                    probe_csv=str(RESULTS / "train_curves.csv"), tag=f"dqn-ep{ep}")
    pickle.dump(pol, open(pth, "wb")); print("saved", pth, "best", pol["best_val"], "@", pol["best_episode"])
if a.ql:
    cfg = size(20, 2); ep = a.episodes or 200_000
    pth = RESULTS / f"policy_ql_B2_seed{a.seed}.pkl"
    if pth.exists(): print(f"{pth} exists"); raise SystemExit
    mk = lambda s: inst_of(cfg, s); mw = lambda s, ws, inst: wind_of(cfg, inst, ws)
    pol = train_ql(a.seed, cfg, mk, mw, TRAIN_SEEDS, ep)
    pickle.dump(pol, open(pth, "wb")); print("saved", pth, "states", pol["states"], f"{pol['train_time_s']:.0f}s")
