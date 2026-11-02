"""dry_run.py — times every method on this machine and estimates the campaign (core-hours).
Writes results/dry_run_<host>.txt. Usage: python dry_run.py"""
import socket, time, statistics as st
from common import *
from lsps_core.dp import solve_ao
from lsps_core.exact import solve_cpsat
from lsps_core.learners import train_dqn, train_ql
import lsps_core.online as O
out = []; P = lambda s: (print(s, flush=True), out.append(s))
P(f"host {socket.gethostname()}")
est = {}
for pt in MAIN:
    cfg = POINTS[pt]; ts = {}
    for s in TEST_SEEDS[:2]:
        inst = inst_of(pt, s); real = wind_of(pt, inst, wind_seed(s, 0))
        t = time.perf_counter(); d = solve_ao(inst, real, time_cap=600); ts.setdefault("PI-AO", []).append(time.perf_counter() - t)
        t = time.perf_counter(); solve_cpsat(inst, real, "free", time_limit=60, workers=1, hint_plan=d.plan); ts.setdefault("PI-FREE(60s cap)", []).append(time.perf_counter() - t)
        for k, p in O.RULES.items():
            t = time.perf_counter(); O.run_online(inst, real, cfg, p); ts.setdefault(k, []).append(time.perf_counter() - t)
        t = time.perf_counter(); O.run_online(inst, real, cfg, O.Static(inst, real, cfg)); ts.setdefault("Static", []).append(time.perf_counter() - t)
        t = time.perf_counter(); O.run_online(inst, real, cfg, O.RollDP(time_cap=60)); ts.setdefault("RollDP", []).append(time.perf_counter() - t)
        P(f"{pt} seed {s}: DP {d.status} labels {d.labels} | " + " ".join(f"{k} {v[-1]:.1f}s" for k, v in ts.items()))
    est[pt] = {k: st.mean(v) for k, v in ts.items()}
    # DQN training speed (20 episodes)
    mk = lambda s: inst_of(cfg, s); mw = lambda s, ws, inst: wind_of(cfg, inst, ws)
    valid = [(mk(s), mw(s, wind_seed(s, 0), mk(s))) for s in VALID_SEEDS[:2]]
    t = time.perf_counter(); train_dqn(0, cfg, mk, mw, TRAIN_SEEDS, valid, episodes=20, ckpt_every=10, log=lambda *x, **k: None)
    est[pt]["DQN episode"] = (time.perf_counter() - t) / 20
    P(f"{pt}: DQN {est[pt]['DQN episode']:.3f}s/episode -> 4000 episodes = {est[pt]['DQN episode']*4000/3600:.2f} h per seed")
cfg = POINTS["V20"]; mk = lambda s: inst_of(cfg, s); mw = lambda s, ws, inst: wind_of(cfg, inst, ws)
t = time.perf_counter(); r = train_ql(0, cfg, mk, mw, TRAIN_SEEDS, 500); qls = (time.perf_counter() - t) / 500
P(f"QL: {qls*1000:.1f} ms/episode -> 200,000 episodes = {qls*2e5/3600:.2f} h per seed")
runs = 60
tot = 0.0
for pt in MAIN:
    e = est[pt]; h = runs * (e["PI-AO"] + e["PI-FREE(60s cap)"] * 10 + sum(e[k] for k in O.RULES) + e["Static"] + e["RollDP"] + 5 * e["MyopicC"]) / 3600
    h += 5 * e["DQN episode"] * 4000 / 3600; tot += h
    P(f"{pt}: ~{h:.1f} core-h (PI-FREE at a 600 s limit, 5 DQN trainings and evaluations)")
sens = len(SENS) * runs * (sum(est["V30"][k] for k in O.RULES) + est["V30"]["Static"] + est["V30"]["RollDP"] + est["V30"]["PI-AO"] + 5 * est["V30"]["MyopicC"]) / 3600
P(f"sensitivity ({len(SENS)} points, no PI-FREE): ~{sens:.1f} core-h; QL 5 seeds ~{5*qls*2e5/3600:.1f} core-h")
P(f"TOTAL ~{tot + sens + 5*qls*2e5/3600:.0f} core-hours (+ PI-FREE time limits)")
(RESULTS / f"dry_run_{socket.gethostname()}.txt").write_text("\n".join(out) + "\n")
