"""3_benchmark.py — every method on one point of the campaign (test pool x 3 realizations, common random numbers).
Methods: PI (one DP solve logged as PI-AO, then CP-SAT FREE hinted with it, logged as PI-FREE with its bound and the
no-wait lower bound), PI-AO (DP only), Static, RollDP, FirstFree, MyopicC, LowExposure,
DQN (seeds 0-4; the policy for that number of berths), QL (V20 only). Resumable (results/runs.csv).
Post-hoc methods (DESIGN.md §9, 23 Oct 2026): RollDP-T (RollDP with ties to the less exposed berth), LookDP-0 and LookDP-R
(lookahead DP of --look vessels, zero / forecast-rollout terminal cost), LookDP-RV (rollout + learned correction,
network seeds 0-4 for that number of berths, results/vnet_B{m}_seed{s}.pkl).
Literature baseline (DESIGN.md §10, 25 Oct 2026): UBE-0.95 / UBE-0.5 = Umang, Bierlaire & Erera (2017) real-time
recovery with planning quantile rho; baseline schedule limit --static-cap (600 s), re-optimization limit --roll-cap (60 s).
Usage: python 3_benchmark.py --point V30 --methods PI-AO,Static,RollDP,FirstFree,MyopicC,LowExposure,DQN
       python 3_benchmark.py --point V50 --methods PI --seeds 1003 --cp-limit 600 --workers 1
--seeds restricts the test pool (a single seed, a range 1000-1004, or a comma list) so that one point can be split
into independent jobs; all jobs append to the same results/runs.csv."""
import argparse, pickle, time
import torch
from common import *
from lsps_core.dp import solve_ao
from lsps_core.model import completion, evaluate_ao
from lsps_core.exact import solve_cpsat
from lsps_core.learners import dqn_policy, ql_policy
import lsps_core.online as O
ap = argparse.ArgumentParser(); ap.add_argument("--point", required=True); ap.add_argument("--methods", required=True)
ap.add_argument("--dp-cap", type=float, default=1800); ap.add_argument("--roll-cap", type=float, default=60)
ap.add_argument("--cp-limit", type=float, default=600); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--episodes", type=int, default=4000); ap.add_argument("--fast", action="store_true")
ap.add_argument("--seeds", default=None); ap.add_argument("--static-cap", type=float, default=600)
ap.add_argument("--look", type=int, default=None, help="lookahead K of the LookDP methods (DESIGN.md §9)"); a = ap.parse_args()
torch.set_num_threads(1)
cfg = POINTS[a.point]; m = int(cfg["n_berths"])
def parse_seeds(x):
    if x is None: return TEST_SEEDS
    out = []
    for part in x.split(","):
        lo, _, hi = part.partition("-"); out += list(range(int(lo), int(hi or lo) + 1))
    assert all(s in TEST_SEEDS for s in out), "seeds outside the test pool"
    return out
seeds = TEST_SEEDS[:2] if a.fast else parse_seeds(a.seeds); reals = range(1 if a.fast else N_REAL)

real_seed_of = {}                          # id(realized wind) -> its seed (UBE draws its sample paths per realization)
def each(method, seed, fn):
    done = done_keys(a.point, method, seed)
    for s in seeds:
        inst = inst_of(a.point, s)
        for r in reals:
            if (s, r) in done: continue
            real = wind_of(a.point, inst, wind_seed(s, r)); real_seed_of[id(real)] = wind_seed(s, r)
            t0 = time.perf_counter(); Z, st, extra = fn(inst, real); dt = time.perf_counter() - t0
            log_run(a.point, method, seed, s, r, Z, st, dt, extra)
    print(f"  {a.point} {method} seed {seed} (instances {seeds[0]}-{seeds[-1]}): done", flush=True)

def greedy_plan(inst, real):
    """Each vessel, in arrival order, to the berth that releases it earliest (the DP's upper-bound assignment)."""
    F = [1] * inst.m; plan = []
    for v in range(inst.n):
        b = min(range(inst.m), key=lambda b: completion(real, b, max(F[b], inst.A[v]), inst.h[v], inst.q[v])[1])
        F[b] = completion(real, b, max(F[b], inst.A[v]), inst.h[v], inst.q[v])[1]; plan.append(b)
    return plan

def nowait_lb(inst, real):
    return float(sum(min(completion(real, b, inst.A[u], inst.h[u], inst.q[u])[1] - inst.A[u] for b in range(inst.m))
                     for u in range(inst.n)))

for meth in a.methods.split(","):
    if meth == "PI":
        dA, dF = done_keys(a.point, "PI-AO", 0), done_keys(a.point, "PI-FREE", 0)
        for s in seeds:
            inst = inst_of(a.point, s)
            for r in reals:
                if (s, r) in dA and (s, r) in dF: continue
                real = wind_of(a.point, inst, wind_seed(s, r))
                t0 = time.perf_counter(); d = solve_ao(inst, real, time_cap=a.dp_cap); dt = time.perf_counter() - t0
                if (s, r) not in dA: log_run(a.point, "PI-AO", 0, s, r, d.cost, d.status, dt, f"labels={d.labels}")
                if (s, r) not in dF:
                    hint = d.plan if d.plan is not None else greedy_plan(inst, real)   # DP capped: greedy AO hint
                    c = solve_cpsat(inst, real, "free", time_limit=a.cp_limit, workers=a.workers, hint_plan=hint)
                    if c["obj"] != c["obj"]:
                        # no incumbent within the limit: the hint's AO schedule is FREE-feasible (status marks it)
                        c["obj"], c["status"] = evaluate_ao(inst, real, hint)["Z"], c["status"] + "_AOHINT"
                    log_run(a.point, "PI-FREE", 0, s, r, c["obj"], c["status"], c["time_s"],
                            f"bound={c['bound']};nowait_lb={nowait_lb(inst, real)};at_horizon={int(bool(c.get('at_horizon', 0)))}")
        print(f"  {a.point} PI seeds {seeds[0]}-{seeds[-1]}: done", flush=True)
    elif meth == "PI-AO":
        def f(inst, real):
            d = solve_ao(inst, real, time_cap=a.dp_cap); return d.cost, d.status, f"labels={d.labels}"
        each(meth, 0, f)
    elif meth == "PI-FREE":
        def f(inst, real):
            d = solve_ao(inst, real, time_cap=a.dp_cap)
            c = solve_cpsat(inst, real, "free", time_limit=a.cp_limit, workers=a.workers, hint_plan=d.plan)
            return c["obj"], c["status"], f"bound={c['bound']};nowait_lb={nowait_lb(inst, real)};at_horizon={int(bool(c.get('at_horizon', 0)))}"
        each(meth, 0, f)
    elif meth in O.RULES:
        pol = O.RULES[meth]
        each(meth, 0, lambda inst, real: (O.run_online(inst, real, cfg, pol)["Z"], "ok", ""))
    elif meth == "Static":
        def f(inst, real):
            p = O.Static(inst, real, cfg, time_cap=a.static_cap); return O.run_online(inst, real, cfg, p)["Z"], "ok", f"fallback={p.fallback}"
        each(meth, 0, f)
    elif meth == "RollDP":
        def f(inst, real):
            p = O.RollDP(time_cap=a.roll_cap); r = O.run_online(inst, real, cfg, p)
            return r["Z"], "ok", f"fallbacks={p.fallbacks};dec_s={r['dec_time_s']:.2f}"
        # protocol cap (60 s) logs as RollDP; any other cap is a separate, declared post-hoc method (CHANGELOG 22 Oct)
        each(meth if a.roll_cap == 60 else f"RollDP-cap{int(a.roll_cap)}", 0, f)
    elif meth == "RollDP-T":
        def f(inst, real):
            p = O.RollDP(time_cap=a.roll_cap, tie_exposure=True); r = O.run_online(inst, real, cfg, p)
            return r["Z"], "ok", f"fallbacks={p.fallbacks};dec_s={r['dec_time_s']:.2f}"
        assert a.roll_cap == 60, "RollDP-T is defined with the protocol cap of 60 s"
        each("RollDP-T", 0, f)
    elif meth in ("LookDP-0", "LookDP-R", "LookDP-RV"):
        from lsps_core.lookahead import LookDP
        assert a.look, "--look K is required (DESIGN.md §9)"
        term = {"LookDP-0": "zero", "LookDP-R": "rollout", "LookDP-RV": "residual"}[meth]
        for sd in (TRAIN_RNG_SEEDS if meth == "LookDP-RV" else [0]):
            net = None
            if meth == "LookDP-RV":
                pth = RESULTS / f"vnet_B{m}_seed{sd}.pkl"
                if not pth.exists(): print(f"missing {pth}"); continue
                net = pickle.load(open(pth, "rb"))["rnet"]
            def f(inst, real, net=net):
                p = LookDP(a.look, term, net); r = O.run_online(inst, real, cfg, p)
                return r["Z"], "ok", f"K={a.look};labels_per_dec={p.labels / inst.n:.0f};dec_s={r['dec_time_s']:.2f}"
            each(meth, sd, f)
    elif meth in ("UBE-0.95", "UBE-0.5"):
        # literature baseline (DESIGN.md §10): Umang, Bierlaire & Erera (2017), real-time recovery, adapted to wind
        from lsps_core.ube import run_ube
        rho = float(meth.split("-")[1])
        def f(inst, real, rho=rho):
            r_idx = next(k for k in range(N_REAL) if wind_seed(inst.seed, k) == real_seed_of[id(real)])
            r = run_ube(inst, real, cfg, rho=rho, S=100, cap0=a.static_cap, cap=a.roll_cap, real_idx=r_idx)
            return r["Z"], "ok", f"rho={rho};events={r['events']};hint_used={r['hint_used']};beyond={r['beyond']}"
        each(meth, 0, f)
    elif meth == "DQN":
        for sd in TRAIN_RNG_SEEDS:
            pth = policy_path(m, sd, a.episodes)
            if not pth.exists(): print(f"missing {pth}"); continue
            pol = dqn_policy(pickle.load(open(pth, "rb")))
            name = "DQN" if a.episodes == 4000 else f"DQN-ep{a.episodes}"
            def f(inst, real, pol=pol):
                r = O.run_online(inst, real, cfg, pol); return r["Z"], "ok", f"dec_s={r['dec_time_s']:.3f}"
            each(name, sd, f)
    elif meth == "QL":
        for sd in TRAIN_RNG_SEEDS:
            pth = RESULTS / f"policy_ql_B{m}_seed{sd}.pkl"
            if not pth.exists(): print(f"missing {pth}"); continue
            pol = ql_policy(pickle.load(open(pth, "rb")))
            def f(inst, real, pol=pol):
                pol.unseen = 0; r = O.run_online(inst, real, cfg, pol)
                return r["Z"], "ok", f"unseen={pol.unseen};decisions={inst.n}"
            each("QL", sd, f)
    else:
        raise SystemExit(f"unknown method {meth}")
