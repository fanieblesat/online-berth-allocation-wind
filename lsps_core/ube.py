"""ube.py — literature baseline (DESIGN.md §10): the optimization-based real-time recovery of Umang, Bierlaire & Erera
(2017, J. Scheduling 20:67-83), adapted to time-dependent wind.

Baseline schedule at period 1 = free-sequencing optimum on planning values; planning value of vessel v at berth b when
ready in period r = rho-quantile of its completion over S OU sample paths conditioned on the wind realized up to the
event period; re-optimization of every vessel not yet berthed (free sequencing) at every completion; berthed vessels
block their berth until the planning value of their completion; a vessel berths at its planned berth when the berth is
free, it has arrived and its planned berthing period is reached. Objective sum (C_v - A_v). Solver CP-SAT (one worker).
"""
from __future__ import annotations
import time
import numpy as np
from ortools.sat.python import cp_model
from .model import Wind, completion, RHO


# ---------------------------------------------------------------- sample paths and vectorized completion
def sample_paths(real: Wind, t: int, cfg: dict, beta, S: int, seed_key):
    """S OU paths equal to the realized wind in periods 1..t and simulated afterwards (Eq. 1), as Wind objects."""
    mu_w, sigma_w, theta_w = float(cfg["mu_w"]), float(cfg["sigma_w"]), float(cfg["theta_w"])
    rng = np.random.default_rng(seed_key)
    L = real.L; W = np.empty((S, L)); W[:, :t] = np.asarray(real.W[:t])[None, :]
    noise = rng.standard_normal((S, L))
    for k in range(t, L):
        W[:, k] = np.maximum(0.0, W[:, k - 1] + theta_w * (mu_w - W[:, k - 1]) + sigma_w * noise[:, k])
    return [Wind(W[s].tolist(), beta, real.alpha) for s in range(S)]


def _berthable_index(w: Wind, b: int):
    return np.flatnonzero(w.S[b, 1:w.L + 1] == 0) + 1           # berthable periods 1..L, ascending


def completion_vec(w: Wind, b: int, r, need):
    """Vectorized model.completion: r (array of ready periods), need (array of work in crane-unit periods = h/(q RHO)).
    Returns C (int array); L+1 marks 'beyond the trajectory'."""
    r = np.maximum(1, np.asarray(r, dtype=np.int64)); need = np.asarray(need, dtype=float)
    idx = _berthable_index(w, b); L = w.L
    k = np.searchsorted(idx, r, side="left")
    tau = np.where(k < len(idx), idx[np.minimum(k, len(idx) - 1)], L + 1)
    ok = tau <= L
    tau_c = np.minimum(tau, L)
    s = tau_c + w.phi[b, tau_c].astype(np.int64)
    cum = w.cum[b]
    target = cum[s] + need - 1e-9
    t_end = np.searchsorted(cum, target, side="left")
    C = np.maximum(t_end, s)
    C = np.where(t_end > L + 1, L + 1, C)
    return np.where(ok, C, L + 1).astype(np.int64)


def remaining_vec(w: Wind, b: int, start, need):
    """Completion of a vessel already berthed whose remaining work (crane-unit periods) is processed from period start."""
    cum = w.cum[b]; start = int(start)
    t_end = int(np.searchsorted(cum, cum[start] + need - 1e-9, side="left"))
    return max(t_end, start) if t_end <= w.L + 1 else w.L + 1


def quantile_tables(inst, samples, U, lo, Tmax, rho):
    """Planning tables: for v in U and berth b, ready periods r = lo[v][b]..Tmax and the rho-quantile (inverted CDF) of
    C_b(v, r) over the samples. Returns {(v, b): (starts list, comp list)} with completions <= Tmax + 1 only."""
    out = {}
    for b in range(inst.m):
        rs, needs, keys = [], [], []
        for v in U:
            r0 = lo[v][b]
            if r0 > Tmax: continue
            rr = np.arange(r0, Tmax + 1); rs.append(rr); needs.append(np.full(len(rr), inst.h[v] / (inst.q[v] * RHO))); keys.append((v, len(rr)))
        if not rs: continue
        R = np.concatenate(rs); N = np.concatenate(needs)
        Cs = np.stack([completion_vec(w, b, R, N) for w in samples])          # (S, total)
        Q = np.quantile(Cs, rho, axis=0, method="inverted_cdf").astype(np.int64)
        pos = 0
        for v, ln in keys:
            st = R[pos:pos + ln]; cq = Q[pos:pos + ln]; pos += ln
            keep = cq <= Tmax + 1
            if keep.any(): out[v, b] = (st[keep].tolist(), cq[keep].tolist())
    return out


# ---------------------------------------------------------------- the re-optimization model
def solve_free_tables(inst, U, tabs, hint, time_limit, workers=1):
    """Free-sequencing model on planning tables. hint: {v: (b, r)} feasible complete plan. Returns (plan, status, obj)."""
    m = inst.m; A = inst.A
    mdl = cp_model.CpModel(); x = {}; s = {}; e = {}; iv = {}; kk = {}; ends = {}; dz = {}
    for v in U:
        lits = []; Tm = max(c for (vv, b), (_, cs) in tabs.items() if vv == v for c in cs)
        ev = mdl.NewIntVar(A[v], Tm, f"e{v}")
        for b in range(m):
            if (v, b) not in tabs: continue
            st, cq = tabs[v, b]
            xb = mdl.NewBoolVar(f"x{v}_{b}"); k = mdl.NewIntVar(0, len(st) - 1, f"k{v}_{b}")
            sv = mdl.NewIntVar(min(st), max(st), f"s{v}_{b}"); evb = mdl.NewIntVar(min(cq), max(cq), f"e{v}_{b}")
            mdl.AddElement(k, st, sv); mdl.AddElement(k, cq, evb)
            size = mdl.NewIntVar(0, max(cq), f"d{v}_{b}"); dz[v, b] = size
            iv[v, b] = mdl.NewOptionalIntervalVar(sv, size, evb, xb, f"i{v}_{b}")
            mdl.Add(ev == evb).OnlyEnforceIf(xb)
            x[v, b], s[v, b], e[v, b], kk[v, b] = xb, sv, evb, k; lits.append(xb)
        mdl.AddExactlyOne(lits); ends[v] = ev
    for b in range(m):
        mdl.AddNoOverlap([iv[v, b] for v in U if (v, b) in iv])
    obj = sum(ends[v] - A[v] for v in U); mdl.Minimize(obj)
    hval = 0
    for v in U:
        b, r = hint[v]; st, cq = tabs[v, b]; i = st.index(r); c = cq[i]; hval += c - A[v]
        for bb in range(m):
            if (v, bb) in x: mdl.AddHint(x[v, bb], 1 if bb == b else 0)
        mdl.AddHint(kk[v, b], i); mdl.AddHint(s[v, b], r); mdl.AddHint(e[v, b], c); mdl.AddHint(ends[v], c)
        mdl.AddHint(dz[v, b], c - r)
        for bb in range(m):                   # absent alternatives: hint them at their first table entry
            if bb != b and (v, bb) in x:
                st2, cq2 = tabs[v, bb]; mdl.AddHint(kk[v, bb], 0); mdl.AddHint(s[v, bb], st2[0]); mdl.AddHint(e[v, bb], cq2[0]); mdl.AddHint(dz[v, bb], cq2[0] - st2[0])
    mdl.Add(obj <= hval)
    sol = cp_model.CpSolver(); sol.parameters.num_workers = int(workers)
    if time_limit is not None: sol.parameters.max_time_in_seconds = float(time_limit)
    st_ = sol.Solve(mdl)
    if st_ not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {v: hint[v] for v in U}, "HINT", float(hval)
    plan = {}
    for v in U:
        b = next(bb for bb in range(m) if (v, bb) in x and sol.Value(x[v, bb]))
        plan[v] = (b, sol.Value(s[v, b]))
    return plan, sol.StatusName(st_), float(sol.ObjectiveValue())


def replay(inst, U, order_plan, tabs, lo):
    """Feasible plan on the tables from a berth assignment and per-berth order: r = earliest table start >= the release
    of the previous vessel at that berth. order_plan: {v: (b, key)}; returns {v: (b, r)} or None if infeasible."""
    plan = {}; rel = {}
    for v in sorted(U, key=lambda v: (order_plan[v][1], v)):
        b = order_plan[v][0]
        if (v, b) not in tabs: return None
        st, cq = tabs[v, b]; need = max(rel.get(b, 0), st[0])
        i = int(np.searchsorted(st, need))
        if i >= len(st): return None
        plan[v] = (b, st[i]); rel[b] = cq[i]
    return plan


def myopic_hint(inst, U, tabs):
    """MyopicC on the planning tables in arrival order (always feasible if every vessel has a table)."""
    plan = {}; rel = {}
    for v in sorted(U, key=lambda v: (inst.A[v], v)):
        best = None
        for b in range(inst.m):
            if (v, b) not in tabs: continue
            st, cq = tabs[v, b]; i = int(np.searchsorted(st, max(rel.get(b, 0), st[0])))
            if i >= len(st): continue
            if best is None or cq[i] < best[2] or (cq[i] == best[2] and b > best[0]): best = (b, st[i], cq[i])
        if best is None: return None
        plan[v] = (best[0], best[1]); rel[best[0]] = best[2]
    return plan


# ---------------------------------------------------------------- the policy (event-driven simulation)
def run_ube(inst, real: Wind, cfg: dict, rho=0.95, S=100, cap0=600.0, cap=60.0, real_idx=0, samples_fn=None, log=None):
    """Simulate UBE on the realized wind. Returns dict(Z, C, events, hint_used, time_s, beyond).
    samples_fn(t) -> list of Wind (tests only; default: S OU paths conditioned on the realized wind up to t)."""
    t0 = time.perf_counter(); n, m, A = inst.n, inst.m, inst.A
    Tmax = int(min(real.L, A[-1] + 400))
    occ = [None] * m                      # per berth: (v, r_commit, C_realized)
    C = [None] * n; berth_of = [None] * n; unberthed = set(range(n)); plan = {}
    events = 0; hint_used = 0; t = 1; last_C = 0
    while unberthed or any(o is not None and o[2] > t for o in occ):
        if t > real.L: break
        completed = [b for b in range(m) if occ[b] is not None and occ[b][2] == t]
        if unberthed and (t == 1 or completed):
            samples = samples_fn(t) if samples_fn else sample_paths(real, t, cfg, inst.beta, S, [inst.seed, real_idx, t, 77])
            # berth releases: quantile of the completion of the berthed vessels from their realized progress at t
            R = [t] * m
            for b in range(m):
                o = occ[b]
                if o is None or o[2] <= t: continue
                v, r0, _ = o
                tau = real.next_berthable(b, r0)
                if tau < t:                       # berthed before t: remaining work from the realized progress
                    s0 = tau + int(real.phi[b, tau]); st = max(s0, t)
                    done = max(0.0, real.cum[b][t] - real.cum[b][min(s0, t)]) * inst.q[v] * RHO
                    need = max(0.0, inst.h[v] - done) / (inst.q[v] * RHO)
                    cs = [remaining_vec(w, b, st, need) for w in samples]
                else:                             # committed, not yet berthed (storm): ready again at t
                    cs = [int(completion_vec(w, b, [t], [inst.h[v] / (inst.q[v] * RHO)])[0]) for w in samples]
                R[b] = int(np.quantile(np.asarray(cs), rho, method="inverted_cdf"))
            U = sorted(unberthed)
            lo = {v: [max(A[v], t, R[b]) for b in range(m)] for v in U}
            tabs = quantile_tables(inst, samples, U, lo, Tmax, rho)
            hint = None
            if plan: hint = replay(inst, U, {v: plan[v] for v in U}, tabs, lo)
            if hint is None: hint = myopic_hint(inst, U, tabs)
            if hint is None: raise RuntimeError("no feasible planning hint (horizon too short)")
            plan_new, status, obj = solve_free_tables(inst, U, tabs, hint, cap0 if t == 1 else cap)
            plan = plan_new; events += 1; hint_used += status == "HINT"
            if log: log(f"t={t} U={len(U)} status={status} obj={obj:.0f}")
        # berthing decisions at t
        for b in range(m):
            if occ[b] is not None and occ[b][2] > t: continue
            cand = [v for v in unberthed if plan[v][0] == b]
            if not cand: continue
            v = min(cand, key=lambda v: (plan[v][1], v))
            if A[v] <= t and plan[v][1] <= t:
                _, Cv = completion(real, b, t, inst.h[v], inst.q[v])
                occ[b] = (v, t, Cv); C[v] = Cv; berth_of[v] = b; unberthed.discard(v)
        t += 1
    beyond = sum(1 for c in C if c is None or c > real.L)
    Z = float(sum((C[v] if C[v] is not None else real.L + 1) - A[v] for v in range(n)))
    return dict(Z=Z, C=C, berth=berth_of, events=events, hint_used=hint_used, time_s=time.perf_counter() - t0, beyond=beyond)
