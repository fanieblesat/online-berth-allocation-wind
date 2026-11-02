"""exact.py — CP-SAT model of the BAP with start-dependent durations (perfect information; DESIGN.md §4).

For every (v, b): optional interval [s_vb, e_vb) present iff x_vb; s_vb ranges over the berthable periods
(no storm at b) in [A_v, Tmax]; e_vb = C_b(v, s_vb) via an element constraint on the precomputed completion table.
NoOverlap per berth. Objective sum_v (e_v - A_v). Variant AO adds: same berth => vessel order = arrival order.
The model lets a vessel berth later than its ready time; under the FIFO proposition this never helps, so its optimum
equals that of the simulator's berthing rule (0_agreement.py checks it on every exact-check instance).
"""
from __future__ import annotations
import time
from ortools.sat.python import cp_model
from .model import completion, evaluate_ao


def solve_cpsat(inst, wind, variant="free", tmax=None, time_limit=3600.0, workers=8, hint_plan=None, log=False):
    n, m, A = inst.n, inst.m, inst.A
    Tmax = int(tmax or min(wind.L, A[-1] + 400))
    mdl = cp_model.CpModel()
    x = {}; s = {}; e = {}; iv = {}; kk = {}; tab = {}
    ends = []
    for v in range(n):
        lits = []
        ev = mdl.NewIntVar(A[v], Tmax + 1, f"e{v}")
        for b in range(m):
            starts = [t for t in range(A[v], Tmax + 1) if wind.S[b, t] == 0]
            if not starts: continue
            comp = [completion(wind, b, t, inst.h[v], inst.q[v])[1] for t in starts]
            ok = [(t, c) for t, c in zip(starts, comp) if c <= Tmax + 1]
            if not ok: continue
            starts, comp = [t for t, _ in ok], [c for _, c in ok]
            xb = mdl.NewBoolVar(f"x{v}_{b}")
            k = mdl.NewIntVar(0, len(starts) - 1, f"k{v}_{b}")
            sv = mdl.NewIntVar(min(starts), max(starts), f"s{v}_{b}")
            evb = mdl.NewIntVar(min(comp), max(comp), f"e{v}_{b}")
            mdl.AddElement(k, starts, sv); mdl.AddElement(k, comp, evb)
            tab[v, b] = starts
            size = mdl.NewIntVar(0, Tmax + 1, f"d{v}_{b}")
            iv[v, b] = mdl.NewOptionalIntervalVar(sv, size, evb, xb, f"i{v}_{b}")
            mdl.Add(ev == evb).OnlyEnforceIf(xb)
            x[v, b], s[v, b], e[v, b], kk[v, b] = xb, sv, evb, k; lits.append(xb)
        mdl.AddExactlyOne(lits); ends.append(ev)
    for b in range(m):
        mdl.AddNoOverlap([iv[v, b] for v in range(n) if (v, b) in iv])
    if variant == "ao":
        for b in range(m):
            vs = [v for v in range(n) if (v, b) in iv]
            for i, u in enumerate(vs):
                for v in vs[i + 1:]:
                    mdl.Add(s[v, b] >= e[u, b]).OnlyEnforceIf([x[u, b], x[v, b]])
    obj = sum(ends[v] - A[v] for v in range(n))
    mdl.Minimize(obj)
    if hint_plan is not None:
        # complete hint = the arrival-ordered schedule of hint_plan (feasible for both variants); its value is a valid
        # upper bound on the optimum, added as a constraint (it cannot cut off an optimal solution)
        sch = evaluate_ao(inst, wind, hint_plan)
        if all(sch["C"][v] <= Tmax + 1 and (v, hint_plan[v]) in x for v in range(n)):
            for v in range(n):
                bv = hint_plan[v]
                for b in range(m):
                    if (v, b) in x: mdl.AddHint(x[v, b], 1 if b == bv else 0)
                ti = tab[v, bv].index(sch["tau"][v]); mdl.AddHint(kk[v, bv], ti)
                mdl.AddHint(s[v, bv], sch["tau"][v]); mdl.AddHint(e[v, bv], sch["C"][v]); mdl.AddHint(ends[v], sch["C"][v])
            mdl.Add(obj <= int(round(sch["Z"])))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = float(time_limit); sol.parameters.num_workers = int(workers)
    sol.parameters.log_search_progress = bool(log)
    t0 = time.perf_counter(); st = sol.Solve(mdl); t = time.perf_counter() - t0
    name = sol.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(status=name, obj=float("nan"), bound=float(sol.BestObjectiveBound()), time_s=t, berth=None, order=None, Tmax=Tmax)
    berth = [next(b for b in range(m) if (v, b) in x and sol.Value(x[v, b])) for v in range(n)]
    start = [sol.Value(s[v, berth[v]]) for v in range(n)]
    order = sorted(range(n), key=lambda v: (start[v], v))
    at_bound = any(sol.Value(ends[v]) >= Tmax for v in range(n))
    return dict(status=name, obj=float(sol.ObjectiveValue()), bound=float(sol.BestObjectiveBound()), time_s=t,
                berth=berth, order=order, start=start, Tmax=Tmax, at_horizon=at_bound)
