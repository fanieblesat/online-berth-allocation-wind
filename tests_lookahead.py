"""tests_lookahead.py — exactness of the window DP (DESIGN.md §9): with K >= n - v and zero terminal cost, the cheapest
window label equals the AO DP optimum from the same state on the same forecast, at every decision of an episode."""
import time
from common import inst_of, wind_of, wind_seed, POINTS
from lsps_core import online as O
from lsps_core.dp import solve_ao
from lsps_core.lookahead import window_labels, LookDP
from lsps_core.model import forecast
bad = 0; n_chk = 0; t0 = time.time()
for pt, seeds in (("V20", [4000, 4001, 4002, 4003, 4004]),):   # K = n without dominance is exponential beyond V20
    cfg = POINTS[pt]
    for s in seeds:
        inst = inst_of(pt, s); real = wind_of(pt, inst, wind_seed(s, 0))
        lists = [[] for _ in range(inst.m)]
        for v in range(inst.n):
            t = inst.A[v]; fc = forecast(real, t, cfg, inst.beta); bl = O.backlog(inst, real, lists, t)
            F = O.forecast_release(inst, fc, bl, t)
            j, labs = window_labels(inst, fc, v, F, inst.n)
            c_win = min(c for c, _ in labs.values()); r = solve_ao(inst, fc, first=v, F0=F)
            n_chk += 1
            if abs(c_win - r.cost) > 1e-6: bad += 1; print("MISMATCH", pt, s, v, c_win, r.cost)
            lists[r.plan[0]].append(v)
        # the policy runs
        Z0 = O.run_online(inst, real, cfg, LookDP(inst.n, "zero"))["Z"]; Zr = O.run_online(inst, real, cfg, O.RollDP(time_cap=None))["Z"]
        print(f"{pt} {s}: LookDP(K=n, zero) Z={Z0}  RollDP Z={Zr}")
print(f"{n_chk} decisions checked, {bad} mismatches, {time.time()-t0:.0f}s")
