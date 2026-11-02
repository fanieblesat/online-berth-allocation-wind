"""online.py — the online protocol of DESIGN.md §3 and the non-learned online policies.

At the arrival period t = A_v of vessel v (arrival order) the decision-maker sees: all A and h (announced), the wind
realized in periods <= t, the backlog of every berth at t (derived from the realized wind up to t only), and the OU
forecast for periods > t. It assigns v to a berth; the decision is irrevocable. After all decisions the realized
schedule is evaluated by model.evaluate_ao (vessels at a berth in arrival order, berthing at the first non-storm period).
"""
from __future__ import annotations
import time
import numpy as np
from .model import completion, forecast, evaluate_ao, RHO
from .dp import solve_ao


def backlog(inst, real, lists, t):
    """Per berth: (in-service tuple or None, queued vessel ids) at period t, using the realized wind in periods < t only.
    in-service = (v, remaining_work, work_start_period) for a vessel berthed before t and not finished before t."""
    out = []
    for b in range(inst.m):
        free = 1; serv = None; queue = []
        for u in lists[b]:
            if serv is not None or free >= t:
                queue.append(u); continue
            r = max(free, inst.A[u])
            if r >= t: queue.append(u); continue
            tau = real.next_berthable(b, r)
            if tau >= t: queue.append(u); continue                     # waiting for a berthable period
            s = tau + int(real.phi[b, tau])
            done = max(0.0, (real.cum[b][t] - real.cum[b][min(s, t)])) * inst.q[u] * RHO
            if done >= inst.h[u] - 1e-9:                               # finished before t: realized C < t is known
                free = completion(real, b, r, inst.h[u], inst.q[u])[1]
                continue
            serv = (u, inst.h[u] - done, max(s, t))
        out.append((serv, queue, free))
    return out


def forecast_release(inst, fc, bl, t):
    """Forecast first-free period of every berth after its backlog, under the forecast wind fc."""
    F = []
    for b, (serv, queue, free) in enumerate(bl):
        cur = max(free, t)
        if serv is not None:
            u, rem, s = serv
            need = rem / (inst.q[u] * RHO); cum = fc.cum[b]
            cur = int(np.searchsorted(cum, cum[s] + need - 1e-9, side="left")); cur = max(cur, s)
        for u in queue:
            cur = completion(fc, b, max(cur, inst.A[u]), inst.h[u], inst.q[u])[1]
        F.append(cur)
    return F


class Obs:
    """Everything a policy may use at decision v."""
    __slots__ = ("v", "t", "fc", "F", "real_w", "bl")
    def __init__(self, v, t, fc, F, real_w, bl):
        self.v, self.t, self.fc, self.F, self.real_w, self.bl = v, t, fc, F, real_w, bl


def run_online(inst, real, cfg, policy, collect=None):
    """Simulate an online policy. policy(inst, obs) -> berth. Returns dict(Z, berth, dec_time_s, beyond)."""
    lists = [[] for _ in range(inst.m)]; berth = []; tdec = 0.0
    for v in range(inst.n):
        t = inst.A[v]
        fc = forecast(real, t, cfg, inst.beta)
        bl = backlog(inst, real, lists, t)
        F = forecast_release(inst, fc, bl, t)
        obs = Obs(v, t, fc, F, real.W[t - 1], bl)
        t0 = time.perf_counter(); b = int(policy(inst, obs)); tdec += time.perf_counter() - t0
        if collect is not None: collect.append(obs)
        lists[b].append(v); berth.append(b)
    r = evaluate_ao(inst, real, berth)
    return dict(Z=r["Z"], berth=berth, dec_time_s=tdec, beyond=r["beyond"], C=r["C"])


# ---------------------------------------------------------------- policies
def pol_first_free(inst, o):
    return min(range(inst.m), key=lambda b: (o.F[b], -b))          # ties: the less exposed (higher index) berth

def pol_myopic(inst, o):
    v = o.v
    return min(range(inst.m), key=lambda b: (completion(o.fc, b, max(o.F[b], inst.A[v]), inst.h[v], inst.q[v])[1], -b))

def pol_low_exposure(inst, o, slack=2):
    ok = [b for b in range(inst.m) if o.F[b] <= inst.A[o.v] + slack]
    return max(ok) if ok else pol_first_free(inst, o)               # beta is decreasing in b

class RollDP:
    """Re-solve the arrival-ordered DP from the current state on the forecast at every arrival; take its first berth.
    tie_exposure=True (RollDP-T, DESIGN.md §9): among optimal plans, the one whose first berth is least exposed."""
    def __init__(self, time_cap=30.0, tie_exposure=False):
        self.time_cap = time_cap; self.fallbacks = 0; self.tie_exposure = tie_exposure
    def __call__(self, inst, o):
        r = solve_ao(inst, o.fc, first=o.v, F0=o.F, time_cap=self.time_cap, tie_exposure=self.tie_exposure)
        if r.plan is None:
            self.fallbacks += 1; return pol_myopic(inst, o)
        return r.plan[0]

class Static:
    """The plan of the AO DP on the forecast made at period 1 (before the first arrival), executed without recourse.
    If the DP reaches its time cap, the plan is the MyopicC assignment simulated on that same forecast (fallback=1)."""
    def __init__(self, inst, real, cfg, time_cap=600.0):
        fc = forecast(real, 1, cfg, inst.beta)
        r = solve_ao(inst, fc, time_cap=time_cap)
        self.fallback = 0
        if r.plan is None:
            self.fallback = 1; F = [1] * inst.m; plan = []
            for v in range(inst.n):
                b = min(range(inst.m), key=lambda b: (completion(fc, b, max(F[b], inst.A[v]), inst.h[v], inst.q[v])[1], -b))
                F[b] = completion(fc, b, max(F[b], inst.A[v]), inst.h[v], inst.q[v])[1]; plan.append(b)
            r.plan = plan
        self.plan = r.plan
    def __call__(self, inst, o):
        return self.plan[o.v]

RULES = {"FirstFree": pol_first_free, "MyopicC": pol_myopic, "LowExposure": pol_low_exposure}
