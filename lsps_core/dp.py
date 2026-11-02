"""dp.py — exact label-setting DP on the arrival-ordered decision graph (DESIGN.md §1 contribution 2).

Layer k holds labels (F, c): F = berth-availability vector after vessels 0..k-1 are assigned (first free period of
each berth), c = turnaround accumulated so far. Assigning vessel k to berth b: ready r = max(F_b, A_k),
(tau, C) = completion(b, r), F_b <- C, c <- c + C - A_k.

Two reductions, both exact under the FIFO proposition (completion non-decreasing in r):
  normalization  F_b <- max(F_b, A_{k+1}) after each layer — no later vessel can use the berth before A_{k+1};
  dominance      (F, c) is dropped if a kept label (F', c') has F' <= F componentwise and c' <= c.
  bounding       (F, c) is dropped if c + LB(F) > UB, with UB the cost of a feasible completion (greedy) and
                 LB(F) = sum over the remaining vessels u of min_b (C_b(u, A_u) - A_u) (no waiting; valid by FIFO).
`dominance=False, normalize=False, bound=False` gives the plain DP of IC_TORS (used by the agreement check).
"""
from __future__ import annotations
import time
import numpy as np
from .model import completion


class DPResult:
    __slots__ = ("cost", "plan", "labels", "peak", "time_s", "status")
    def __init__(self, cost, plan, labels, peak, time_s, status):
        self.cost, self.plan, self.labels, self.peak, self.time_s, self.status = cost, plan, labels, peak, time_s, status


class _Cap(Exception):
    pass


def _pareto(cands, deadline=None, tie_exposure=False):
    """cands: dict F(tuple) -> (c, parent_idx, berth[, first_berth]). Returns list of (F, c, parent, berth) not dominated.
    tie_exposure: among equal costs, labels whose first berth is less exposed (higher index) are kept first.
    Raises _Cap if the deadline passes (checked every 2048 candidates)."""
    key = (lambda kv: (kv[1][0], -kv[1][3], sum(kv[0]))) if tie_exposure else (lambda kv: (kv[1][0], sum(kv[0])))
    items = sorted(cands.items(), key=key)
    if not items: return []
    m = len(items[0][0])
    arr = np.empty((len(items), m), dtype=np.int64); K = 0; kept = []
    for i, (F, (c, par, b, *_)) in enumerate(items):
        if deadline is not None and i % 2048 == 0 and time.perf_counter() > deadline: raise _Cap
        f = np.asarray(F, dtype=np.int64)
        # every kept label has c' <= c (cost order), so F' <= F means (F, c) is dominated
        if K and np.any(np.all(arr[:K] <= f, axis=1)):
            continue
        arr[K] = f; K += 1; kept.append((F, c, par, b))
    return kept


def solve_ao(inst, wind, first=0, F0=None, dominance=True, normalize=True, bound=True, label_cap=2_000_000, time_cap=None,
             tie_exposure=False):
    """Optimal arrival-ordered assignment of vessels first..n-1 from availability vector F0 under `wind`.
    Returns DPResult with cost = sum over those vessels of (C_v - A_v) and plan = berths for first..n-1.
    tie_exposure (added 23 Oct 2026, DESIGN.md §9): among the optimal labels kept by the DP, the plan whose first berth is
    the least exposed (highest index); default False reproduces the campaign exactly."""
    t0 = time.perf_counter()
    m, n, A = inst.m, inst.n, inst.A
    F0 = tuple(F0) if F0 is not None else tuple([1] * m)
    if normalize and first < n:
        F0 = tuple(max(f, A[first]) for f in F0)
    layers = [[(F0, 0.0, -1, -1)]]
    total = 1; peak = 1
    UB = float("inf"); LBr = [0.0] * (n + 1)
    if bound and first < n:
        # LB of the vessels after k: each finishes no earlier than with no waiting at its best berth
        for u in range(n - 1, first - 1, -1):
            LBr[u] = LBr[u + 1] + min(completion(wind, b, A[u], inst.h[u], inst.q[u])[1] - A[u] for b in range(m))
        # UB: greedy completion (each vessel to the berth that finishes it first)
        F = list(F0); ub = 0.0
        for u in range(first, n):
            best = min(range(m), key=lambda b: completion(wind, b, max(F[b], A[u]), inst.h[u], inst.q[u])[1])
            C = completion(wind, best, max(F[best], A[u]), inst.h[u], inst.q[u])[1]; F[best] = C; ub += C - A[u]
        UB = ub + 1e-9
    deadline = t0 + time_cap if time_cap else None
    for k in range(first, n):
        cands = {}
        nxtA = A[k + 1] if k + 1 < n else None
        for idx, (F, c, _, _) in enumerate(layers[-1]):
            if deadline is not None and idx % 512 == 0 and time.perf_counter() > deadline:
                return DPResult(float("nan"), None, total, peak, time.perf_counter() - t0, "cap")
            for b in range(m):
                r = max(F[b], A[k])
                _, C = completion(wind, b, r, inst.h[k], inst.q[k])
                F2 = list(F); F2[b] = C
                if normalize and nxtA is not None:
                    F2 = [max(x, nxtA) for x in F2]
                F2 = tuple(F2); c2 = c + C - A[k]
                if c2 + LBr[k + 1] > UB: continue
                old = cands.get(F2)
                if tie_exposure:
                    fb = b if k == first else fbs[idx]
                    if old is None or c2 < old[0] or (c2 == old[0] and fb > old[3]):
                        cands[F2] = (c2, idx, b, fb)
                elif old is None or c2 < old[0]:
                    cands[F2] = (c2, idx, b)
        try:
            layer = _pareto(cands, deadline, tie_exposure) if dominance else [(F, v[0], v[1], v[2]) for F, v in cands.items()]
            if tie_exposure: fbs = [cands[L[0]][3] for L in layer]
        except _Cap:
            return DPResult(float("nan"), None, total, peak, time.perf_counter() - t0, "cap")
        layers.append(layer); total += len(layer); peak = max(peak, len(layer))
        if total > label_cap or (time_cap and time.perf_counter() - t0 > time_cap):
            return DPResult(float("nan"), None, total, peak, time.perf_counter() - t0, "cap")
    # backtrack from the cheapest terminal label
    last = layers[-1]
    j = min(range(len(last)), key=(lambda i: (last[i][1], -fbs[i])) if tie_exposure and first < n else (lambda i: last[i][1]))
    cost = last[j][1]; plan = []
    for L in range(len(layers) - 1, 0, -1):
        F, c, par, b = layers[L][j]; plan.append(b); j = par
    plan.reverse()
    return DPResult(float(cost), plan, total, peak, time.perf_counter() - t0, "optimal")
