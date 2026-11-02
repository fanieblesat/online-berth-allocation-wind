"""model.py — instances, wind, the completion function and the realized-schedule simulator (DESIGN.md §2, §5).

Time is in hourly periods t = 1, 2, ...  A vessel v that is ready at berth b in period r
  * berths at tau = first period >= r with S[b][tau] == 0 (no berthing in a storm),
  * moors phi[b][tau] periods (no work),
  * then receives q_v * RHO * mu[b][t] moves in every period t (mu = 0 in a storm),
  * ends at C = (last period with work) + 1, i.e. the berth is free from period C on.
Turnaround of v is C_v - A_v; Z = sum_v (C_v - A_v).

The wind model (the discretized Ornstein-Uhlenbeck process, productivity mu(w), mooring delay phi(w), limit WLIM), the
berth series, cumulative-work tables for O(log T) completion queries, and the OU forecast are defined in this module.
"""
from __future__ import annotations
import bisect, csv, os
from datetime import datetime
import numpy as np

# ---------------------------------------------------------------- constants (Table 1 of the paper)
W1, WLIM, ALPHA_DEFAULT, RHO, PHI1 = 10.0, 20.0, 0.5, 30.0, 1
WIND_LEN = 1000
CLASS_GANG = {"feeder": 1, "seagoing vessel": 2}          # fixed nominal gang per class (DESIGN §2, decision 2)
BETA = {2: [1.0, 0.4], 3: [1.0, 0.7, 0.4], 4: [1.0, 0.8, 0.6, 0.4], 5: [1.0, 0.85, 0.7, 0.55, 0.4]}

def mu_of(w, alpha=ALPHA_DEFAULT):
    if w < W1: return 1.0
    if w < WLIM: return round(1.0 - alpha * (w - W1) / (WLIM - W1), 3)
    return 0.0

def phi_of(w):
    return 0 if w < W1 else PHI1        # berthing is impossible at w >= WLIM, so phi in {0, 1} at admissible berthings

# ---------------------------------------------------------------- the Hamburg record
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hamburg_bap_vessels_with_weather.csv")
_REC = None
def record():
    """(calls, gaps): calls = [(class, H0 hours)], gaps = observed inter-arrival times in hours (sorted record)."""
    global _REC
    if _REC is None:
        rows = list(csv.DictReader(open(DATA)))
        ts = [datetime.strptime(r["arrival_time"], "%Y-%m-%dT%H:%MZ") for r in rows]
        order = sorted(range(len(rows)), key=lambda i: ts[i])
        ts = [ts[i] for i in order]; rows = [rows[i] for i in order]
        gaps = [(ts[i] - ts[i - 1]).total_seconds() / 3600.0 for i in range(1, len(ts))]
        _REC = ([(r["vessel_type"], float(r["H_v0"])) for r in rows], gaps)
    return _REC


class Instance:
    __slots__ = ("n", "m", "A", "h", "q", "vtype", "beta", "seed", "cfg")
    def __init__(self, n, m, A, h, q, vtype, beta, seed, cfg):
        self.n, self.m, self.A, self.h, self.q, self.vtype, self.beta, self.seed, self.cfg = n, m, A, h, q, vtype, beta, seed, cfg


def make_instance(seed: int, cfg: dict) -> Instance:
    """|V| calls bootstrapped from the record (deterministic in seed): (class, H0) resampled jointly, inter-arrival
    gaps resampled from the observed gaps. Load control (DESIGN §5): the gaps are rescaled so that the calm per-berth
    utilization E[H0] / (|B| * mean gap) equals cfg['load'] (default 0.8) at every size. Clustering (sensitivity):
    g -> mean * (g / mean) ** kappa, renormalized to the same mean (kappa = 1: the record's dispersion)."""
    rng = np.random.RandomState(seed)
    calls, gaps = record()
    n, m = int(cfg["n_vessels"]), int(cfg["n_berths"])
    load, kappa = float(cfg.get("load", 0.8)), float(cfg.get("kappa", 1.0))
    mean_H0 = float(np.mean([c[1] for c in calls])); mean_gap = float(np.mean(gaps))
    target_gap = mean_H0 / (load * m)
    idx = rng.randint(0, len(calls), size=n)
    g = np.asarray(rng.choice(gaps, size=n - 1, replace=True), dtype=float) if n > 1 else np.zeros(0)
    if kappa != 1.0 and len(g):
        base = np.asarray(gaps, dtype=float); norm = float(np.mean((base / mean_gap) ** kappa))
        g = mean_gap * (g / mean_gap) ** kappa / norm
    g = g * (target_gap / mean_gap)
    t = 1.0; A = [1]
    for x in g:
        t += x; A.append(int(round(t)))
    h, q, vt = [], [], []
    for i in idx:
        cls, H0 = calls[i]; qv = CLASS_GANG[cls]
        h.append(float(round(RHO * qv * H0))); q.append(qv); vt.append(cls)
    return Instance(n, m, A, h, q, vt, list(cfg.get("beta", BETA[m])), seed, cfg)


# ---------------------------------------------------------------- wind
class Wind:
    """Berth series of one wind trajectory W[1..L] (realized or forecast). Index t = 1..L, position 0 unused."""
    __slots__ = ("W", "m", "S", "mu", "phi", "cum", "L", "alpha", "_next_ok")
    def __init__(self, W, beta, alpha=ALPHA_DEFAULT):
        self.W = list(W); self.m = len(beta); self.L = len(W); self.alpha = alpha
        L = self.L
        self.S = np.zeros((self.m, L + 2), dtype=np.int8)
        self.mu = np.zeros((self.m, L + 2))
        self.phi = np.zeros((self.m, L + 2), dtype=np.int8)
        Wv = np.asarray(self.W, dtype=float)
        for b, bb in enumerate(beta):
            w = bb * Wv
            mu = np.where(w < W1, 1.0, np.where(w < WLIM, np.round(1.0 - alpha * (w - W1) / (WLIM - W1), 3), 0.0))
            self.S[b, 1:L + 1] = (w >= WLIM).astype(np.int8)
            self.mu[b, 1:L + 1] = mu
            self.phi[b, 1:L + 1] = np.where(w < W1, 0, PHI1).astype(np.int8)
        # cum[b][t] = sum_{s < t} mu[b][s]  (work of one crane at RHO = 1); completion via bisection
        self.cum = np.concatenate([np.zeros((self.m, 1)), np.cumsum(self.mu, axis=1)], axis=1)
        self._next_ok = None

    def next_berthable(self, b, r):
        """First period >= r with no storm at berth b (L+1 if none within the trajectory)."""
        if self._next_ok is None:
            nxt = np.full((self.m, self.L + 3), self.L + 1, dtype=np.int64)
            for bb in range(self.m):
                cur = self.L + 1
                for t in range(self.L, 0, -1):
                    if self.S[bb, t] == 0: cur = t
                    nxt[bb, t] = cur
            self._next_ok = nxt
        r = max(1, int(r))
        return int(self._next_ok[b, r]) if r <= self.L else self.L + 1


def completion(wind: Wind, b: int, r: int, work: float, q: int):
    """Berth at the first berthable period tau >= r, moor phi, work q*RHO*mu per period.
    Returns (tau, C) with C = first period at which the berth is free again. C = L+1 marks 'beyond the trajectory'."""
    tau = wind.next_berthable(b, r)
    if tau > wind.L: return tau, wind.L + 1
    s = tau + int(wind.phi[b, tau])                  # first work period
    need = work / (q * RHO)
    cum = wind.cum[b]                                # cum[t] = work of one crane-unit in periods < t
    target = cum[s] + need - 1e-9
    t_end = int(np.searchsorted(cum, target, side="left"))   # smallest t_end with cum[t_end] >= target
    if t_end > wind.L + 1: return tau, wind.L + 1
    return tau, max(t_end, s)


def make_wind(seed: int, cfg: dict, beta) -> Wind:
    """The discretized Ornstein-Uhlenbeck wind process of Eq. (1), seeded per (instance, realization)."""
    mu_w, sigma_w, theta_w = float(cfg["mu_w"]), float(cfg["sigma_w"]), float(cfg["theta_w"])
    rng = np.random.RandomState(seed)
    W = np.empty(WIND_LEN)
    W[0] = max(0.0, mu_w + rng.normal(0.0, 2.0))
    noise = rng.normal(size=WIND_LEN)
    for t in range(1, WIND_LEN):
        W[t] = max(0.0, W[t - 1] + theta_w * (mu_w - W[t - 1]) + sigma_w * noise[t])
    return Wind(W.tolist(), beta, float(cfg.get("alpha", ALPHA_DEFAULT)))


def forecast(realized: Wind, t0: int, cfg: dict, beta) -> Wind:
    """Information at period t0: the realized wind in periods <= t0, the OU conditional mean afterwards,
    E[W_{t0+k} | W_{t0}] = mu_w + (W_{t0} - mu_w)(1 - theta_w)^k (clipped at 0)."""
    mu_w, theta_w = float(cfg["mu_w"]), float(cfg["theta_w"])
    W = list(realized.W[:t0])                        # periods 1..t0
    w0 = realized.W[t0 - 1] if t0 >= 1 else mu_w
    f = 1.0
    for _ in range(realized.L - t0):
        f *= (1.0 - theta_w); W.append(max(0.0, mu_w + (w0 - mu_w) * f))
    return Wind(W, beta, realized.alpha)


# ---------------------------------------------------------------- schedules
def evaluate_ao(inst: Instance, wind: Wind, berth_of):
    """Realized schedule of an arrival-ordered assignment (vessels at a berth served in arrival order).
    Returns dict(Z, C, tau, beyond) ; beyond = number of vessels whose completion falls beyond the trajectory."""
    free = [1] * inst.m; C = [0] * inst.n; tau = [0] * inst.n; beyond = 0
    for v in range(inst.n):
        b = berth_of[v]; r = max(free[b], inst.A[v])
        tv, cv = completion(wind, b, r, inst.h[v], inst.q[v])
        tau[v], C[v] = tv, cv; free[b] = cv; beyond += cv > wind.L
    return dict(Z=float(sum(C[v] - inst.A[v] for v in range(inst.n))), C=C, tau=tau, beyond=beyond)


def evaluate_sequence(inst: Instance, wind: Wind, berth_of, order):
    """Realized schedule when each berth serves its vessels in the given order (list of vessel ids, global order)
    with the same berthing rule (first berthable period after ready). Used to replay FREE solutions."""
    free = [1] * inst.m; C = [0] * inst.n; tau = [0] * inst.n
    for v in order:
        b = berth_of[v]; r = max(free[b], inst.A[v])
        tv, cv = completion(wind, b, r, inst.h[v], inst.q[v]); tau[v], C[v] = tv, cv; free[b] = cv
    return dict(Z=float(sum(C[v] - inst.A[v] for v in range(inst.n))), C=C, tau=tau)


def fifo_check(inst: Instance, wind: Wind, horizon: int | None = None) -> bool:
    """Proposition (FIFO): for every vessel and berth, completion(r) is non-decreasing in the ready period r."""
    H = horizon or min(wind.L, inst.A[-1] + 200)
    for v in range(inst.n):
        for b in range(inst.m):
            prev = -1
            for r in range(inst.A[v], H):
                c = completion(wind, b, r, inst.h[v], inst.q[v])[1]
                if c < prev: return False
                prev = c
    return True
