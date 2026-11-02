"""lookahead.py — rolling DP with a bounded lookahead and a terminal cost-to-go (DESIGN.md §9, added 23 Oct 2026).

At the arrival of vessel v (period t, forecast fc, forecast release vector F of the berths) the arrival-ordered problem
of the next K vessels v..j-1, j = min(n, v+K), is solved exactly on fc from F, as in solve_ao but without dominance
(the terminal cost need not be monotone in F, so every distinct berth-availability vector is kept, with its cheapest
cost). Each terminal label (F_end, c) is scored c + T(F_end, j), and the first berth of the best label is taken.
Terminal costs T:
  zero     T = 0                                   (LookDP-0: a K-vessel myopic DP; the role-of-T baseline)
  rollout  T = cost of MyopicC on fc from F_end    (LookDP-R: deterministic rollout of the base policy on the forecast)
  learned  T = (n-j) * g(x(F_end, j))              (LookDP-V: g a network that estimates the expected turnaround per
                                                    remaining vessel of the base policy from state features x)
  residual T = rollout + (n-j) * g_r(x)            (LookDP-RV: the deterministic rollout corrected by a network g_r
                                                    that estimates the expected realized-minus-forecast turnaround
                                                    per remaining vessel of the base policy, i.e. the cost of wind
                                                    uncertainty that the conditional-mean forecast does not show)
With K >= n - v and T = 0 the decision equals that of the full rolling DP (tested in tests_lookahead.py).
"""
from __future__ import annotations
import math, random, time, pickle
import numpy as np
from .model import completion, forecast, RHO, WLIM, evaluate_ao
from . import online as O

H_WIN = 24          # hours of forecast summarized per berth in the state features
N_NEXT = 5          # next arrivals described individually


# ---------------------------------------------------------------- window DP
def window_labels(inst, fc, first, F0, K):
    """Labels after vessels first..j-1 on fc from F0 (normalized to A_first); returns (j, {F_end: (c, first_berth)})."""
    n, m, A = inst.n, inst.m, inst.A
    j = min(n, first + K)
    F0 = tuple(max(f, A[first]) for f in F0)
    cur = {F0: (0.0, -1)}
    for k in range(first, j):
        nxtA = A[k + 1] if k + 1 < n else None
        nxt = {}
        for F, (c, b0) in cur.items():
            for b in range(m):
                C = completion(fc, b, max(F[b], A[k]), inst.h[k], inst.q[k])[1]
                F2 = list(F); F2[b] = C
                if nxtA is not None: F2 = [max(x, nxtA) for x in F2]
                F2 = tuple(F2); c2 = c + C - A[k]; fb = b if b0 < 0 else b0
                old = nxt.get(F2)
                if old is None or c2 < old[0] or (c2 == old[0] and fb > old[1]):
                    nxt[F2] = (c2, fb)
        cur = nxt
    return j, cur


def rollout_cost(inst, fc, j, F):
    """Turnaround of vessels j..n-1 when MyopicC assigns them on fc from availability F."""
    F = list(F); tot = 0.0
    for u in range(j, inst.n):
        best, bc = 0, None
        for b in range(inst.m):
            C = completion(fc, b, max(F[b], inst.A[u]), inst.h[u], inst.q[u])[1]
            if bc is None or C < bc or (C == bc and b > best): best, bc = b, C
        F[best] = bc; tot += bc - inst.A[u]
    return tot


# ---------------------------------------------------------------- state features of the cost-to-go
def n_vfeatures(m): return 6 + 4 * m + 2 * N_NEXT

def vfeatures(inst, fc, t, j, F):
    """Features of the state 'vessel j is next, berths free at F', seen from period t with forecast fc."""
    n, m, A = inst.n, inst.m, inst.A
    Aj = A[j]; r = n - j
    x = [r / (10.0 * m), (Aj - t) / 24.0]
    rem_H = sum(inst.h[u] / (RHO * inst.q[u]) for u in range(j, n))
    span = max(1, A[n - 1] - Aj + 1)
    x += [rem_H / (m * span), rem_H / (24.0 * m * max(1, r)), fc.W[min(Aj, fc.L) - 1] / WLIM,
          float(np.mean(fc.W[min(Aj, fc.L) - 1:min(Aj + H_WIN, fc.L)])) / WLIM]
    for b in range(m):
        lo, hi = min(Aj, fc.L), min(Aj + H_WIN, fc.L)
        x += [max(0, F[b] - Aj) / 24.0, inst.beta[b], float(fc.mu[b, lo:hi + 1].mean()), float(fc.S[b, lo:hi + 1].mean())]
    for i in range(1, N_NEXT + 1):
        u = j + i - 1
        if u < n: x += [(A[u] - Aj) / 24.0, inst.h[u] / (RHO * inst.q[u]) / 24.0]
        else:     x += [0.0, 0.0]
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------- the policy
class LookDP:
    """terminal in {'zero','rollout','learned'}; net = trained cost-to-go (learned only)."""
    def __init__(self, K, terminal="zero", net=None):
        self.K, self.terminal, self.net = K, terminal, net
        self.labels = 0
    def __call__(self, inst, o):
        j, labs = window_labels(inst, o.fc, o.v, o.F, self.K)
        self.labels += len(labs)
        keys = list(labs.keys())
        cost = np.array([labs[F][0] for F in keys], dtype=float)
        if j < inst.n and self.terminal == "rollout":
            cost = cost + np.array([rollout_cost(inst, o.fc, j, F) for F in keys])
        elif j < inst.n and self.terminal == "learned":
            X = np.stack([vfeatures(inst, o.fc, o.t, j, F) for F in keys])
            cost = cost + (inst.n - j) * 24.0 * self.net.predict(X)
        elif j < inst.n and self.terminal == "residual":
            X = np.stack([vfeatures(inst, o.fc, o.t, j, F) for F in keys])
            cost = cost + np.array([rollout_cost(inst, o.fc, j, F) for F in keys]) + (inst.n - j) * 24.0 * self.net.predict(X)
        best = min(range(len(keys)), key=lambda i: (cost[i], -labs[keys[i]][1]))
        return labs[keys[best]][1]


# ---------------------------------------------------------------- learning the cost-to-go
def sample_states(inst, real, cfg, Kmax, base=O.pol_myopic, rng=None):
    """One episode of the base policy on the realized wind. At every arrival v, the base policy is simulated on the
    forecast of that moment for k = 1..Kmax further vessels; each predicted state (j = v+k, F_pred) is a sample with
    features vfeatures(fc_v, t_v, j, F_pred); targets: y = realized turnaround per remaining vessel of the base policy
    from vessel j on (days); yr = y minus the same quantity of the base policy simulated on the forecast fc_v from
    (j, F_pred). Returns (X, y, yr)."""
    lists = [[] for _ in range(inst.m)]; berth = []; snaps = []
    for v in range(inst.n):
        t = inst.A[v]
        fc = forecast(real, t, cfg, inst.beta); bl = O.backlog(inst, real, lists, t)
        F = O.forecast_release(inst, fc, bl, t); o = O.Obs(v, t, fc, F, real.W[t - 1], bl)
        Fp = [max(f, t) for f in F]
        for k in range(1, Kmax + 1):
            u = v + k - 1
            if u >= inst.n - 1: break
            bb = min(range(inst.m), key=lambda b: (completion(fc, b, max(Fp[b], inst.A[u]), inst.h[u], inst.q[u])[1], -b))
            Fp[bb] = completion(fc, bb, max(Fp[bb], inst.A[u]), inst.h[u], inst.q[u])[1]
            Fp = [max(x, inst.A[u + 1]) for x in Fp]
            snaps.append((v + k, vfeatures(inst, fc, t, v + k, Fp), rollout_cost(inst, fc, v + k, Fp) / 24.0))
        b = int(base(inst, o)); lists[b].append(v); berth.append(b)
    C = evaluate_ao(inst, real, berth)["C"]
    ctg = np.zeros(inst.n + 1)
    for u in range(inst.n - 1, -1, -1): ctg[u] = ctg[u + 1] + (C[u] - inst.A[u]) / 24.0
    X = np.stack([s for _, s, _ in snaps])
    y = np.array([ctg[j] / (inst.n - j) for j, _, _ in snaps], dtype=np.float32)              # LookDP-V target
    yr = np.array([(ctg[j] - rc) / (inst.n - j) for j, _, rc in snaps], dtype=np.float32)    # LookDP-RV target
    return X, y, yr


class VNet:
    """MLP g(x) ~ expected turnaround per remaining vessel (days); fitted by MSE with early stopping on a validation set."""
    def __init__(self, d, hidden=128, seed=0):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.mu = np.zeros(d, dtype=np.float32); self.sd = np.ones(d, dtype=np.float32)
        self.f = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def predict(self, X):
        import torch
        with torch.no_grad():
            z = torch.from_numpy(((X - self.mu) / self.sd).astype(np.float32))
            return self.f(z).squeeze(-1).numpy().astype(float)
    def fit(self, X, y, Xv, yv, epochs=200, lr=1e-3, batch=512, patience=20, log=print):
        import torch, torch.nn as nn
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xt = torch.from_numpy((X - self.mu) / self.sd); yt = torch.from_numpy(y)
        opt = torch.optim.Adam(self.f.parameters(), lr=lr); best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), batch):
                idx = perm[i:i + batch]; opt.zero_grad()
                loss = nn.functional.mse_loss(self.f(Xt[idx]).squeeze(-1), yt[idx]); loss.backward(); opt.step()
            v = float(np.mean((self.predict(Xv) - yv) ** 2))
            if v < best - 1e-7: best, bad = v, 0; best_state = {k: t.clone() for k, t in self.f.state_dict().items()}; best_ep = ep
            else: bad += 1
            if bad >= patience: break
        self.f.load_state_dict(best_state)
        log(f"  VNet: best validation MSE {best:.5f} at epoch {best_ep + 1}")
        return best
