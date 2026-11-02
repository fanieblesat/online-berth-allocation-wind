"""learners.py — tabular Q-learning and DQN on the online protocol (DESIGN.md §4). Greedy-policy evaluation only.

Decision process (identical for both, and identical to the one the paper describes):
  decision epoch  = arrival of vessel v (arrival order); action = berth b in {0..|B|-1};
  observation     = online.Obs (realized wind up to A_v, berth backlogs, OU forecast) -> features below;
  cost of action  = realized turnaround C_v - A_v of v (known to the environment, not to the agent at decision time);
  episode         = one instance x one wind realization; return = -Z; gamma = 1 (finite horizon |V|).
Training instances, validation instances and test instances are disjoint seed pools (common.py).
"""
from __future__ import annotations
import math, os, random, time, pickle
import numpy as np
import torch, torch.nn as nn
from .model import completion, RHO, WLIM, forecast, evaluate_ao
from . import online as O


# ---------------------------------------------------------------- features
H_LOOK = 12
def features(inst, o):
    v, t = o.v, o.t
    x = [inst.h[v] / (RHO * inst.q[v]) / 10.0, inst.q[v] - 1.0, v / inst.n]
    for b in range(inst.m):
        x.append((o.F[b] - t) / 24.0)
        x.append(inst.beta[b] * o.real_w / WLIM)
        x.append(float(o.fc.S[b, t + 1:t + 1 + H_LOOK].mean()))
        x.append(float(o.fc.mu[b, t + 1:t + 1 + H_LOOK].mean()))
        x.append((completion(o.fc, b, max(o.F[b], inst.A[v]), inst.h[v], inst.q[v])[1] - inst.A[v]) / 24.0)
    return np.asarray(x, dtype=np.float32)

def n_features(m): return 3 + 5 * m


def episode_steps(inst, real, cfg, policy_fn, record=False):
    """Run one episode with policy_fn(obs, feats) -> b. Returns (Z, transitions [(s, a, cost, s_next, done)])."""
    lists = [[] for _ in range(inst.m)]; berth = []; trans = []; prev = None
    free = [1] * inst.m
    for v in range(inst.n):
        t = inst.A[v]
        fc = forecast(real, t, cfg, inst.beta); bl = O.backlog(inst, real, lists, t)
        F = O.forecast_release(inst, fc, bl, t); o = O.Obs(v, t, fc, F, real.W[t - 1], bl)
        s = features(inst, o); b = int(policy_fn(o, s))
        r = max(free[b], inst.A[v]); C = completion(real, b, r, inst.h[v], inst.q[v])[1]; free[b] = C
        cost = (C - inst.A[v]) / 24.0
        if record:
            if prev is not None: trans.append((prev[0], prev[1], prev[2], s, 0.0))
            prev = (s, b, cost)
        lists[b].append(v); berth.append(b)
    if record and prev is not None:
        trans.append((prev[0], prev[1], prev[2], np.zeros_like(prev[0]), 1.0))
    Z = evaluate_ao(inst, real, berth)["Z"]
    return Z, trans


# ---------------------------------------------------------------- DQN
class QNet(nn.Module):
    def __init__(self, d, m, hidden=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, m))
    def forward(self, x): return self.f(x)


def train_dqn(seed, cfg, make_inst, make_wind_, train_seeds, valid_pairs, episodes=4000, ckpt_every=200,
              lr=5e-4, lr_final=5e-5, batch=64, buf_cap=100_000, gamma=1.0, eps_min=0.05, eps_decay=None,
              target_every=20, hidden=128, resume_path=None, probe_csv=None, tag="dqn", log=print):
    """Double DQN, costs minimized (Q = expected cost-to-go). Checkpoint selection on valid_pairs [(inst, wind)].
    resume_path: full training state saved at every validation step and restored on restart (bit-identical on the
    same machine and software stack)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = int(cfg["n_berths"]); d = n_features(m)
    net, tgt = QNet(d, m, hidden), QNet(d, m, hidden); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    eps_decay = eps_decay or (eps_min ** (1.0 / (0.8 * episodes)))
    S = np.zeros((buf_cap, d), np.float32); S2 = np.zeros((buf_cap, d), np.float32)
    Aa = np.zeros(buf_cap, np.int64); Cc = np.zeros(buf_cap, np.float32); D = np.zeros(buf_cap, np.float32)
    nb = 0; ptr = 0; eps = 1.0; best = math.inf; best_state = None; best_ep = 0; hist = []; ep0 = 0; spent = 0.0
    guard = dict(seed=seed, m=m, episodes=episodes, tag=tag, hidden=hidden)
    if resume_path and os.path.exists(resume_path):
        blob = torch.load(resume_path, weights_only=False)
        if blob["guard"] != guard: raise RuntimeError(f"checkpoint {resume_path} was written by another configuration")
        net.load_state_dict(blob["net"]); tgt.load_state_dict(blob["tgt"]); opt.load_state_dict(blob["opt"])
        S, S2, Aa, Cc, D = blob["S"], blob["S2"], blob["A"], blob["C"], blob["D"]
        nb, ptr, eps, best, best_state, best_ep, hist, ep0, spent = (blob[k] for k in
            ("nb", "ptr", "eps", "best", "best_state", "best_ep", "hist", "ep", "spent"))
        random.setstate(blob["rng_py"]); np.random.set_state(blob["rng_np"]); torch.set_rng_state(blob["rng_t"])
        log(f"  [{tag} seed {seed}] resumed at episode {ep0}")
    t0 = time.time()
    def act_greedy(s):
        with torch.no_grad(): return int(net(torch.from_numpy(s)[None]).argmin(1).item())
    for ep in range(ep0, episodes):
        isd = train_seeds[ep % len(train_seeds)]
        inst = make_inst(isd); real = make_wind_(isd, 30_000_000 + 7919 * ep + 101 * seed, inst)
        rng = random.Random(1_000_003 * seed + ep)
        pol = lambda o, s: rng.randrange(m) if rng.random() < eps else act_greedy(s)
        _, trans = episode_steps(inst, real, cfg, pol, record=True)
        for (s, a, c, s2, dn) in trans:
            S[ptr], Aa[ptr], Cc[ptr], S2[ptr], D[ptr] = s, a, c, s2, dn; ptr = (ptr + 1) % buf_cap; nb = min(nb + 1, buf_cap)
        if nb >= batch:
            for _ in range(len(trans)):
                idx = np.random.randint(0, nb, batch)
                bs, bs2 = torch.from_numpy(S[idx]), torch.from_numpy(S2[idx])
                q = net(bs).gather(1, torch.from_numpy(Aa[idx])[:, None]).squeeze(1)
                with torch.no_grad():
                    a2 = net(bs2).argmin(1, keepdim=True)
                    y = torch.from_numpy(Cc[idx]) + gamma * (1 - torch.from_numpy(D[idx])) * tgt(bs2).gather(1, a2).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        eps = max(eps_min, eps * eps_decay)
        for g in opt.param_groups:
            g["lr"] = lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * (ep + 1) / episodes))
        if (ep + 1) % target_every == 0: tgt.load_state_dict(net.state_dict())
        if (ep + 1) % ckpt_every == 0 or ep + 1 == episodes:
            net.eval()
            val = float(np.mean([episode_steps(i, w, cfg, lambda o, s: act_greedy(s))[0] for i, w in valid_pairs]))
            net.train()
            hist.append((ep + 1, val, eps, spent + time.time() - t0))
            if val < best: best, best_ep = val, ep + 1; best_state = {k: v.clone() for k, v in net.state_dict().items()}
            log(f"  [{tag} seed {seed}] ep {ep+1}/{episodes} val={val:.2f} best={best:.2f}@{best_ep} eps={eps:.2f}", flush=True)
            if probe_csv:
                with open(probe_csv, "a") as f: f.write(f"{tag},{m},{seed},{ep+1},{val:.4f},{eps:.4f},{spent+time.time()-t0:.0f}\n")
            if resume_path and ep + 1 < episodes:
                tmp = resume_path + ".tmp"
                torch.save(dict(guard=guard, net=net.state_dict(), tgt=tgt.state_dict(), opt=opt.state_dict(), S=S, S2=S2,
                                A=Aa, C=Cc, D=D, nb=nb, ptr=ptr, eps=eps, best=best, best_state=best_state, best_ep=best_ep,
                                hist=hist, ep=ep + 1, spent=spent + time.time() - t0, rng_py=random.getstate(),
                                rng_np=np.random.get_state(), rng_t=torch.get_rng_state()), tmp)
                os.replace(tmp, resume_path)
    if resume_path and os.path.exists(resume_path): os.remove(resume_path)
    return dict(state_dict=best_state, best_val=best, best_episode=best_ep, history=hist, m=m, d=d, hidden=hidden,
                seed=seed, episodes=episodes, train_time_s=spent + time.time() - t0, tag=tag)


def dqn_policy(policy):
    net = QNet(policy["d"], policy["m"], policy["hidden"]); net.load_state_dict(policy["state_dict"]); net.eval()
    def pol(inst, o):
        s = features(inst, o)
        with torch.no_grad(): return int(net(torch.from_numpy(s)[None]).argmin(1).item())
    return pol


# ---------------------------------------------------------------- tabular Q-learning (|V| = 20 only)
def ql_key(inst, o):
    """Discretized state: (k, availability of each berth relative to now in 2-h buckets capped at 12,
    vessel class, calm-handling-time bucket, wind level at the most exposed berth)."""
    av = tuple(min(12, max(0, (o.F[b] - o.t) // 2)) for b in range(inst.m))
    hb = min(5, int(inst.h[o.v] / (RHO * inst.q[o.v]) // 4))
    wl = 0 if o.real_w < 10 else (1 if o.real_w < 20 else 2)
    return (o.v, av, inst.q[o.v], hb, wl)


def train_ql(seed, cfg, make_inst, make_wind_, train_seeds, episodes, alpha=0.1, eps_min=0.05):
    rng = random.Random(seed); Q = {}; m = int(cfg["n_berths"])
    eps_decay = eps_min ** (1.0 / (0.8 * episodes)); eps = 1.0; t0 = time.time()
    for ep in range(episodes):
        isd = train_seeds[ep % len(train_seeds)]
        inst = make_inst(isd); real = make_wind_(isd, 40_000_000 + 7919 * ep + 101 * seed, inst)
        steps = []
        def pol(o, s):
            k = ql_key(inst, o); q = Q.get(k)
            b = rng.randrange(m) if (q is None or rng.random() < eps) else int(np.argmin(q))
            steps.append((k, b)); return b
        Z, trans = episode_steps(inst, real, cfg, pol, record=True)
        for i, (k, b) in enumerate(steps):
            c = trans[i][2]
            nxt = 0.0 if i + 1 == len(steps) else float(np.min(Q.get(steps[i + 1][0], np.zeros(m))))
            q = Q.setdefault(k, np.zeros(m)); q[b] += alpha * (c + nxt - q[b])   # gamma = 1; terminal successor = 0
        eps = max(eps_min, eps * eps_decay)
    return dict(Q=Q, m=m, seed=seed, episodes=episodes, train_time_s=time.time() - t0, states=len(Q))


def ql_policy(policy):
    """Greedy tabular policy; a state never visited in training is resolved by MyopicC and counted in pol.unseen."""
    Q = policy["Q"]
    def pol(inst, o):
        q = Q.get(ql_key(inst, o))
        if q is None:
            pol.unseen += 1; return O.pol_myopic(inst, o)
        return int(np.argmin(q))
    pol.unseen = 0
    return pol
