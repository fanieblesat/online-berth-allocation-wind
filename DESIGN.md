# Paper II (LSPS / BAP) — design v2 (19 Oct 2026), frozen before any code or run

Target: Computers & Operations Research, regular submission, end of January 2027.
Preliminary version: IC_TORS'26 (LNOR), "A Graph-Based Learning Shortest-Path Framework for the Berth Allocation
Problem" — cited as such. Previous journal manuscript rejected at TRB (TRB-D-26-00454) and C&IE (CAIE-D-26-05446);
AUDIT.md records why nothing from its experiments is reused. v1 of this design (comparison study) is kept as
DESIGN_v1_superseded.md.

House rules (as JoS and QCSP): code first and exactly as written here; protocol frozen before the campaign; every number
from a logged, seeded run; text written from the record; deviations declared in the paper and in CHANGELOG.md.

---------------------------------------------------------------------------------------------------------------------
## 1. Research question and contributions

**Question.** In the discrete berth allocation problem with time-dependent, berth-specific wind (productivity loss,
berth closures, mooring delay) whose realization is revealed period by period, how much turnaround does an online
decision-maker lose relative to the perfect-information optimum, how much of that loss is the value of information and
how much is the policy, what does the arrival-order rule cost, and where do learned policies stand in this decomposition?

**Contributions** (as they will be stated):
1. *Exact perfect-information optima at realistic sizes* under time-dependent wind, for free sequencing (CP-SAT with
   start-dependent durations) and for the arrival-ordered variant (label-setting DP on the IC_TORS decision graph).
2. *A dominance result that makes the arrival-ordered DP exact and fast*: under the wind model the completion time of a
   vessel is non-decreasing in its ready time at a berth when mooring delays satisfy phi_{b,t} <= 1 + phi_{b,t+1}
   (always true here, phi in {0,1} at admissible berthings); hence availability vectors can be (a) normalized to the
   next arrival and (b) Pareto-pruned on (availability vector, cost so far) without loss of optimality. Without the
   condition, optimal delayed berthing restores monotonicity (stated and proved for the general case).
3. *An exact loss decomposition for online policies*:
     Z(pi) - Z*_free = [Z*_AO - Z*_free]          cost of the arrival-order rule (perfect information)
                     + [Z(RollDP) - Z*_AO]        loss of the best tested online policy = upper bound on the value of
                                                  wind information within the arrival-ordered class
                     + [Z(pi) - Z(RollDP)]        policy loss of pi relative to that benchmark
   reported for dispatching rules, the static plan, rolling DP, tabular Q-learning and DQN, at every size.
4. *Learning measured, not sold*: greedy-policy evaluation only, held-out instances, five seeds, model selection on a
   validation pool, training budget as a sensitivity axis; claims limited to the tested architectures.

**Not claimed:** a new exact method for the deterministic DBAP (Kramer, Lalla-Ruiz, Iori & Voß, EJOR 2019 solve
it exactly to 150x15 with free sequencing); a new learning architecture; the classical DP/RL equivalence (IC_TORS and
Bertsekas & Tsitsiklis 1991 — stated as background, not as a result).

**Positioning to verify in the literature pass (before the text):** Kramer et al. 2019 (EJOR 278(1)); Nishi, Okura,
Lalla-Ruiz & Voß (Ann Oper Res 2020, DP-based matheuristic); Lalla-Ruiz et al. (EJOR 250(3), 2016, time-dependent
limitations / tides); Correcher et al. (EJOR 275(1), 2019); Umang, Bierlaire & Erera (J Sched 2017, real-time BAP);
"Algorithm Design for an Online Berth Allocation Problem" (competitive analysis; [[VERIFY authors/venue]]); RL-for-BAP
papers (Çolak, Heilig & Voß survey 2025; MDPI DQN-BAP 2024); time-dependent scheduling / FIFO literature.

---------------------------------------------------------------------------------------------------------------------
## 2. Problem

- Discrete berths b in B, one vessel at a time per berth; vessels v in V with announced arrival periods A_v (known in
  advance: ETAs are announced; the uncertainty studied is the wind) and workload h_v (moves).
- Fixed class gang (decided): feeder q_v = 1, seagoing q_v = 2 cranes; no crane decisions. All berths admit all
  vessels (stated explicitly; answers C&IE R1 on berth sizes).
- Hourly periods. Wind: the JoS Ornstein–Uhlenbeck process W_t (mu_w, sigma_w, theta_w), berth exposure beta_b,
  berth wind beta_b W_t. Storm S_bt = [beta_b W_t >= WLIM]: no berthing and no work. Productivity mu_bt = mu(beta_b W_t)
  (JoS, continuous piecewise linear, ALPHA = loss at WLIM). Mooring delay phi_bt in {0, 1} periods at berthing.
- Berthing rule: a vessel ready at berth b in period r (r = max(A_v, release of the previous vessel at b)) berths at the
  first non-storm period tau >= r; moors phi_{b,tau} periods; then receives q_v * RHO * mu_bt moves per period; released
  in the first period in which cumulative work >= h_v; C_v = release period. Objective Z = sum_v (C_v - A_v).
- Variants: AO (arrival order: vessels at the same berth are served in arrival order) and FREE (any order per berth).

## 3. Information and decisions (the online protocol; answers C&IE R2 points 1-2)

- Online methods decide one berth per vessel at the vessel's arrival period A_v (AO variant), observing: all A, h
  (announced), the wind realized up to A_v at every berth, the current backlog of every berth (in-service vessel's
  remaining work, queued vessels), and the forecast: the OU conditional mean E[W_{t+k} | W_{A_v}] = mu_w + (W - mu_w)
  (1 - theta_w)^k. Decisions are irrevocable; berthing and handling then follow the realized wind (simulator).
- Perfect-information methods see the whole realized trajectory (bounds, not policies).
- Static plan: berth assignment fixed at time 0 by the AO DP on the forecast from W_0, executed without recourse.

## 4. Methods

| Row | Kind | Definition |
|---|---|---|
| PI-free | bound | CP-SAT, optional intervals per (v,b), start-dependent duration via element constraint, NoOverlap per berth, storm starts excluded; time limit 3600 s; proven optimum or bound reported |
| PI-AO | bound | label-setting DP with normalization + Pareto dominance on the realized wind (exact) ; also CP-SAT-AO (FREE model + same-berth arrival-order precedences) as a cross-check |
| Static | policy | AO DP on the forecast from period 0; berths fixed; executed on the realization |
| RollDP | policy | at every arrival, AO DP from the current state on the forecast; take its berth for v |
| FirstFree | rule | berth whose forecast release of its backlog is earliest (ties: lower exposure) |
| MyopicC | rule | berth minimizing v's own forecast completion |
| LowExposure | rule | among berths whose forecast release <= A_v + slack, the least exposed; else FirstFree |
| QL | learned | tabular, gamma = 1, state = (k, availability vector relative to A_k bucketed to 1 h and capped, wind level bucket at decision); n = 20 only (tabular beyond that is infeasible — stated); greedy evaluation |
| DQN | learned | state: v's workload & class, per berth (forecast release - A_v, beta_b W_{A_v}, forecast storm share next 12 h), k/|V|; |B| actions; double DQN 2x128, Huber, Adam 5e-4 cosine to 5e-5, buffer 1e5, batch 64, target sync 20 episodes, eps 1 -> 0.05; trained per size on the training pool, 4000 episodes, checkpoint every 200 on the validation pool; five seeds |

## 5. Instances

- Hamburg record (JoS `make_instance`): inter-arrival gaps and (class, H0) resampled; workload h_v = RHO q_v H0
  (calm handling time = H0 hours).
- Sizes (confirmed 19 Oct): |V| in {20, 30, 40, 50} with |B| = |V|/10 in {2,3,4,5}.
  The 10:1 ratio is kept; |V| = 10 is dropped because |B| = 1 leaves no decision. Gaps scaled by 3/|B| so that the
  per-berth arrival rate is the same at every size (JoS reference: 3 berths); the dry run reports berth utilization.
  Exposures beta: 2 berths (1.0, 0.4); 3 (1.0, 0.7, 0.4); 4 (1.0, 0.8, 0.6, 0.4); 5 (1.0, 0.85, 0.7, 0.55, 0.4).
- Pools (disjoint seeds): train 5000+, validation 4000-4009, test 1000-1019 (20 per point), exact-check 3000-3009.
  Three wind realizations per test instance; common random numbers across methods.
- Sensitivity (reference |V| = 30, |B| = 3): mu_w in {5, 10, 15, 20}; sigma_w in {1.5, 3.5, 5.5, 8.0}; persistence
  theta_w in {0.06, 0.12, 0.24}; productivity loss ALPHA in {0.25, 0.5, 0.75}; ratio at |V| = 30: |B| in {3, 4, 5}
  (10, 7.5, 6); arrival clustering: gaps x {0.5, 1, 2}; DQN training budget {1000, 4000, 16000} episodes.

## 6. Evaluation protocol

- Every method on the same 20 test instances x 3 realizations per point (CRN). Paired Wilcoxon on (instance,
  realization) means, paired Cohen's d, wins, TOST +-1% for "no difference", Friedman/Nemenyi per size; per-seed rows
  for DQN; training time, inference time per decision, per-decision compute of RollDP.
- The decomposition of Section 1 per point with 95% intervals over instances; PI-free reported with its proven status
  (gap to bound where unproven).
- Agreement checks (0_agreement.py, exact-check pool, small sizes): DP-AO = CP-SAT-AO optimum; simulator replay of every
  exact solution reproduces its objective; FIFO check of the completion function on every instance (assert).

## 7. Code (to be written now; one driver per table)

```
lsps/scripts/
  common.py, data/ (Hamburg record), lsps_core/{model.py (instance, wind, completion function, simulator),
  dp.py (label-setting DP + dominance + normalization), exact.py (CP-SAT FREE/AO), online.py (state, forecast,
  policies, harness), learners.py (QL, DQN with checkpoints/resume)}
  0_agreement.py  1_exact_scaling.py  2_train.py  3_benchmark.py  4_sensitivity.py  dry_run.py  paper_rows.py  figures.py
```

## 8. Decisions (19 Oct 2026)

1. CP-SAT with and without the arrival-order rule — yes (now: PI-free vs PI-AO, a reported term of the decomposition).
2. Class-dependent fixed gang — yes.
3. Static plans: the clairvoyant static plan is now the PI bound itself; the blind/forecast static plan is the "Static" row.
4. Ratio 10:1 — kept; size grid {20, 30, 40, 50} x {2, 3, 4, 5} berths, per-berth arrival rate constant (confirmed 19 Oct).
5. Title: working title to be revised to the new question, e.g. "How much does an online berth planner lose to the
   wind? Exact bounds, the value of information and learned policies" — fixed when the text is written.
Framing: online problem + exact bounds (decided after Kramer et al. 2019 was found).

---------------------------------------------------------------------------------------------------------------------
## 9. Post-hoc extension (23 Oct 2026, after the campaign): rolling DP with bounded lookahead and learned cost-to-go

Written before any pilot or test run of the new method; declared in the paper as an extension made after the campaign
(all campaign results stand unchanged). Motivation from the record: RollDP is the best online policy but is intractable
at 40-50 vessels (20% / 48% MyopicC fallbacks at 60 s per decision; a 300-s pilot changed nothing material); learned
policies that map the state to a berth (DQN, QL) do not beat the dispatching rules.

**Method (lsps_core/lookahead.py).** At the arrival of vessel v the AO problem of the next K vessels is solved exactly on
the forecast from the current berth-release vector (window DP; every distinct availability vector kept, no dominance,
because the terminal cost need not be monotone). Each terminal label is scored by its accumulated turnaround plus a
terminal cost T for the remaining vessels; the first berth of the best label is taken (ties: the less exposed berth).
- LookDP-0: T = 0 (a K-vessel myopic DP) — isolates what the terminal cost adds.
- LookDP-R: T = turnaround of MyopicC on the forecast from the terminal state (deterministic rollout of the base policy).
- LookDP-V: T = (n - j) x g(x), g a neural network estimating the expected turnaround per remaining vessel of the base
  policy MyopicC from features x of the terminal state seen at time t (berth releases, berth exposure, forecast
  productivity/storm share over 24 h, remaining workload, next arrivals, forecast lead time). This is one step of
  approximate policy iteration (rollout with a learned cost-to-go; Bertsekas); the cost-improvement property of exact
  rollout is cited, not claimed for the approximation.
- Exactness check: with K >= n - v and T = 0 the window optimum equals the AO DP optimum at every decision
  (tests_lookahead.py, V20 validation instances).

**Training of g.** Per berth count m (as the DQN): n = 10 m, reference wind and load, the 200 training instances,
2 realizations each (wind seeds 50, 51); at every arrival the base policy is simulated on the forecast of that moment
for k = 1..6 further vessels and each predicted state is a sample whose target is the realized turnaround per remaining
vessel of the base policy from that vessel on. MLP 2x128, MSE, Adam 1e-3, batch 512, early stopping (patience 20) on
the 10 validation instances (realization 0). Five network seeds (0-4) per m, as for the DQN.

**Choice of K (frozen before any test run).** Pilot on the VALIDATION pool only (10 instances, realization 0) at V30
and V40: K in {2, 3, 4, 5} for LookDP-0, -R and -V (network seed 0) with MyopicC as reference. One K for all points:
the smallest K whose LookDP-V mean is within 0.5% of the best K's mean at both sizes, and whose time per decision is
at most 10 s at V50 (measured on one V50 validation instance). The pilot table goes into the CHANGELOG and the paper.

**Evaluation.** 3_benchmark.py --methods LookDP-0,LookDP-R,LookDP-V on all 20 points, 20 test instances x 3
realizations, LookDP-V with the five network seeds (reported as their mean, per-seed spread). Comparisons and tests as
§6: LookDP-V against RollDP (and, where it has a lower mean, its effect on the information bound, which it then
tightens), against the best rule, and LookDP-V against LookDP-0 and LookDP-R (what learning adds). No second policy-
iteration step and no other variant unless declared here before it is run.

### 9.1 Frozen after the validation pilots (23 Oct 2026, before any test-pool run of the new methods)
Validation pool only (10 instances, realization 0; results/look_pilot*.csv). Total turnaround (h):

| | MyopicC | LookDP-0 K=5 | LookDP-R K=4 / 5 / 6 / 7 | LookDP-RV K=5 | s per decision, LookDP-R K=5 |
|---|---|---|---|---|---|
| V30 | 5,324 | 5,070 | 5,026 / 4,947 / 4,943 / 4,943 | 5,000 | 0.02 |
| V40 | 6,203 | 5,875 | 5,798 / 5,733 / 5,786 / 5,781 | 5,794 | 0.15 |
| V50 | 7,586 | 7,210 | 7,042 / 6,973 / 7,026 / --- | 7,271 | 0.71 |

At V30 also: RollDP 4,970, RollDP-T 4,892 (exact tie search and the single-DP version agree), Monte Carlo rollout of
MyopicC over 20 OU scenarios 4,947. The learned cost-to-go alone (LookDP-V) was unstable at V30 (4,959-5,072 over K) and
is dropped; the learned correction (LookDP-RV) is kept as the ablation that measures what learning adds to the rollout.
Amendment to §9 made before any test run: the K rule is applied to the method that is proposed, LookDP-R (not -V).

**Frozen:** K = 5 at every point (best LookDP-R K at V40 and V50, within 0.1% of the best at V30; 0.71 s per decision at
V50). Methods on the test pool: RollDP-T (60-s decision cap, as RollDP), LookDP-0, LookDP-R and LookDP-RV (network seeds
0-4) with K = 5, all 20 points x 20 instances x 3 realizations. Networks: 5_lookahead.py --train for m = 2..5, seeds
0-4 (2 realizations per training instance). Claims to be tested: LookDP-R vs RollDP and RollDP-T (all sizes), vs the best
rule; LookDP-RV vs LookDP-R (the value of the learned correction); RollDP-T vs RollDP and its effect on the information
bound (Delta_info is recomputed with RollDP-T where it has the lower mean; both are reported).

### 9.2 Record of the test runs (25 Oct 2026, after queue C2 finished)
Record: results/runs.csv on laptop-fni, 23,952 rows = the 14,352 campaign rows (unchanged, byte-identical) + 9,600
extension rows (RollDP-T, LookDP-0, LookDP-R 1,200 each; LookDP-RV 6,000 = 5 network seeds), all status ok.
Analyses as frozen in §9.1 (build_tables.py: tab_online, tab_decomp with Delta_info for RollDP, RollDP-T and LookDP-R;
tab_sens, tab_time, tab_pilot; 4_figures.py: fig_scale; ext_numbers.py: every other quoted number).
Post-hoc, declared as such in the paper: 6_correction_r2.py (share of variance of the two learning targets explained by
the trained networks on the validation samples), written after the test runs to explain the LookDP-RV result; it does
not enter any comparison. A Monte Carlo rollout of MyopicC tried during the V30 validation pilot (§9.1) was a probe
without a logged script; the paper mentions it only as an informal, unlogged probe (Sec. 7) and reports no number from it.
Deviations from §9 found by the audit of draft 4 (25 Oct 2026) and declared in the paper (Sec. 6.2): the pilot used
K in {4..7} (V50: {4,5,6}) instead of the written grid {2..5}; RollDP-T and LookDP-RV were added during the validation
pilot and were declared only in §9.1 (before any test run), not before their first run as §9 required. The K=n check
of tests_lookahead.py is logged in results_final/tests_lookahead.txt (100 decisions, 0 mismatches).

## 10. Literature baseline (written 25 Oct 2026, before any code of it was run)

**Source.** Umang, Bierlaire & Erera (2017), "Real-time management of berth allocation with stochastic arrival and
handling times", Journal of Scheduling 20(1):67-83 (technical report TRANSP-OR 130215, 2013, read in full). Their
optimization-based real-time recovery was the best of their three approaches in all mild-congestion cases and is the
method reimplemented here ("UBE"). Their smart and traditional greedy rules are not reimplemented.

**Faithful elements.** (1) A baseline schedule at the start: the free-sequencing (set-partitioning) optimum on planning
values. (2) Planning values: the rho_h-quantile of the uncertain duration, Prob(h <= h^t) = rho_h, rho_h = 0.95 as in
their experiments. (3) Re-optimization of all vessels not yet berthed, free sequencing (each berth any order, berthing
time free), triggered when an actual handling time is revealed (a completion; their arrival-update trigger never fires
here because arrivals are announced and exact). (4) Berthed vessels keep their berth, which is blocked until the
planning value of their completion. (5) Execution: a vessel berths at its planned berth when the berth is free, the
vessel has arrived and its planned berthing time is reached, not earlier. (6) Objective: the service cost of the
re-optimized vessels, sum (C_v - A_v) (their Z1); comparison against an a-posteriori optimum.

**Adaptations (declared in the paper).** (a) The uncertainty is the wind, so the planning value of vessel v at berth b
when ready in period r is the rho-quantile of its completion C_b(v, r) over S = 100 OU sample paths conditioned on the
wind realized up to the event period t (inverted-CDF quantile); paths drawn from numpy default_rng([instance seed,
realization, t, 77]), independent of the realized wind. For a berthed vessel the quantile of its completion from its
realized progress at t. (b) Their Z2/Z3 terms (quay shifting, deviation from the baseline schedule) are omitted: the
evaluation criterion here is total turnaround, berths are discrete, and their Set-1 experiments set c1 = c2 = 0.
(c) Solver: CP-SAT with one worker on the time-indexed model of Section 4.2 (element constraints on the quantile
tables, NoOverlap per berth), the equivalent of their set partitioning; time limit 600 s for the baseline schedule
(as the Static plan) and 60 s per re-optimization (as RollDP per decision); complete hint = the previous plan
replayed from the current state (the baseline: MyopicC on the quantile tables), whose value bounds the objective;
if CP-SAT returns no solution the hint is executed (counted). (d) UBE may re-sequence vessels that have arrived but
not berthed, so it is not an arrival-ordered policy: it lies in the larger class bounded by Z*_FREE and is compared
with the other policies by realized total turnaround; it does not enter the AO decomposition.

**Checks before the pilot (tests_ube.py).** (i) Vectorized quantile completion equals the scalar completion() for
every sample, berth and start on random cases. (ii) With S = 1 and the realized wind as the only sample (perfect
information) and no time limit, on the exact-check instances (6-12 vessels), the baseline schedule's value equals the
PI-FREE optimum and the executed schedule realizes it (no re-optimization changes it). (iii) Non-anticipation: the
decisions up to period t are unchanged when the realized wind after t is replaced.

**Validation pilot (descriptive, then frozen).** Validation pool (10 instances, realization 0) at V20 and V30:
rho in {0.5, 0.95}, run time per run. Rule fixed now: UBE-0.95 (faithful) goes to the test pool; UBE-0.5 goes too if its
V30 validation total is lower than UBE-0.95's. No other parameter is tuned.

**Test.** Main grid V20-V50 and the strong/volatile-wind points mu20 and sig8.0, 20 instances x 3 realizations, common
random numbers, same machine as the campaign. If the pilot timing extrapolates to more than 120 core-hours, mu20 and
sig8.0 are dropped (rule fixed now). Comparisons: UBE vs LookDP-R, RollDP, RollDP-T and the best rule (Wilcoxon, TOST
+-1%), and UBE's gap to the PI-FREE incumbent and bound.

### 10.1 Amendment before the pilot and before any test run (25 Oct 2026)
Run time measured in the first pilot runs: UBE uses its full 60 s at almost every re-optimization (CP-SAT does not
prove the free-sequencing optimum), about 17 min per run at V20 (rho 0.95) and up to about 60 min at V50: about 170
core-hours per setting on the main grid. Decided with the author: (1) the test pool gets UBE-0.95 only (the faithful
setting), on V20-V50; mu20 and sig8.0 are dropped (by the 120-core-hour rule of §10). (2) The validation pilot
(rho in {0.5, 0.95}, V20 and V30, 10 instances) is descriptive only and selects nothing; it is reported with the
results. (3) Code change before the pilot: the CP-SAT hint now also covers the interval sizes and the absent
alternatives (complete hint); the hint was always feasible (checked with fixed hinted values). Pilot rows produced
with the incomplete hint (V20: UBE-0.5 on instance 4000, Z 352; UBE-0.95 on instance 4000, Z 364; references) were
seen and set aside (results/ube_pilot_discarded_incomplete_hint.csv); the pilot restarts from scratch. Memory: about
0.8 GB per job at V50.

### 10.2 Test record (29 Oct 2026)
The test queue (queue_D.txt, 80 jobs: UBE-0.95 on V20-V50, 20 test instances x 3 realizations) completed on the
campaign machine: 240 runs, no malformed rows, no CP-SAT failure (hint never executed), code md5 ube.py
6072d9be15c7446c15cd198d55e3a0e4, 3_benchmark.py 031707f74a166bfe73fd919e5291bcb3 (as tested). Non-UBE rows of the
uploaded runs.csv are identical to results_final/runs.csv (23,952 rows, max |dZ| = 0). Core time 112 h.
Deviation, recorded: after the main-grid results had been seen, a queue for mu20 and sig8.0 was started on 29 Oct by
mistake (the assistant overlooked §10.1(1)) and stopped on the same day. Any rows it produced are excluded from the
record (results_d5/runs.csv keeps UBE rows of V20-V50 only) and are not reported.
Pilot as reported: complete-hint pilot stopped by the 2-h limit of the cloud environment (not the campaign machine);
both rho on V20 instances 4000-4003 and V30 4000-4001; rho 0.5 lower by 2.7% (V20) and 4.1% (V30); under the original
§10 rule rho 0.5 would have been tested; reported in the paper (Sec. 6.2).
