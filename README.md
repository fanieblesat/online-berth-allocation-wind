# LSPS / BAP under wind — experiment scripts (C&OR paper; DESIGN.md v2, 19 Oct 2026)

Online berth allocation under time-dependent, berth-specific wind: exact perfect-information optima (CP-SAT free
sequencing; label-setting DP with normalization, dominance and bounding for the arrival-ordered variant), and online
policies (static plan, rolling DP on the forecast, dispatching rules, tabular QL, DQN) measured against them.
Every number in the paper comes from `results/runs.csv`, `results/exact.csv` and `results/agreement.txt`.

| step | command | writes |
|---|---|---|
| 0 | `python3 0_agreement.py --workers 8` | `results/agreement.txt` (must end with ALL AGREE) |
| dry | `python3 dry_run.py` | `results/dry_run_<host>.txt` (timings and campaign estimate) |
| queue | `python3 make_queue.py`, then `queue_A.txt` and `queue_B.txt` with `xargs -P` (below) | the whole campaign |
| 1 | `python3 1_exact_scaling.py --point V20 [--seeds S]` (V30, V40, V50): DP vs CP-SAT-AO | `results/exact.csv` |
| 2 | `python3 2_train.py --dqn --berths B --seed k` (B = 2..5, k = 0..4); `python3 2_train.py --ql --seed k` | `results/policy_*.pkl`, `train_curves.csv` |
| 3 | `python3 3_benchmark.py --point P --methods PI-AO,Static,RollDP,FirstFree,MyopicC,LowExposure,DQN` | `results/runs.csv` |
| 3 | `python3 3_benchmark.py --point P --seeds S --methods PI --cp-limit 600 --workers 1` (main points: one DP solve logged as PI-AO, then CP-SAT-FREE hinted with it, logged as PI-FREE with its bound and the no-wait bound) | `results/runs.csv` |
| rows | `python3 paper_rows.py results/runs.csv > results/paper_rows.tex` | table rows |
| tables | `python3 build_tables.py --out <overleaf>/tables` | the manuscript's result tables |
| figures | `python3 4_figures.py --out <overleaf>` | fig_decomp, fig_voi, fig_exact, fig_train, fig_sens (Section 6) |

Points: main grid V20, V30, V40, V50 (|B| = |V|/10); sensitivity at V30: mu5/15/20, sig1.5/5.5/8.0, theta0.06/0.24,
alpha0.25/0.75, B4/B5 (ratio), load0.6/0.95, kappa0.5/2.0 (common.py). Every driver is resumable; DQN training is
checkpointed every validation step (restart = same command). One thread per job; run as many jobs as cores minus one.

Campaign: `python3 make_queue.py` writes `queue_A.txt` (387 jobs; the main points split one job per test instance,
longest first) and `queue_B.txt` (23 jobs, the learned-policy evaluations), and creates the CSV headers. Run
`queue_A`, then `queue_B`, each with `setsid nohup sh -c 'xargs -P <P> -I{} sh -c "{}" < queue_X.txt' > results/queue_X.log 2>&1 < /dev/null &`.

## Post-hoc extension (DESIGN.md §9, 23 Oct 2026)
| step | command | writes |
|---|---|---|
| test | `python3 tests_lookahead.py` | window DP = AO DP check (expect 0 mismatches) |
| queues | `python3 make_queue_C.py --look 5` | queue_C1.txt (networks), queue_C2.txt (test-pool runs) |
| C1 | `xargs -P 3 -I{} sh -c "{}" < queue_C1.txt` | results/vnet_B{m}_seed{s}.pkl, results/look_train.csv |
| C2 | `xargs -P 3 -I{} sh -c "{}" < queue_C2.txt` (after C1) | runs.csv rows RollDP-T, LookDP-0, LookDP-R, LookDP-RV |

Results of the extension (25 Oct 2026):

| step | command | writes |
|---|---|---|
| tables | `python3 build_tables.py --results results --out <overleaf>/tables` | also tab_online, tab_pilot (needs look_pilot*.csv) |
| figures | `python3 4_figures.py --results results --out <overleaf>` | also fig_scale |
| numbers | `python3 ext_numbers.py --results results > results/ext_numbers.txt` | numbers quoted in the text |
| diagnostic | `python3 6_correction_r2.py --results results` | results/correction_r2.csv (post hoc) |

Literature baseline (DESIGN.md §10, 25 Oct 2026):

| step | command | writes |
|---|---|---|
| test | `python3 tests_ube.py` | results/tests_ube.txt (must end with ALL PASS) |
| queue | `python3 make_queue_D.py --methods UBE-0.95 --points V50,V40,V30,V20` | queue_D.txt (80 jobs) |
| D | `xargs -P 3 -I{} sh -c "{}" < queue_D.txt` | runs.csv rows UBE-0.95 |
| pilot | `python3 7_ube_pilot.py --point V20` (and V30) | results/ube_pilot.csv (validation, descriptive) |
