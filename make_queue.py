"""make_queue.py — writes the campaign as two job lists for `xargs -P`:
  queue_A.txt  everything that does not need a trained policy (exact scaling, bounds, rules, Static, RollDP, training)
  queue_B.txt  DQN and QL evaluations (run after queue_A has finished)
Main points are split into one job per test instance (3 realizations each), so no single job runs for more than a
few hours; jobs are ordered longest first (V50, V40, training, then the rest). Every job is resumable.
Usage: python3 make_queue.py ; then
  setsid nohup sh -c 'xargs -P <P> -I{} sh -c "{}" < queue_A.txt' > results/queue_A.log 2>&1 < /dev/null &"""
import csv
from common import MAIN, SENS, TRAIN_RNG_SEEDS, TEST_SEEDS, RESULTS, RUNS, FIELDS
ENV = "CUDA_VISIBLE_DEVICES= "          # CPU only; silences the CUDA driver warning
A, B = [], []
for p in ["V50", "V40", "V30", "V20"]:
    for s in TEST_SEEDS:
        A.append(f"{ENV}python3 3_benchmark.py --point {p} --seeds {s} --methods PI --cp-limit 600 --workers 1 > results/pi_{p}_{s}.log 2>&1")
        A.append(f"{ENV}python3 3_benchmark.py --point {p} --seeds {s} --methods RollDP > results/r_{p}_{s}.log 2>&1")
        A.append(f"{ENV}python3 3_benchmark.py --point {p} --seeds {s} --methods Static,FirstFree,MyopicC,LowExposure > results/b_{p}_{s}.log 2>&1")
        A.append(f"{ENV}python3 1_exact_scaling.py --point {p} --seeds {s} --workers 1 > results/x_{p}_{s}.log 2>&1")
    if p == "V40":   # training right after the two largest points
        for m in (2, 3, 4, 5):
            for k in TRAIN_RNG_SEEDS:
                A.append(f"{ENV}python3 2_train.py --dqn --berths {m} --seed {k} > results/t_B{m}_{k}.log 2>&1")
        for ep in (16000, 1000):
            for k in TRAIN_RNG_SEEDS:
                A.append(f"{ENV}python3 2_train.py --dqn --berths 3 --seed {k} --episodes {ep} > results/t_B3_{k}_ep{ep}.log 2>&1")
        for k in TRAIN_RNG_SEEDS:
            A.append(f"{ENV}python3 2_train.py --ql --seed {k} > results/q_{k}.log 2>&1")
for p in SENS:
    A.append(f"{ENV}python3 3_benchmark.py --point {p} --methods PI-AO,Static,FirstFree,MyopicC,LowExposure > results/b_{p}.log 2>&1")
    A.append(f"{ENV}python3 3_benchmark.py --point {p} --methods RollDP > results/r_{p}.log 2>&1")
for p in MAIN + SENS:
    B.append(f"{ENV}python3 3_benchmark.py --point {p} --methods DQN > results/d_{p}.log 2>&1")
for ep in (1000, 16000):
    B.append(f"{ENV}python3 3_benchmark.py --point V30 --methods DQN --episodes {ep} > results/d_V30_ep{ep}.log 2>&1")
B.append(f"{ENV}python3 3_benchmark.py --point V20 --methods QL > results/q_V20.log 2>&1")
# headers written once here, so parallel jobs never race on them
if not RUNS.exists():
    with open(RUNS, "w", newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()
EX = RESULTS / "exact.csv"
if not EX.exists():
    with open(EX, "w", newline="") as f:
        csv.writer(f).writerow(["point", "instance", "dp_status", "dp_obj", "dp_labels", "dp_peak", "dp_time",
                                "ao_status", "ao_obj", "ao_bound", "ao_time"])
open("queue_A.txt", "w").write("\n".join(A) + "\n"); open("queue_B.txt", "w").write("\n".join(B) + "\n")
print(len(A), "jobs in queue_A.txt,", len(B), "in queue_B.txt")
