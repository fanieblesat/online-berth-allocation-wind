"""make_queue_C.py — the post-hoc extension of DESIGN.md §9 as two job lists for `xargs -P`:
  queue_C1.txt  cost-to-go networks: 4 berth counts x 5 seeds (5_lookahead.py --train)
  queue_C2.txt  RollDP-T, LookDP-0, LookDP-R and LookDP-RV at every point of the campaign (3_benchmark.py)
Run C2 only after C1 has finished. Every job is resumable; jobs at V40/V50 are split per test instance.
Usage: python3 make_queue_C.py --look K"""
import argparse
from common import MAIN, SENS, TEST_SEEDS, TRAIN_RNG_SEEDS
ap = argparse.ArgumentParser(); ap.add_argument("--look", type=int, required=True); a = ap.parse_args()
ENV = "CUDA_VISIBLE_DEVICES= "
C1 = [f"{ENV}python3 5_lookahead.py --train --berths {m} --seed {s} > results/vt_B{m}_{s}.log 2>&1" for m in (2, 3, 4, 5) for s in TRAIN_RNG_SEEDS]
C2 = []
for p in ["V50", "V40"]:
    for s in TEST_SEEDS:
        C2.append(f"{ENV}python3 3_benchmark.py --point {p} --seeds {s} --methods RollDP-T > results/rt_{p}_{s}.log 2>&1")
        C2.append(f"{ENV}python3 3_benchmark.py --point {p} --seeds {s} --look {a.look} --methods LookDP-0,LookDP-R,LookDP-RV > results/lk_{p}_{s}.log 2>&1")
for p in ["V30", "V20"] + SENS:
    C2.append(f"{ENV}python3 3_benchmark.py --point {p} --methods RollDP-T > results/rt_{p}.log 2>&1")
    C2.append(f"{ENV}python3 3_benchmark.py --point {p} --look {a.look} --methods LookDP-0,LookDP-R,LookDP-RV > results/lk_{p}.log 2>&1")
open("queue_C1.txt", "w").write("\n".join(C1) + "\n"); open("queue_C2.txt", "w").write("\n".join(C2) + "\n")
print(len(C1), "jobs in queue_C1.txt,", len(C2), "in queue_C2.txt")
