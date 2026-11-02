"""make_queue_D.py — queue of the literature-baseline test runs (DESIGN.md §10): one job per (point, test instance,
method), three realizations each, largest points first. Resumable: 3_benchmark.py skips logged runs.
Usage: python3 make_queue_D.py --methods UBE-0.95[,UBE-0.5] [--points V50,V40,V30,V20,mu20,sig8.0]  -> queue_D.txt"""
import argparse
from common import TEST_SEEDS
ap = argparse.ArgumentParser(); ap.add_argument("--methods", required=True)
ap.add_argument("--points", default="V50,V40,V30,V20,mu20,sig8.0"); a = ap.parse_args()
jobs = [f"python3 3_benchmark.py --point {p} --methods {m} --seeds {s} > results/d_{p}_{m}_{s}.log 2>&1"
        for p in a.points.split(",") for s in TEST_SEEDS for m in a.methods.split(",")]
open("queue_D.txt", "w").write("\n".join(jobs) + "\n")
print(f"queue_D.txt: {len(jobs)} jobs")
