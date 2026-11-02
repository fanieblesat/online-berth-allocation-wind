# Audit record — manuscript CAIE-D-26-05446 vs. the experimental code (11 Oct 2026)

Scope: the submitted manuscript (`main_jos.tex`, 827 lines) against the only experimental code found on
the personal laptop (`bap_core.py`, `methods.py`, `experiment_1..4_*.py`, `analyze_results.py`, `README.md`,
`results/experiment_1_tabular_validation.csv`). The office and work laptops are to be checked for newer
versions; until then this record assumes none exist. Purpose: an accurate statement of what the
submitted paper's claims rested on, for the supervisor and for the rebuild. Nothing here is reused.

| # | Manuscript states | Code does | Severity |
|---|---|---|---|
| 1 | §6: DQN over state (A_s, T_s), actions = all unassigned (vessel, berth) pairs with masking, output layer |V|·|B| (320 at 40v-8b); the 320-action space is given as the cause of the 40-vessel breakdown | `tabular_q` and `dqn_direct` both use the arrival-order MDP: state = next vessel + berth times, actions = the |B| berths; output layer |B| | Central explanation of the paper has no implementation (R2 point 1; editor point 1) |
| 2 | §7.1: instances "built on real vessel-call data from HHLA/Eurogate sailing lists; smaller instances drawn directly from these records" | `generate_instance`: uniform random arrivals, class-dependent uniform handling times, lengths, priorities; no data file read anywhere | Factual claim about data without implementation |
| 3 | §5.1/§7.1: three-state Markov weather chain (Clear/Moderate/Severe, δ = 0/0.15/0.40) with a stated transition matrix; 100 trajectories per evaluation | four i.i.d. sea states (×1.00/1.15/1.30/1.55, p = .50/.25/.15/.10); one state drawn per sample and applied to every vessel of the whole schedule; no chain, no per-period weather | Weather model in the text is not the one run; the "19–22 % weather variance" is the expected handling-time inflation of this i.i.d. draw |
| 4 | §7.1: exact solver Gurobi 11.0 via gurobipy; Table 3 column "Proven 5/5, 5/5, 5/5, 5/5, 1/5" | PuLP + CBC; the code records no optimality status (PuLP returns status 1 at the time limit with an incumbent); no source for "proven" | Factual claim about solver and proofs without implementation |
| 5 | Table 2 / §7.1: DQN 256×256, lr 5e-4, buffer 50 000, target update every 100 steps, gradient clip 10, episodes 2 000 / 5 000 / 10 000 | 128×64, lr 1e-3, buffer 10 000, target update every 200 episodes, clip 1.0, 2 000 episodes at every size | Reported configuration is not the one run |
| 6 | DQN and QL results (all tables) | both report the best trajectory encountered during ε-greedy training; no greedy-policy evaluation exists for the DQN; trained and evaluated on one instance, one seed, no test instances | DQN numbers are effectively a 2 000-sample random search; R2 point 3 applies to the DQN as well as to QL |
| 7 | §5.2, §7.9: emission penalty c_e, emission rates, sensitivity study (Table 8) | not present in any file; the objective is total turnaround only | Section without implementation |
| 8 | §7.1: GA "follows standard settings"; GA systematically beaten by FCFS at 40v | population 100 × 100 generations, random initial population, OX on vessel ids, no local search, deterministic-calm fitness, best of 3 runs | Baseline not fit for purpose (R1, editor point 3) |
| 9 | Table 3: 25 instances | the CSV on the laptop is a `--quick` run (2 instances per size, halved episodes, MIP limits ÷ 10); the 25-instance table came from a run not found on this laptop | Provenance of the published table unknown |
| 10 | QL implementation | Q-update treats a successor with all-zero Q-values as terminal (`target = reward`); γ = 0.99 rather than the shortest-path γ = 1 of Prop. 2; berth times rounded to 0.25 h | Method not the one the proposition describes; contributes to the 11–41 % greedy-policy gaps in the CSV |
| 11 | README.md (shipped with the code) | states "expected findings" and "paper impact" before the experiments were run | Narrative written ahead of evidence |

Items 2, 4 and 7 are statements of fact about what was done that the code does not support. Items 1, 3, 5,
6 and 10 are descriptions of methods that differ from the methods run. Items 8, 9 and 11 are quality and
provenance failures. The reviewers found 1, 3 (partly), 6 (partly) and 8.

Consequence: the manuscript is not revisable; the paper is rebuilt from new code under the protocol of
DESIGN.md, with the research question retained and the method of the JoS revision applied (code first,
frozen protocol, every number from a seeded run, text written from the record).
