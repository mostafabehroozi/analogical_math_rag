# Adaptive analogical training: three supervised stages

The executable workflow is `adaptive_analogical_training.py`. The Kaggle
notebook embeds the same definitions. Both use recorded Layer-1 run logs;
training and offline evaluation make no provider calls.

## Evidence and models

Each default question has three zero-shot candidates, five retrieved examples,
and one one-shot candidate generated from each example. Retrieval order is
descending similarity, with log order breaking ties. Baseline and cross-CCS
measurements come from the recorded log. The neural observation contains
measurement values and presence masks, never question text, answer identity,
teacher ranks, correctness labels, or future measurements.

1. **Stage 1: reference ranking.** A ResNet reads 223 observation features and
   predicts correctness for eight candidates. Complete-state predictions define
   the teacher's deterministic ranking. Supervised questions use out-of-fold
   predictions. Its loss masks absent candidates. Training includes complete
   observations, candidate/evaluator masking, individual missing measurements,
   slot permutations, and training-only hidden-source examples.
2. **Stage 2: recognition predictor.** A second ResNet uses the same observation
   representation. It predicts eight correctness values, eight SAFE memberships,
   eight MAX memberships, and global SAFE/MAX presence: 26 outputs. Its
   candidate-specific losses mask absent candidates. Development-only
   temperature calibration is fitted, then the 128-dimensional encoder,
   prediction heads, normalization state, and calibration are frozen.
3. **Stage 3: acquisition head.** A `128 → 64 → 11` network learns the next
   acquisition from the frozen hidden vector. It is supervised with masked
   cross-entropy. There is no STOP output, reward, replay buffer, or Bellman
   target. Stage 3 does not use the Stage 1/2 hidden-source augmentation.

SAFE is the leading uninterrupted run of actually correct candidates in the
teacher order; MAX is its first member. If rank one is wrong, both are empty.
These historical labels define offline goals and audit strata, not runtime
truth. More evidence can lower a recognition probability, so the planner
checks previously achieved goals at every intermediate state.

## Acquisition state, actions, and cost

`State(candidate_mask, evaluator_mask)` records explicit presence. A one-shot
candidate requires its own evaluator. A Stage 3 state uses complete recorded
measurements for every present candidate/evaluator pair. The runtime begins
with ZS1 and no evaluators. The zero-shot action fills the first missing slot;
the head cannot select a specific unseen answer. Evaluator identities remain
aligned with recorded similarity order.

| Action index | Operation |
| --- | --- |
| `0 … k-1` | Add the specified missing evaluator, its baseline, and cross evaluations against present candidates. |
| `k` | Generate another zero-shot candidate and cross-evaluate it against active evaluators. |
| `k+1 … 2k` | Generate the one-shot candidate from the specified active evaluator and cross-evaluate it against active evaluators. |

The default `k=5` gives eleven actions. Invalid or unaffordable actions are
masked. The incremental cost is the difference between the existing state-cost
formula before and after acquisition. With five repeated solves per estimate,
the default full pool costs **233 solver calls**: eight generations, 25
baseline solves, and 200 cross-CCS solves. `cost_unit="total_calls"` also
counts evaluator grading according to that formula.

## Exhaustive Stage 3 dataset

For every policy and development question, Stage 3 enumerates every nonempty
candidate subset and evaluator subset satisfying the one-shot prerequisite.
This is **1,912 structural states** for three zero shots and five evaluators.
It evaluates each under all six zero-shot permutations: **11,472 state/order
cases per question**. Enumeration is independent of runtime acquisition order
and budget. Cases above the budget stay in the dataset with an explicit status.

The frozen Stage 2 predictor supplies recognition probabilities for each
in-budget state, and complete-state reference probabilities once per
permutation. Candidate identities, teacher ties, SAFE/MAX labels, and rank
order move consistently through permutations. The first unfinished reference
rank is the objective:

- With MAX present, rank one requires the candidate to be present and global
  MAX, MAX-i, global SAFE, and SAFE-i probabilities all at least the configured
  recognition threshold (0.5 by default).
- Without MAX, rank one requires the candidate's partial-state correctness
  probability to exceed every lower-ranked candidate's frozen full-state
  probability.
- Later ranks require candidate presence and, by default, the same comparison
  to lower-ranked full-state probabilities. A later SAFE member additionally
  requires SAFE-i at the threshold.

Search adds valid items while preserving every earlier goal at **each step**.
A path that temporarily loses one is rejected even if later evidence restores
it. Shared visible observations across zero-shot scenarios must choose the
same action at every search depth. Only newly observed evidence may separate
their continuations. The objective is lexicographic: maximize goal-reaching
probability, then minimize expected additional cost, then minimize expected
acquisition steps. Genuine ties split the soft target evenly; a concrete
tied continuation uses the lowest action index. Once a rank is reached, the
successor state's target addresses the next unfinished rank.

Every case has exactly one disposition, with this precedence:

| Status | Meaning | Head loss |
| --- | --- | --- |
| `OUT_OF_BUDGET` | The state itself exceeds the configured budget. | Excluded |
| `COMPLETE` | All reference recognition goals hold. | Excluded |
| `EXHAUSTED` | The full pool is acquired and a goal remains unfinished. | Excluded |
| `BUDGET` | The pool is incomplete and no affordable action remains. | Excluded |
| `ACTION` | An admissible continuation can reach the current goal. | Masked cross-entropy |
| `UNREACHABLE` | Actions remain, but no admissible path reaches the goal. | Excluded |

Non-action cases have a null training-row reference and no fabricated target.
Identical visible observations within a question share one actionable row.
Each row stores the frozen hidden vector, valid mask, normalized soft target,
goal-rank distribution, and per-action reach/cost/steps for diagnosis. All
ordinary Stage 3 rows have equal weight. Offline rank and historical truth
never enter the head input.

## Training, artifacts, and evaluation

The disjoint roles are `supervised` (60%), `policy` (20%), `dev` (10%), and
`audit` (10%). Stage 3 writes atomic question-sized shards in
`decision_shards/policy` and `decision_shards/dev`, with
`decision_dataset_manifest.json` and `decision_label_coverage.json`. The
manifest and shards bind to the config, split, teacher labels, and frozen
predictor fingerprint. An interrupted build resumes completed matching
questions; incompatible shards fail explicitly. A role with no actionable
rows fails with its coverage report. During training, policy question and row
order are shuffled deterministically each epoch, one shard is read at a time,
and dev shards are read in stable order for masked-loss checkpoint selection.
The frozen Stage 2 state is checked after training.

Offline reports compare the complete-pool baseline, the fixed sequence
`ZS1 → R1 → OS1 → ZS2 → R2 → OS2 → ZS3 → R3 → OS3 → R4 → OS4 → R5 → OS5`,
and learned acquisitions. Fixed order stops at its next unaffordable scheduled
action. Learned and fixed rollouts otherwise use the same application rule:
stop when global MAX/SAFE and the selected candidate's MAX-i/SAFE-i signals
reach the threshold, or no affordable action remains. This runtime rule is
separate from offline Stage 3 labels.

Reports keep answer correctness, exact reference-MAX recovery, no-MAX results,
costs and savings, and preservation of earlier recognition goals separate.
They also expose exhaustive status counts, actionable coverage by target
rank, evaluator masks, and action names. The run saves teacher/snapshot/head
checkpoints, an inference bundle, trajectories, and `results.json`.

This implementation uses **revision 3**, **contract/inference schema 4**, and
**decision-dataset schema 3**. Older action/state bundles and checkpoints are
rejected. Use the new default output directory for a fresh run; old runs are
not migrated. The notebook's `QUICK_PILOT` checks execution, not final model
quality. Recorded outcomes cannot establish live provider behavior or the
results of newly generated candidates.
