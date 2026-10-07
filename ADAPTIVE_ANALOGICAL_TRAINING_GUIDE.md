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
   predictions from the fold model that did not train on that question;
   development questions use the final teacher. The final saved teacher need
   not reproduce the reference ranking used for each training question.
   Its loss masks absent candidates. Training includes complete
   observations, candidate/evaluator masking, individual missing measurements,
   slot permutations, and training-only hidden-source examples.
2. **Stage 2: recognition predictor.** A second ResNet uses the same observation
   representation. It predicts eight correctness values, eight SAFE memberships,
   eight MAX memberships, and global SAFE/MAX presence: 26 outputs. Its
   candidate-specific losses mask absent candidates. Development-only
   temperature calibration is fitted, then the 128-dimensional encoder,
   prediction heads, normalization state, and calibration are frozen.
3. **Stage 3: acquisition head.** A `128 → 64 → 7` network learns the next
   acquisition from the frozen hidden vector. It is supervised with masked
   cross-entropy. There is no STOP output, reward, replay buffer, or Bellman
   target. Stage 3 uses complete measurements for coherent structural states;
   hidden-source and individual missing-measurement augmentations belong to
   Stages 1 and 2 only.

SAFE is the leading uninterrupted run of actually correct candidates in the
teacher order; MAX is its first member. If rank one is wrong, both are empty.
These historical labels define offline goals and audit strata, not runtime
truth. For example, ordered correctness [correct, wrong, correct] makes only
the first candidate SAFE and MAX; [wrong, correct, correct] makes both empty.
More evidence can lower a recognition probability. Temporary loss is allowed,
but the first unfinished rank is recomputed at every state: restoring an
earlier lost goal takes priority over continuing a later goal.

## Acquisition state, actions, and cost

`State(candidate_mask, evaluator_mask)` records explicit presence. A one-shot
candidate requires its own evaluator. A Stage 3 state uses complete recorded
measurements for every present candidate/evaluator pair. Every state contains
at least one candidate; candidate-empty states are outside this contract.
The runtime begins with ZS1 and no evaluators, so its first generation precedes
the head's decisions. The zero-shot action fills the first missing slot;
the head cannot select a specific unseen answer. The evaluator action fills
the first missing evaluator in similarity order; the head cannot choose its
identity. Training still audits non-prefix presence masks from masking.

| Action index | Operation |
| --- | --- |
| `0` | Add the next missing evaluator in similarity order, its baseline, and cross evaluations against present candidates. |
| `1` | Generate another zero-shot candidate and cross-evaluate it against active evaluators. |
| `2 … k+1` | Generate the one-shot candidate from the specified active evaluator and cross-evaluate it against active evaluators. |

The default `k=5` gives seven actions. Invalid or unaffordable actions are
masked. The incremental cost is the difference between the existing state-cost
formula before and after acquisition. With five repeated solves per estimate,
the default objective counts **458 total calls**: eight generations, 25
baseline solves, 200 cross-CCS solves, and 225 corresponding graders.
`cost_unit="solver_calls"` optionally excludes those graders and gives a
233-call complete pool. Retrieval/embedding work and target-answer grading
remain outside this recorded-data accounting. There is no separate penalty
for the number of acquired candidates or evaluators.

## Exhaustive Stage 3 dataset

For every policy and development question, Stage 3 enumerates every nonempty
candidate subset and evaluator subset satisfying the one-shot prerequisite.
This is **1,912 structural states** for three zero shots and five evaluators.
It evaluates each under all six zero-shot permutations: **11,472 state/order
cases per question**. Enumeration is independent of runtime acquisition order
and budget. Cases above the budget stay in the dataset with an explicit status.

Thus **2,000 training questions produce 22,944,000 state/order cases**, before
development questions. These are exhaustive audit cases, not 22,944,000 head
training rows: every in-budget case with a valid acquisition contributes,
and identical visible observations share one row. This expansion, together with continuation search
and shard I/O, explains why Stage 3 can be slow even though the head is small.
Question count alone is not its training-set size.

The frozen Stage 2 predictor supplies recognition probabilities for each
in-budget state, and complete-state reference probabilities once per
permutation. The full state is the comparison reference; more evidence does
not guarantee that its predictions are more accurate or all goals hold.
Candidate identities, teacher ties, SAFE/MAX labels, and rank
order move consistently through permutations. The first unfinished reference
rank is the objective:

- With MAX present, rank one requires the candidate to be present and selected
  by the same correctness argmax used at runtime. Global MAX, MAX-i, global
  SAFE, and SAFE-i probabilities must all pass the configured threshold
  (0.5 by default).
- Without MAX, rank one requires candidate presence and, when
  `later_rank_filter=True`, its partial-state correctness probability must
  exceed every lower-ranked candidate's frozen full-state probability.
- Later ranks require candidate presence and, when the same filter is enabled,
  the comparison to lower-ranked full-state probabilities. A later SAFE member additionally
  requires SAFE-i at the threshold.

Search adds valid items toward a destination satisfying the current goal and
all earlier goals. Teacher order prioritizes recognition goals, not physical
generation: a lower-ranked supporting candidate may be acquired first.
Temporary loss of an earlier light is allowed. At each successor, however,
the continuation policy prioritizes its first unfinished rank, including an
earlier lost goal. Cost minimization follows those continuation decisions;
it is not an unrestricted shortest-path search holding the original goal
fixed. A legal route to that goal can therefore be cheaper than the chosen
recovery route, or succeed where the chosen route fails.

Shared visible observations within a question across zero-shot scenarios must
choose the same action at every search depth. Only newly observed evidence
may separate their continuations. Their hidden orders are treated as
equiprobable. The objective is lexicographic: maximize average goal reach,
then minimize average additional calls, then average acquisition steps. Costs
include unsuccessful continuations up to action exhaustion. On a remaining
tie, adding an evaluator takes priority over generating a candidate. Other
genuine ties split the soft target evenly; the planner uses the lowest action
index as its concrete tied continuation. A shared choice need not be cheapest
or successful for every hidden case.

When the available actions have zero average reach under the constructed
continuation policy, the head still gets an acquisition target so it can
operate through the full pool. This fallback chooses the next
similarity-ranked evaluator whenever affordable, even if a candidate is
cheaper; otherwise it chooses the cheapest valid candidate, with action index
breaking ties. This evaluator-first fallback is stronger than the ordinary
objective's tie preference. It does not claim the unresolved goal was achieved
or prove that no other legal route could reach it.

Every case has exactly one disposition, with this precedence:

| Status | Meaning | Head loss |
| --- | --- | --- |
| `OUT_OF_BUDGET` | The state itself exceeds the configured budget. | Excluded |
| `COMPLETE` | All reference recognition goals hold; this requires the complete candidate and evaluator pool. | Excluded. |
| `EXHAUSTED` | The full pool is acquired and a goal remains unfinished. | Excluded |
| `BUDGET` | The pool is incomplete and no affordable action remains. | Excluded |
| `ACTION` | A shared continuation has positive average reach for the current goal. | Masked cross-entropy |
| `UNREACHABLE` | Actions remain, but the current goal has zero average reach under the constructed continuation policy; other legal paths may exist. | Fill-pool fallback target. |

Cases without a valid acquisition have a null training-row reference.
Identical visible observations within a question share one training row.
Each row stores the frozen hidden vector, valid mask, normalized soft target,
goal-rank distribution, and per-action reach/cost/steps for diagnosis. All
ordinary Stage 3 rows have equal loss weight, regardless of how many original
state/order cases they represent. Thus exhaustive case coverage does not mean
one equally weighted training example per case. Terminal and over-budget
cases remain audit records, with no STOP or abstention target. Offline rank
and historical truth never enter the head input.

## Training, artifacts, and evaluation

All three stages train on exactly the same question group: `supervised`.
The historical `policy` role is an alias containing the same IDs in the same
order, so it no longer takes a separate fraction of the training data.

`use_test_files_for_dev_and_audit=True` is enabled by default in the module
and editable Kaggle configuration. It uses every eligible question from
`train_file` for training and pools every eligible question from `test_files`
for both `dev` and `audit`. Exact normalized-text duplicates across files are
still excluded. Internal `split_fractions` are ignored in this mode.

| Stage | Weight training | Development checks |
| --- | --- | --- |
| 1: teacher | Shared `supervised` questions | `dev`: final teacher checkpoint selection and heuristic selection |
| 2: snapshot encoder and prediction heads | Same `supervised` questions | Same `dev`: checkpoint selection and temperature calibration |
| 3: acquisition head | Same `supervised` questions, through the `policy` alias | Same `dev`: acquisition-head checkpoint selection |

The teacher's fold models still use internal training/validation folds and
produce out-of-fold labels for all shared training questions. The final
teacher trains on all shared training questions. The snapshot predictor
stays frozen while the acquisition head trains.

Stage 2 selects its checkpoint by development correctness Top-1, then Brier
score; Stage 3 selects by development action cross-entropy. These criteria do
not directly optimize recognition-goal coverage or realized API savings.

Set `use_test_files_for_dev_and_audit=False` for separate internal development
and audit groups. `split_fractions=(0.80, 0.10, 0.10)` then means **shared
training / dev / audit**. All stages use the same 80%; `policy` is still an
alias. External test logs are used only for reporting in this mode. The old
four-fraction configuration is no longer used.

With test-file development enabled, audit and external benchmark scores are
on questions also used for checkpoint selection and calibration. They are
not an independent held-out test. `evaluation_protocol` in both
`split_manifest.json` and `results.json` records the development/audit overlap
and the external files used for model selection. Final reports include the
pooled audit set and each benchmark separately.

Stage 3 writes atomic question-sized shards in
`decision_shards/policy` and `decision_shards/dev`, with
`decision_dataset_manifest.json` and `decision_label_coverage.json`. The
manifest and shards bind to the config, split, teacher labels, and frozen
predictor fingerprint. An interrupted build resumes completed matching
questions; incompatible shards fail explicitly. A role with no actionable
rows fails with its coverage report. During training, policy question and row
order are shuffled deterministically each epoch, and dev questions are read in
stable order for masked-loss checkpoint selection. All exhaustive labels and
frozen hidden vectors are constructed **once before the head's epoch loop**;
epochs reuse those rows and never regenerate states, planner targets, or
encoder features. A new invocation validates matching shards before reuse.
The frozen Stage 2 state is checked after training.

Stage 3 prints question progress, elapsed time, and estimated remaining time
for each role's label build and each epoch's training/development pass. Updates
appear on the first/last question, every `print_every` questions, or after 30
seconds at the next question boundary. Each epoch also prints both losses and
early-stopping patience. The `Stage 3 policy: ...` coverage summary means policy
labels are complete; development labels come next, before head training.
With the default external-development setting, all four benchmark logs supply
development questions, and their exhaustive label build can be substantial.
Keep the same output directory and configuration to reuse completed shards
after interruption. After every completed training/development epoch,
`decision_head_progress.pt` atomically saves current and best head weights,
Adam state, question/row shuffle RNG, losses, and early-stopping bookkeeping.
With `resume=True`, head optimization continues from the last completed epoch;
an interrupted partial epoch repeats from its start. Failed checkpoint writes
preserve the previous completed epoch. Older interrupted runs without this
progress file begin head training at epoch 1. Keep the training and development
settings and selected device consistent when comparing a resumed run to an
uninterrupted run. The final selected head is saved when Stage 3 finishes.

Audit shards are now gzip-compressed losslessly at level 1. This preserves
every row, array dtype/value, state/order case, and fingerprint; it does not
reduce coverage or change labels. `load_checkpoint` detects compressed and
legacy uncompressed shards. Matching legacy shards are validated and then
atomically compressed in place during resume. Packed training companions and
public model checkpoints remain ordinary PyTorch files. Use this workflow's
`load_checkpoint` rather than bare `torch.load` for compressed audit shards.

A `torch.save` iostream error followed by `unexpected pos` is a checkpoint
write failure, commonly caused by exhausted disk space or a storage quota.
It is not evidence of a GPU-memory failure. Stage 3 prints the output path
and free filesystem space, removes known incomplete shard `.pt.tmp` files
left by old interrupted runs, and preserves completed files on failed writes.
Write errors report the affected path, free space, and original exception;
free filesystem space alone does not rule out a quota or I/O problem.

On Kaggle, check `shutil.disk_usage(CFG.output_dir)` and free unneeded files
if the output filesystem is full. Compression still needs enough temporary
space to write one shard, and packed companions consume additional space.
To recover an interrupted **revision-5** run, use the same `output_dir`, input
files, and training configuration with `resume=True`. Revision-4 runs cannot
resume under the new action and label contract; start revision 5 in its new
default output directory.

The Stage 3 implementation reuses budget-specific structure/action tables
across questions, constructs observations in NumPy batches, reuses each
scenario's observations for both inference and visible-state grouping, and
evaluates the scalar goal rules in arrays.
It still enumerates every state/order case and uses shared continuation
decisions and masked training. The evaluator action and tie rule changed in
revision 5, so older label shards cannot be reused.

After validating each full shard, it writes a compact `.training.pt` companion
containing only `h`, `target`, and `valid` tensors. Epochs read these companions
instead of unpickling the cases and audit metadata repeatedly. Matching
companions are validated and reused; missing or stale companions are rebuilt
from validated source rows. Matching revision-5 schema-3 shards and manifests
remain usable with the same configuration and output directory. Training transfers
one question's tensors to the device
once, preserves the existing shuffled minibatch order and optimizer steps,
and reads the accumulated loss back once per question.

`decision_cache_mb=4096` sets a 4 GiB requested packed-tensor CPU cache cap,
further limited to half the currently available host RAM. If available RAM
cannot be detected, the cache is disabled. It retains memory-mapped tensors
for a role only if every question in that role fits in the remaining budget,
considering policy first. Other roles stream one memory-mapped packed question
at a time. The OS pages these tensors on demand; the cache does not eagerly
clone the entire dataset into RAM. Whole-role admission avoids repeatedly
filling and evicting a partial cache when question order changes each epoch.
`decision_cache_mb=0` disables the cache. This is a runtime setting: changing
it does not invalidate matching experiment artifacts. The cap covers retained
tensor payloads, not the entire process; records, one question's audit
metadata, model state, and temporary arrays also need RAM. Every row remains
available in either mode. Only one question's training tensors occupy GPU
memory at a time; compact companions require additional disk space.

`decision_eval_batch_size=4096` uses larger development-only forward batches
for head evaluation. Every development row still contributes to masked loss,
and training minibatches and optimizer steps stay unchanged. Floating-point
roundoff can produce tiny development-loss differences and affect checkpoint
selection or early stopping when losses are nearly tied. This is also a
runtime setting; set it to `128` to use the earlier development batch size.
Keep it unchanged across an interrupted run when preserving its checkpoint
selection behavior matters.

On Kaggle, enable an available GPU in **Settings > Accelerator** and keep
`device="auto"`. The selected device runs the frozen encoder's batched
inference during labeling and the head's forward/backward passes. Observation
assembly, state enumeration, continuation search, grouping, and shard
serialization still run on the CPU; enabling CUDA does not move Python search
or file I/O to the GPU. The Stage 0 notebook cell reports the selected device,
available GPUs, memory, and CPU threads so the active session can be checked.
Kaggle offers GPU options such as T4 x2 and P100; this workflow uses one
selected device and does not automatically combine two GPUs' memory.
See [NVIDIA's Kaggle setup instructions](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/).

The default `decision_batch_size=128` retains the same optimizer updates and
shuffle order. For a new CUDA experiment, try `decision_batch_size=1024` to
reduce the number of small head updates, then compare development loss and
checkpoint quality. It keeps every training row but changes update count,
gradient grouping, and the optimization trajectory; use a **new `output_dir`**
when changing it. It does not accelerate label construction. The default
head is too small to assume that adding distributed workers, mixed precision,
or a second GPU improves total runtime. Reduced device transfers and loss
readbacks follow the synchronization guidance in
[PyTorch's performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

The following benchmark results are **historical revision-4 measurements**.
They do not validate the new revision-5 labels or runtime policy. The old
`benchmark_adaptive_stage3.py` compared against commit `1034634` and wrote
`stage3_full_dataset_performance_report.json`, checks every row field and
exhaustive case for exact equality, checks training losses/weights, checks
development loss within floating-point roundoff, and measured label
generation, shard loading, and head training/development separately. Do not
use its old exact-parity assertion as a revision-5 validation gate.

The recorded local CPU comparison uses one CPU thread and five synthetic
questions with the default state geometry. Median label construction fell
from **1.774 s to 0.701 s (2.53x)**; development forwards fell from **0.005934 s
to 0.002930 s (2.03x)**. All exhaustive cases, row fields, and batch-128 training
losses/weights matched exactly. Pure head optimization at the unchanged
training batch size remained about **0.024-0.025 s**, so that measurement does
not show a training-update speedup. These fixture measurements do not predict
the runtime of the actual logs or Kaggle hardware.

`stage3_local_cuda_smoke_report.json` records a separate synthetic regression
on a local RTX 3050 Ti laptop GPU with 4 GiB memory. It verified exact case,
label, and batch-128 head-update parity. Peak PyTorch allocated memory was
about **24 MiB** for the comparison fixture, including both predictor replicas
and heads; CUDA context, reserved allocator memory, and other processes are
outside that number. Its single-pass timings include first-use overhead and
are not representative speedup measurements. This was not a Kaggle/T4 run.

Offline reports compare the complete-pool baseline, configurable fixed
sequences, and learned acquisitions. Defaults for five evaluators and three
zero-shot candidates are:

| Method | Fixed acquisition order |
| --- | --- |
| `fixed` (`interleaved`) | ZS1, R1, OS1, ZS2, R2, OS2, ZS3, R3, OS3, R4, OS4, R5, OS5 |
| `fixed_evaluators_first` | ZS1, R1, R2, R3, R4, R5, ZS2, ZS3, OS1, OS2, OS3, OS4, OS5 |
| `fixed_zero_shots_first` | ZS1, ZS2, ZS3, R1, OS1, R2, OS2, R3, OS3, R4, OS4, R5, OS5 |

Edit `fixed_acquisition_orders` in the Kaggle configuration cell to select
these built-ins or add a method such as
`"fixed_custom": ["ZS1", "R1", "OS1", "ZS2", "R2", "OS2", "ZS3"]`.
`Rj` adds the next evaluator at retrieval rank j; `OSj` generates its one-shot
candidate. Every sequence starts with the already generated `ZS1`, and
unseen zero-shots arrive in slot order. Duplicates, out-of-range items, and
one-shots before their evaluators and out-of-order retrieval are rejected. A custom sequence may be a
partial pool; ending it produces `fixed_sequence_exhaustion` rather than
claiming successful recognition. Fixed order stops at its next unaffordable
scheduled action and never skips ahead.

Stage-3 training labels cover acquisition decisions through the complete
pool, including states after MAX recognition. Evaluation checks the four
MAX/SAFE recognition signals after the initial ZS1 and after every
acquisition. Both fixed and learned evaluation rollouts stop at the first
predicted MAX, then compare top-1 correctness and API calls with the
complete-pool reference. The historical MAX label scores the stop but never
causes it. If no MAX is recognized, acquisition continues until the budget
or available actions end. The full-pool reference acquires the complete
pool even when it exceeds the adaptive budget.

An application wrapper can choose its own stopping point. A direct learned
`rollout(...)` uses `continue_after_max=True` by default: it records the
first MAX recognition and continues until the pool is acquired or the budget
prevents another action. Its `ranked_list_complete` reason means **acquisition
complete**, even if MAX/SAFE lights or later recognition goals fail. It differs
from the dataset's `COMPLETE` status, which requires all recognition goals.
The result also exposes `acquisition_complete` and `final_max_recognized`
separately, so callers can distinguish pool completion from current MAX
recognition without interpreting the legacy reason string. Full-pool reference
runs record recognition at their observed full state as well.
Pass `continue_after_max=False` to stop that call at predicted MAX, or change
the configuration default. This runtime-only choice does not change the
Stage-3 labels or training fingerprints. With the default full-pool budget,
a continuing call has no API savings against the full reference.

The returned `ranked_candidates` list sorts present candidates by their current
Stage 2 correctness probabilities, and `selected` is their correctness argmax.
Stage 1 defines offline reference goals; it does not define this runtime list.
The inference bundle contains the frozen Stage 2 predictor and action head,
not the Stage 1 teacher. These two rankings may disagree.

Continuation diagnostics keep the best-ever prefix in
`mean_rank_prefix_reached` and `per_rank_reach`. They separately report the
ending prefix in `mean_final_rank_prefix` and `per_rank_final_recognition`,
all-goal recognition in `all_goals_recognized_rate`, and acquired-pool
completion in `acquisition_complete_rate`. These are averaged over zero-shot
orders within each question, then equally across questions. An earlier
recognition peak does not establish that those goals still hold at the end.

The additional API-savings table has two rows per method: **all questions**
and **MAX-present questions** (nonempty historical teacher MAX labels).
It shows question counts, mean API calls used, mean calls saved per question
versus the full pool, savings percentage, and exact MAX recovery. Savings
include failed/incorrect stops; they are not conditional on successful MAX
recovery. `first_max_recognition` and `max_recognition_rate` also record when
recognition occurs. JSON `policies[method].call_savings` stores both
`solver_calls` and `total_calls`, expected total savings across questions,
predicted-MAX stopping rates, and wrong predicted-MAX stopping rates.

The visible table also divides saved calls into **True MAX/q**, **Wrong
MAX/q**, and **Other/q**. True MAX savings require a predicted-MAX stop
that returns the historical MAX candidate. Wrong MAX savings come from an
incorrect predicted-MAX stop, including questions with no historical MAX.
Other savings come from budget limits or an ended custom sequence. These
three contributions add up to total saved calls per question; each uses the
whole row's question group as its denominator. Thus even a budget-limited
run that happens to return MAX is counted under Other, not True MAX.

A compact **predicted-MAX stops only** table makes the other meaning of
"MAX enabled" explicit: the prediction gates actually triggered the stop.
For both all questions and reference-MAX-present questions, it shows the
MAX-stop rate, mean calls used/saved per such stop, and the fraction of those
stops that returned the correct reference MAX. Incorrect stops are included.
Question weights are equal before conditioning on the stop event, and all
zero-shot orders are averaged within each question. When no MAX stop occurs,
conditional savings and precision are unavailable (`null`/`--`). This differs
from overall savings, which also include runs ending at a budget or sequence
limit. The JSON keys are `predicted_max_stop_precision` and each cost unit's
`mean_used_on_predicted_max_stop`, `mean_saved_on_predicted_max_stop`, and
`saved_fraction_on_predicted_max_stop`.

Total API calls include candidate generations, repeated baseline/cross
measurement solves, and their grading calls. With default settings the
complete pool uses 233 solver calls and 225 graders: **458 API calls**.
Retrieval and target-answer grading are excluded. This accounting follows
the recorded-pool simulator, not observed live traffic. Costs and savings
are averaged across zero-shot orders within each question before aggregating,
so a question is counted once. Empty MAX-present cohorts have null metrics.
Trajectories name each acquired evaluator/candidate and its incremental cost.
Every prediction check, including the initial ZS1 and the final state, records
available candidate IDs, the selected candidate, the four MAX/SAFE signals,
and the threshold. These make the reason for stopping inspectable.

Fixed comparison orders affect evaluation only. They are excluded from
training/checkpoint fingerprints, so matching revision-5 checkpoints can
be reused when changing orders and rerunning reports.

Reports keep answer correctness, exact reference-MAX recovery, no-MAX results,
costs and savings, and temporary loss of earlier recognition goals separately.
They also expose exhaustive status counts, actionable coverage by target
rank, evaluator masks, and action names. The run saves teacher/snapshot/head
checkpoints, an inference bundle, trajectories, and `results.json`.

This implementation uses **revision 5**, **contract/inference schema 4**, and
**decision-dataset schema 3**. Older action/state bundles and checkpoints are
rejected. Use the new default output directory
`/kaggle/working/adaptive_analogical_shared_training_v5_run` for a fresh run; old runs are
not migrated. The notebook's `QUICK_PILOT` checks execution, not final model
quality. Recorded outcomes cannot establish live provider behavior or the
results of newly generated candidates.
