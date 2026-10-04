# Adaptive analogical training: three supervised stages

The executable workflow is `adaptive_analogical_training.py`. The Kaggle
notebook embeds the same definitions. Both use recorded Layer-1 run logs;
training and offline evaluation make no provider calls.

## Kaggle GPU execution

Enable **Settings > Accelerator > GPU** before running the notebook. Leave
`CFG.device="auto"`: all three stages select CUDA when PyTorch can see a GPU,
and fall back to CPU when none is available. Each stage prints its selected
device. Set `device="cuda"` if you want training to stop with a clear error
when a GPU is unavailable; `device="cpu"` explicitly selects CPU.

Stage 1 trains every fold teacher and the final teacher on the selected device.
Stage 2 trains the snapshot predictor there. Stage 3 runs the frozen encoder
and trains the acquisition head there. JSON loading, observation construction,
continuation planning, and shard I/O remain CPU operations, so Stage 3 can
remain CPU-bound between GPU forwards. One selected GPU is used; multiple
Kaggle GPUs do not pool their memory. Completed checkpoints resume rather than
retrain, and can move between CPU and GPU without changing the data contract.

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

Thus **2,000 training questions produce 22,944,000 state/order cases**, before
development questions. These are exhaustive audit cases, not 22,944,000 head
training rows: only `ACTION` cases contribute, and identical visible
observations share one row. This expansion, together with continuation search
and shard I/O, explains why Stage 3 can be slow even though the head is small.
Question count alone is not its training-set size.

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
To recover an existing run, use the updated notebook definitions with exactly
the same `output_dir`, input files, and training configuration, including
`resume=True`. Rerun configuration and Stage 0, then
`WORK.train_decision_head()`: it loads completed Stages 1/2 and resumes matching
shards. Do not create a fresh output directory for this storage-only update.

The Stage 3 implementation reuses budget-specific structure/action tables
across questions, constructs observations in NumPy batches, reuses each
scenario's observations for both inference and visible-state grouping, and
evaluates the scalar goal rules in arrays.
It still enumerates every state/order case and keeps the same tie rules,
shared continuation decisions, and masked training objective.

After validating each full shard, it writes a compact `.training.pt` companion
containing only `h`, `target`, and `valid` tensors. Epochs read these companions
instead of unpickling the cases and audit metadata repeatedly. Matching
companions are validated and reused; missing or stale companions are rebuilt
from validated source rows. Existing schema-3 shards and manifests remain
usable with the same configuration and output directory. Training transfers
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

For a reproducible **synthetic CPU** comparison against commit `1034634`, run
`python -B benchmark_adaptive_stage3.py`. It writes
`stage3_full_dataset_performance_report.json`, checks every row field and
exhaustive case for exact equality, checks training losses/weights, checks
development loss within floating-point roundoff, and measures label
generation, shard loading, and head training/development separately. This is
not evidence of a live Kaggle/GPU speedup or a full-run runtime estimate.

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
`"fixed_custom": ["ZS1", "R2", "OS2", "ZS2", "R1", "OS1", "ZS3"]`.
`Rj` adds the evaluator at retrieval rank j; `OSj` generates its one-shot
candidate. Every sequence starts with the already generated `ZS1`, and
unseen zero-shots arrive in slot order. Duplicates, out-of-range items, and
one-shots before their evaluators are rejected. A custom sequence may be a
partial pool; ending it produces `fixed_sequence_exhaustion` rather than
claiming successful recognition. Fixed order stops at its next unaffordable
scheduled action and never skips ahead.

Learned and fixed rollouts use the same application rule after the initial
ZS1 and after **every single evaluator or candidate acquisition**:
stop when global MAX/SAFE and the selected candidate's MAX-i/SAFE-i signals
reach the threshold, or no affordable action remains. This runtime rule is
separate from offline Stage 3 labels. A historical MAX label is never read
to decide whether to stop. The full-pool reference acquires the complete
pool without early stopping, even when it exceeds the adaptive budget.

The additional API-savings table has two rows per method: **all questions**
and **MAX-present questions** (nonempty historical teacher MAX labels).
It shows question counts, mean API calls used, mean calls saved per question
versus the full pool, savings percentage, and exact MAX recovery. Savings
include failed/incorrect stops; they are not conditional on successful MAX
recovery. JSON `policies[method].call_savings` stores these cohorts, both
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
Every stopping check, including the initial ZS1 and the final state, records
available candidate IDs, the selected candidate, the four MAX/SAFE signals,
and the threshold. These make the reason for stopping inspectable.

Fixed comparison orders affect evaluation only. They are excluded from
training/checkpoint fingerprints, so matching revision-4 checkpoints can
be reused when changing orders and rerunning reports.

Reports keep answer correctness, exact reference-MAX recovery, no-MAX results,
costs and savings, and preservation of earlier recognition goals separate.
They also expose exhaustive status counts, actionable coverage by target
rank, evaluator masks, and action names. The run saves teacher/snapshot/head
checkpoints, an inference bundle, trajectories, and `results.json`.

This implementation uses **revision 4**, **contract/inference schema 4**, and
**decision-dataset schema 3**. Older action/state bundles and checkpoints are
rejected. Use the new default output directory
`/kaggle/working/adaptive_analogical_shared_training_v4_run` for a fresh run; old runs are
not migrated. The notebook's `QUICK_PILOT` checks execution, not final model
quality. Recorded outcomes cannot establish live provider behavior or the
results of newly generated candidates.
