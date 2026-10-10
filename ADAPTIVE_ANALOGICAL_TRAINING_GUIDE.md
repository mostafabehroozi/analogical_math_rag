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
and identical visible observations share one row. This expansion, together with continuation search,
explains why Stage 3 label construction can be slow even though the head is small.
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
Each row has a visible state key, valid mask, normalized soft target,
goal-rank distribution, and per-action reach/cost/steps for diagnosis. Only
the head's inputs are saved (see below); the diagnostic fields are recomputed
on demand. All
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

Stage 3 writes `decision_labels/policy.pt` and `decision_labels/dev.pt`, with
`decision_dataset_manifest.json` and `decision_label_coverage.json`. The
manifest and label files bind to the config, split, teacher labels, and frozen
predictor fingerprint. While a role builds, each completed question is one
small atomic file in `decision_labels/<role>/`; an interrupted build resumes
completed matching questions, and the finished role is merged into one file.
Incompatible files fail explicitly. A role with no actionable rows fails with
its coverage report. During training, policy question and row order are
shuffled deterministically each epoch, and dev questions are read in stable
order for masked-loss checkpoint selection. All exhaustive labels are
constructed **once before the head's epoch loop**; epochs reuse those rows and
never regenerate states or planner targets. Frozen hidden vectors are not
stored: each training and development pass rebuilds every row's observation
and re-encodes it with the frozen predictor on the selected device. A new
invocation verifies saved labels before reuse. The frozen Stage 2 state is
checked after training.

Stage 3 prints question progress, elapsed time, and estimated remaining time
for each role's label build and each epoch's training/development pass. Updates
appear on the first/last question, every `print_every` questions, or after 30
seconds at the next question boundary. Each epoch also prints both losses and
early-stopping patience. The `Stage 3 policy: ...` coverage summary means policy
labels are complete; development labels come next, before head training.
With the default external-development setting, all four benchmark logs supply
development questions, and their exhaustive label build can be substantial.
Keep the same output directory and configuration to reuse completed questions
after interruption, including in a new Kaggle session when `hf_repo` mirrors
the folder (see "Surviving Kaggle session limits"). After every completed
training/development epoch,
`decision_head_progress.pt` atomically saves current and best head weights,
Adam state, question/row shuffle RNG, losses, and early-stopping bookkeeping.
With `resume=True`, head optimization continues from the last completed epoch;
an interrupted partial epoch repeats from its start. Failed checkpoint writes
preserve the previous completed epoch. Older interrupted runs without this
progress file begin head training at epoch 1. Keep the training and development
settings and selected device consistent when comparing a resumed run to an
uninterrupted run. The final selected head is saved when Stage 3 finishes.

### Surviving Kaggle session limits

Kaggle keeps `/kaggle/working` only while a session lives, and a session ends
after at most 12 hours. Every stage already completes through atomic files in
`output_dir`, so the run becomes resumable across sessions once that folder is
mirrored. Set `hf_repo` to a private Hugging Face **dataset** repository id
such as `user/adaptive-run` (runtime-only, outside the contract and the label
fingerprint). The token comes from the `HF_SYNC_TOKEN` or `HF_TOKEN`
environment variable, the Kaggle Secret of the same name (attach it to the
notebook under Add-ons > Secrets, with Internet enabled), or an existing
`huggingface_hub` login; it is never printed or stored.

- **Pull.** Constructing `Workflow(CFG)` downloads the mirrored run into an
  empty output folder (a non-empty folder is used as is). The repository is
  created on the first upload if it does not exist.
- **Push.** Stage 0 ends with a required upload, so a bad token, repository
  id, or disabled Internet fails in the first minute. Each later stage uploads
  when it completes. Inside long loops the run uploads every `hf_sync_minutes`
  (default 15): after Stage 1 folds, during the Stage 3 label build, and after
  Stage 3 epochs; each merged role uploads immediately. Uploads skip unchanged
  files by content hash, never include `.tmp` files, and remove per-question
  label files and migrated shards from the mirror once they are gone locally.
  A failed periodic upload prints a warning and retries at the next sync
  point; local files are never discarded. `WORK.push()` uploads on demand,
  and `WORK.sync.describe()` reports the upload state.
- **Resume.** In a new session, download the same input logs, keep the same
  configuration, and run the cells. Inputs are identified by size and
  SHA-256, so a new path or mtime keeps the same run (contracts written by
  earlier code, which recorded an mtime, are upgraded in place when they still
  match). Stages 0-2 load their completed checkpoints; Stage 1 also reuses
  completed `teacher_fold_<n>.pt` and `teacher_final.pt` files whose question
  roles match exactly, so an interrupted Stage 1 repeats at most one fold.
  Stage 3 reuses completed questions and continues from the last completed
  epoch. Stage 2 and Stage 4 restart if interrupted; neither takes long
  compared with Stage 3.
- **One session per repository.** Two sessions writing the same mirror would
  overwrite each other; use a different `hf_repo` for a different run
  (`QUICK_PILOT` appends `_pilot` to both the output folder and the mirror).

### Stage 3 label storage

Stage 3 saves only what the head trains on (label-file schema 5):

- `preferred`, one value per row: a bitmask of the row's tied optimal actions
  (one byte for the default seven actions).
- `first_cases`, one value per question: a bitmap over its in-budget
  state/order cases (1,434 bytes for the default 6 x 1,912 cases) marking the
  first case mapped to each row.

Rows are stored in the planner's order: larger structures (acquired candidates
plus evaluators) first, ties in case order. That order is a function of the
bitmap, so each row's source (`scenario * in-budget states + state`) is rebuilt
from it rather than stored. A synthetic default-geometry question with about
8,000 rows takes about **12 KB** (about 1.5 bytes per row with the bitmap and
file overhead), and 2,000 policy questions plus the pooled test-file development
questions take roughly 40 MB. Everything else is a deterministic function of
these values and the fingerprinted contract:

| Head input | Rebuilt from |
| --- | --- |
| Row source | The question's `first_cases` bitmap, in planning order |
| Observation | The question's measurements, permuted by `scenario`, at `state` |
| Frozen hidden vector `h` | The observation, re-encoded by the frozen predictor |
| Valid-action mask | The budget-filtered action table at `state` |
| Soft target | An even split over `preferred`, rounded exactly as the planner did |

When a question is labeled, the bitmap must reproduce the planner's row order,
every row's rebuilt observation is checked against the row's visible state key,
and targets must use valid actions. Each question also stores the SHA-256 of its
rebuilt observations. Reused questions are re-verified against that hash, so an
observation-code or data change cannot silently alter training inputs. On CPU
the re-encoded features match label-time features bit for bit; on a GPU they
can differ by about 2e-7 from floating-point roundoff.

Planner diagnostics (goal-rank distributions, per-action reach/cost/steps,
objective) and the 11,472 case dispositions per question are not stored.
`decision_label_coverage.json` keeps their counts. To inspect one question,
`WORK.decision_label_audit("policy", position)` (or
`decision_question_audit(...)`) rebuilds its full rows and cases and checks
them against the saved labels. Run it on the labeling device; another device
can round probabilities differently and fail the check.

**Memory.** The planner keeps its working state in arrays: one status, goal
rank, and training row per case, and solved continuations at 24 bytes per case
and target rank. The per-row and per-case audit dictionaries are built only for
`decision_label_audit`. Labeling one default-geometry question therefore peaks
near 20 MiB of Python/NumPy allocations, down from about 60 MiB. During head
training only the action bytes, bitmaps, and row offsets stay in RAM, about
30 MiB for the default run (schema 4 held about 125 MiB); each question's row
sources are rebuilt from its bitmap when its rows are encoded. There is no cache
setting; the earlier `decision_cache_mb` option is gone. Only one question's
re-encoded features and training tensors occupy GPU memory at a time. Before
writing, Stage 3 prints free disk space and an upper bound for its label files,
and warns if the bound does not fit.

**Why the earlier layouts used more.** Earlier revision-5 code pickled every
audit row and case per question, first with the 128-value feature vector stored
twice (about 8 MiB per question; the v5 run hit `No space left on device` after
1,467 policy questions), later about 1 MiB per question in two files, with a
4 GiB in-RAM tensor cache. Most of that was derivable data, such as a
64-character hash string per row. Schema 4 then kept an `int32` source per row
(about 5 bytes per row), which the bitmap now replaces.

**Recognizing the earlier code.** The notebook embeds its implementation,
so a Kaggle copy uploaded before these changes keeps its old storage. A copy
that still writes audit shards fails inside `atomic_torch(path, shard, compress=True)`
while writing `decision_shards/<role>/<position>_<hash>.pt`, and its Stage 3
progress reads `building/validating N question shards`; the current code has
neither that call nor that directory as a write target. Stage 0 of the current
workflow prints the implementation revision and the label schema
(`Implementation revision 5; Stage 3 stores compact labels (schema 5, one byte
per row plus one bitmap per question)`) before any stage trains, so the running
code can be checked in the first minute rather than hours later.

**Recovering such a run.** Keep the same `output_dir`, input files, and
training configuration with `resume=True`. Constructing `Workflow(CFG)` on an
existing run deletes the derived `.training.pt` companions (up to 4 MiB each)
and interrupted `.tmp` files before Stage 0 writes its reports, and prints
the reclaimed size, so a completely full disk needs no manual cleanup.
Stages 0-2 load their completed checkpoints; `WORK.train_decision_head()`
then converts each completed schema-3 shard to schema-5 labels **without
relabeling** and deletes the shard, so free space only grows; the progress
line counts these as `migrated`. Schema-4 labels convert the same way:
per-question files are rewritten in place and counted as `migrated`, and a
merged schema-4 role file is rewritten once. Only missing questions are built,
and a matching schema-3 or schema-4 manifest is upgraded in place. Head
training resumes from `decision_head_progress.pt` with the same row order.
Revision-4 runs cannot resume under the revision-5 action and label contract;
start revision 5 in a new output directory.

A `torch.save` iostream error followed by `unexpected pos` is a checkpoint
write failure, commonly caused by exhausted disk space or a storage quota.
It is not evidence of a GPU-memory failure. Failed writes report the affected
path, free space, and original exception, remove the partial temporary file,
and preserve completed files. Free filesystem space alone does not rule out a
quota or I/O problem.

The Stage 3 implementation reuses budget-specific structure/action tables
across questions, constructs observations in NumPy batches, reuses each
scenario's observations for both inference and visible-state grouping, and
evaluates the scalar goal rules in arrays.
It still enumerates every state/order case and uses shared continuation
decisions and masked training. The evaluator action and tie rule changed in
revision 5, so revision-4 labels cannot be reused. Training preserves the
shuffled minibatch order and optimizer steps, and reads the accumulated loss
back once per question.

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
assembly, state enumeration, continuation search, grouping, and label
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

This implementation uses **revision 5**, **contract/inference schema 4**,
**decision-dataset schema 3** (the label fingerprint), and **label-file
schema 5**. Older action/state bundles and checkpoints are
rejected. Use the new default output directory
`/kaggle/working/adaptive_analogical_shared_training_v5_run` for a fresh run; old runs are
not migrated. The notebook's `QUICK_PILOT` checks execution, not final model
quality. Recorded outcomes cannot establish live provider behavior or the
results of newly generated candidates.
