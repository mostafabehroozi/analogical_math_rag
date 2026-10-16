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
   Its loss masks absent candidates. Training visits every presence state
   under every zero-shot order (see "Exhaustive Stage 1-2 training views");
   only the complete-state predictions define the ranking.
2. **Stage 2: recognition predictor.** A second ResNet uses the same observation
   representation. It predicts eight correctness values, eight SAFE memberships,
   eight MAX memberships, and global SAFE/MAX presence: 26 outputs. Its
   candidate-specific losses mask absent candidates. Development-only
   temperature calibration is fitted, then the 128-dimensional encoder,
   prediction heads, normalization state, and calibration are frozen.
3. **Stage 3: acquisition head.** A network reads the frozen 128-dimensional
   hidden vector plus the plain masks of the acquired candidates and evaluators
   (141 inputs), passes them through 64 hidden units, and scores eleven actions:
   the next zero-shot, adding the evaluator of source j, or generating its
   one-shot candidate. It is supervised with masked cross-entropy in two
   phases. Stage 3a pretrains on every first step that lies on a cheapest route
   to the next goal; Stage 3b fine-tunes from those weights on one action per
   row, chosen by rolling the pretrained head forward. Both heads are kept and
   reported. There is no STOP output, reward, replay buffer, or Bellman target.
   Stage 3 uses complete measurements for every presence state, the same 8,160
   states Stages 1 and 2 train on, except the complete state, which has no move.

## Exhaustive Stage 1-2 training views

Stages 1 and 2 train on every view of every training question; nothing is
sampled. A view is a presence state seen under one zero-shot order:

- **Presence states:** every nonempty candidate subset with every evaluator
  subset, `(2**8 - 1) * 2**5 = 8,160` by default. A one-shot candidate does
  not need its own evaluator, so these are all the states that acquisition,
  and hiding candidates or evaluators, can produce; Stage 3 labels these
  states too, except the complete one.
- **Zero-shot orders:** all `3! = 6`, so each question has
  `8,160 * 6 = 48,960` views. Candidate IDs, labels, SAFE/MAX, and
  measurements move together, exactly as in Stage 3.
- **Measurements:** complete for every present candidate/evaluator pair.
  Individual failed or missing measurements (about `3**45` combinations per
  state) and evaluator shuffles cannot be enumerated and are not used; the
  offline rollouts and Stage 3 never contain them either.

The partial states only train the models. The teacher's reference ranking,
its out-of-fold scores, and the SAFE/MAX labels built from them always come
from **complete-state** predictions, and teacher checkpoints are selected by
the complete-state development correctness loss. Stage 2 is selected by its
own training loss, and calibrated, on every reachable runtime state (1,092 by
default) of every development question under every zero-shot order. Average
precision (AP) is a reported metric only (see "Checkpoint selection").

**Complete-view share.** Each question has only 6 complete views among its
48,960, so an unweighted epoch would spend 0.012% of its training on the one
state whose predictions later stages use. A larger loss weight would not fix
this: a 16,384-row batch holds about two complete rows, and weighting them by
thousands makes the gradient follow those two rows. Instead, each epoch
repeats every complete view until complete views form a configured share of
its rows, and shuffles the repeats in with the partial views; every partial
view is still visited exactly once. The copy count is
`round(share * (8,160 - 1) / (1 - share))`:

| Setting | Default | Copies of each complete view | Rows per question per epoch |
| --- | --- | --- | --- |
| `teacher_complete_share` | 0.15 | 1,440 | 57,594 |
| `snapshot_complete_share` | 0 | 1 (no repeats) | 48,960 |

The teacher's ranking is read only from complete views, so 15% of its
training is on them. The snapshot model's main job is partial-state
recognition, so by default its complete views count like any other view
(0.012% of its rows). `0` disables the repeats. The teacher is selected on its
complete-state development loss, Stage 2 on its loss over every reachable state.

One epoch visits every view once (plus those repeats): about 98 million views
for 2,000 training questions. The defaults are therefore `batch_size=16384`,
`teacher_epochs=snapshot_epochs=10`, and `dev_checks_per_epoch=8`. Development
checks are spread evenly through each epoch, and the checkpoint's position is a
fractional epoch such as `1.375`. Every minibatch holds rows of every training
question (see below), so an epoch is thousands of passes over the questions'
labels and the best check comes early in the first epoch. Stages 1-2 therefore
stop at the first check `view_patience` (default `0.25`) or more epochs after the
best one, inside an epoch if need be; `teacher_epochs` and `snapshot_epochs` only
cap a run that keeps improving. `patience` counts whole epochs in Stage 3 alone,
where an epoch is one pass over the questions. On real logs (fold-sized splits of
the AIME, GSM8K and MATH-500 logs: 1,280 training and 320 development questions,
`dev_checks_per_epoch=64`, local GPU), fold teachers were best at epochs
0.016-0.045 with a development loss of about 0.58, passed the constant-prior loss
near epoch 0.1 and reached about 1.2 by the end of the first epoch, while their
complete-state training loss fell to 0.07 and the reported AP fell too. The
snapshot model was best at epoch 0.17. Revision 9 counted whole epochs in these
stages too, checked only at epoch ends, so `patience=15` with `teacher_epochs=10`
never stopped a teacher: each fold trained ten epochs to keep its epoch-0.031
weights. The training loop prints one line per epoch, and one where it stops:
the mean training loss over the epoch, the latest check's development loss, the
best development loss and the epoch of its check, and the epochs without
improvement, for example `epoch  0.625 | train loss 0.4846 | latest development
loss 0.6120 | best development loss 0.5874 at epoch 0.375 | epochs without
improvement 0.25/0.25`.
Stage 2 also shows Top-1 and Brier in parentheses after each loss; they are
information, not selection criteria.

**Memory and disk.** No view is stored, on disk or in RAM. A shared state
table (about 7 MiB) holds the state-only observation bytes and masks; each
question adds its measurements, labels, and SAFE/MAX under each order (under
1 KiB). `SnapshotViews` assembles each minibatch on the selected device from
those tensors, bit-identical to `observation(...)`.
Every minibatch takes an even slice of **every** question's rows: with 1,280
fold questions and `batch_size=16384`, 12 rows of each. Each question walks one
shared random order of its rows from its own random start, so an epoch holds a
single permutation of one question's rows (under 0.5 MiB), not of all 98
million. An earlier revision-6 build (`c003b05`) instead shuffled the views of
about 36 questions at a time, so about 126 consecutive optimizer steps saw the
same few questions. On real logs (640 GSM8K training questions, 500 MATH-500
development questions, 4 checks per epoch, local GPU, measured while checkpoints
were still selected on development AP), its epoch-mean training
loss went 0.298, 0.282, 0.277 over three epochs, with the best development AP
0.5606 at epoch 1.25. All-question batches went 0.364, 0.221, 0.175, with the
best AP 0.5552 at epoch 0.25 and lower AP afterward: each question appears tens
of thousands of times per epoch, so the model fits its training questions
within the first epoch, and the best checkpoint comes early. Development
metrics,
temperature calibration, and the Stage 4 snapshot diagnostic stream per-batch
sums; no logits are kept. Adam uses its fused CUDA kernel.

Measured on a local RTX 3050 Ti laptop GPU with 120 synthetic default-geometry
questions: about **1.8 million views/s** for Stage 2 training including its
development checks and calibration, a flat host RSS (+1 MiB after CUDA
initialization), and 180 MiB peak CUDA memory. At that rate one epoch takes
about 1 minute for Stage 2 and, with the teacher's complete-view repeats
adding 18% to its rows, about 5 minutes for Stage 1's five folds plus final
teacher, so the 10-epoch maximum is about an hour for both stages before
early stopping. Kaggle hardware will differ, but a 1,280-question fold epoch
with 64 checks takes about 70 seconds on this GPU; one that takes many minutes
is probably running on the CPU, which Stage 0's `DATA AUDIT | device=` line
shows. This was measured when each
Stage 2 development check covered 189 reachable states; it now covers 1,092, so
a check costs about six times as much. Lower `dev_checks_per_epoch` to spend
less time on checks.

SAFE is the leading uninterrupted run of actually correct candidates in the
teacher order; MAX is its first member. If rank one is wrong, both are empty.
These historical labels define offline goals and audit strata, not runtime
truth. For example, ordered correctness [correct, wrong, correct] makes only
the first candidate SAFE and MAX; [wrong, correct, correct] makes both empty.
More evidence can lower a recognition probability. Temporary loss is allowed,
but the first unfinished rank is recomputed at every state: restoring an
earlier lost goal takes priority over continuing a later goal.

## Acquisition state, actions, and cost

`State(candidate_mask, evaluator_mask)` records explicit presence. Any nonempty
candidate subset combines with any evaluator subset: a one-shot candidate does
not need its own evaluator, and can be generated before, after, or without it.
A state uses complete recorded measurements for every present
candidate/evaluator pair. A one-shot candidate whose evaluator is absent shows
its source slot but none of that source's similarity, baseline, or
cross-evaluation. Every state contains at least one candidate; candidate-empty
states are outside this contract. Retrieval is not part of the state: the
recorded logs already hold the retrieved samples and all their measurements.
The runtime begins with ZS1 and no evaluators, so its first generation precedes
the head's decisions.

Which acquisition may come next depends on which sources are in use. Sources
are ordered by similarity (source 1 is the nearest), and a source is **in use**
once its evaluator or its one-shot candidate is present.

- The zero-shot action fills the first missing slot; the head cannot select a
  specific unseen answer.
- A source in use may be completed at any time with its missing half, in either
  order: the head chooses which evaluator to add and which one-shot candidate to
  generate among the sources in use.
- The nearest source that nothing has used yet may be opened with either half,
  its evaluator or its one-shot candidate. The head cannot open a farther
  source first or choose which unused source is nearest.

At the start the available acquisitions are therefore the next zero-shot, R1,
and OS1; after R1 they are the next zero-shot, R2, OS1, and OS2. Of the 8,160
presence states, **1,092** are reachable from the start under these rules: three
zero-shot prefixes times the `1 + 3 + 9 + 27 + 81 + 243 = 364` ways to use the
nearest m sources (one-shot only, evaluator only, or both). The other states
are never visited at run time, but they are labeled and trained on.

| Action index | Operation |
| --- | --- |
| `0` | Generate the next zero-shot candidate and cross-evaluate it against the active evaluators. |
| `1 … k` | Add the evaluator of source j (`R1 … Rk`), its baseline, and its cross evaluations against the present candidates. |
| `k+1 … 2k` | Generate the one-shot candidate of source j (`OS1 … OSk`), whether or not that source's evaluator is active, and cross-evaluate it against the active evaluators. A source added later as an evaluator is then cross-evaluated against every present candidate, this one included. |

The default `k=5` gives eleven actions. Unavailable or unaffordable actions are
masked. The incremental cost is the difference between the existing state-cost
formula before and after acquisition. With five repeated solves per estimate,
the default objective counts **458 total calls**: eight generations, 25
baseline solves, 200 cross-CCS solves, and 225 corresponding graders.
`cost_unit="solver_calls"` optionally excludes those graders and gives a
233-call complete pool. Retrieval/embedding work and target-answer grading
remain outside this recorded-data accounting. There is no separate penalty
for the number of acquired candidates or evaluators. The cost depends only on
how many candidates and evaluators are present, so a candidate generated while
no evaluator is active costs its single generation call, and each evaluator
added later pays to measure it. Every order of adding the same items therefore
costs the same, which is what makes "cheapest route" well defined below.

## Stage 3 labels

For every policy and development question, Stage 3 enumerates every nonempty
candidate subset and evaluator subset: **8,160 presence states** for three zero
shots and five evaluators, the same states Stages 1 and 2 train on. It
evaluates each under all six zero-shot permutations: **48,960 state/order
cases per question**. Enumeration is independent of runtime acquisition order
and budget. Cases above the budget stay in the dataset with an explicit status.
The complete state has no acquisition and so no label.

Thus **2,000 training questions produce 97,920,000 state/order cases**, before
development questions. Almost every case is a head-training row: it needs only
an affordable acquisition, and cases that look identical to a case with a route
to its goal defer to that case (see "Status and rows"). Rows are not merged, so
each case counts once, like every other.

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

Teacher order prioritizes recognition goals, not physical generation: a
lower-ranked supporting candidate may be acquired first. A recognition
probability can fall again as evidence arrives, and the **next goal of a case
is recomputed from its own state**: the first rank for which ranks one through
that rank do not all hold. A lost earlier goal is therefore restored before a
later one is pursued.

### Stage 3a: every cheapest first step

For each case, over the acquisitions the rules allow:

1. **Distance.** The fewest additional API calls from the state to a state where
   its next goal holds. A route ends the first time it reaches such a state.
   Because cost depends only on the items present, this is a shortest-path
   search over which items to add, solved backwards over the acquired-item count.
2. **Cheapest first steps.** Every allowed acquisition whose own cost plus the
   distance that remains afterwards equals the distance. Several can tie, for
   example R1 and OS1 when the goal needs both, since either order costs the
   same. A step that overshoots the destination is not on any cheapest route.
3. **Label.** An even split over the case's cheapest first steps. There is no
   other tie rule: no preference for an evaluator over a candidate, no lowest
   index, and no fallback acquisition.

Cases that look identical to the head (byte-identical observations, which
happens mostly where nothing was measured yet: a state without evaluators looks
the same under all six zero-shot orders, and so does a state without zero-shot
candidates) keep their own labels; training averages them.

### Status and rows

Every case has exactly one disposition, with this precedence:

| Status | Meaning | Stage 3a row |
| --- | --- | --- |
| `OUT_OF_BUDGET` | The state itself exceeds the configured budget. | None |
| `COMPLETE` | All reference recognition goals hold. This needs every candidate, which leaves only evaluator actions, but not every evaluator. | No preference: every available move |
| `EXHAUSTED` | The full pool is acquired and a goal remains unfinished. | None (no move) |
| `BUDGET` | The pool is incomplete and no affordable action remains. | None (no move) |
| `ACTION` | Acquisitions remain and a state where the next goal holds can be reached by adding items. | Its cheapest first steps |
| `UNREACHABLE` | Acquisitions remain, but no state reachable by adding items satisfies the next goal. | No preference: every available move |

A case with no preference is labeled with every move it has, except that it has
no row when an identical-looking case is `ACTION`: that case speaks for the
group. Every state but the complete one therefore has a label for its input, and
the head is still trained through the full pool, but uninformative cases never
dilute informative ones. All rows have equal loss weight. Terminal and
over-budget cases remain audit records with no STOP or abstention target.
Offline rank and historical truth never enter the head input.

### Stage 3b: the pretrained head's own continuations

The pretrained head labels every state with a set of tied cheapest first steps.
Stage 3b keeps one, as follows. For each cheapest first step of a case:

1. Take the step.
2. Let the pretrained head choose every further move, greedily among the
   available moves (the same rule as at run time), until the same next goal
   holds or no move remains.
3. Count the calls spent.

Cases the head cannot tell apart share one decision, judged on average over
them: the largest share of threads that reach the goal first, then the fewest
calls, then the pretrained head's own probability of the step. If no thread
reaches the goal, that probability alone decides. A case without a route (no
preference) keeps the head's favorite available move. Every row then has one
label, and the fine-tuning loss is cross-entropy against it. Threads run to the
next goal only, not to every later goal, and the pretrained weights are the
starting point of fine-tuning.

Where a case has one cheapest first step, that step is also its Stage 3b label;
the continuations only decide ties among several. If the pretrained head were
perfect, tied steps would cost the same and Stage 3b would reproduce its
choices. The log reports the rows with tied steps (`tied_rows`), how many of
those the continuations moved away from the pretrained head's favorite move
(`tied_rows_moved`), how many rows in all differ from that favorite
(`changed_rows`, which includes the head's mistakes on rows with one cheapest
step), and how many routed rows have no thread that reaches the goal
(`no_thread_reaches_rows`). Because the continuation comes from a head trained
on the same questions, the measured costs are slightly optimistic about how it
continues on new ones.

Measured on a local RTX 3050 Ti laptop GPU with six synthetic default-geometry
questions and an untrained predictor (`benchmark_adaptive_stage3.py`, which
writes `stage3_two_phase_performance_report.json`): labeling one question takes
about **0.7 s** per phase (the previous planner took 3 to 4 s) and peaks near
100 MiB (3a) and 80 MiB (3b) of Python/NumPy allocations. Each question has
about 41,000 to 48,000 rows, 18 to 45% of them with a route; a trained predictor
changes that split. One head epoch takes about **0.55 s** per training question
and **0.06 s** per development question at `decision_batch_size=128`, roughly 20
minutes for 2,000 + 1,900 questions: about 320 optimizer steps of 128 rows per
question dominate it, and `decision_batch_size=1024` takes about one seventh of
the step time. Question count alone is not a training-set size.

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
| 1: teacher | Shared `supervised` questions | `dev`: final teacher checkpoint selection (complete-state correctness loss); the reported heuristic baselines are also chosen on `dev`, by AP |
| 2: snapshot encoder and prediction heads | Same `supervised` questions | Same `dev`: checkpoint selection and temperature calibration |
| 3a: pretrained head | Same `supervised` questions, through the `policy` alias | Same `dev`: checkpoint selection against the 3a labels |
| 3b: fine-tuned head | Same `supervised` questions, starting from the 3a weights | Same `dev`: checkpoint selection against the 3b labels |

The teacher's fold models still use internal training/validation folds and
produce out-of-fold labels for all shared training questions. The final
teacher trains on all shared training questions. The snapshot predictor
stays frozen while the acquisition head trains.

**Checkpoint selection.** Every stage keeps the checkpoint with the lowest
development loss. Average precision (AP) is neither a training nor a selection
signal in any stage; it is a metric of the final reports.

| Stage | Training loss | Development loss that picks the checkpoint | Reported, never selected on |
| --- | --- | --- | --- |
| 1: teacher | Correctness cross-entropy of the present candidates | The same loss on the complete-state `dev` views, where every candidate is present | AP, Top-1 and coverage in the ranking tables |
| 2: snapshot | Correctness cross-entropy plus `auxiliary_weight` times the SAFE, MAX and presence cross-entropies | The same loss over every reachable state and zero-shot order of `dev`, before calibration | Top-1, Brier, SAFE/MAX precision and recall (shown during training and in the reports) |
| 3a: pretrained head | Masked action cross-entropy against an even split over the cheapest first steps | The same loss on the `dev` rows | |
| 3b: fine-tuned head | Masked action cross-entropy against one chosen action | The same loss on the `dev` rows | |

The heuristic baselines in the ranking tables are not trained: the best one per
mode is picked on `dev` by AP and reported next to the teacher. None of these
criteria directly optimizes recognition-goal coverage or realized API savings.

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

Stage 3 writes `decision_labels/pretrain_policy.pt`, `pretrain_dev.pt`,
`finetune_policy.pt` and `finetune_dev.pt`, with `decision_dataset_manifest.json`
and `decision_label_coverage.json`. The manifest and label files bind to the
config (not the input log paths), the records, split, teacher labels, and frozen
predictor fingerprint; the fine-tuning labels also bind to the exact pretrained
head they were rolled out from. While a role builds, each completed question is
one small atomic file in `decision_labels/<phase>_<role>/`; an interrupted build
resumes completed matching questions, and the finished role is merged into one
file. Incompatible files fail explicitly. A role with no actionable rows fails
with its coverage report. During training, policy question and row order are
shuffled deterministically each epoch, and dev questions are read in stable
order for masked-loss checkpoint selection. All labels of a phase are
constructed **once before that phase's epoch loop**; epochs reuse those rows and
never regenerate states or routes. Frozen hidden vectors are not stored: each
training and development pass rebuilds every row's observation and re-encodes it
with the frozen predictor on the selected device. A new invocation verifies
saved labels before reuse. The frozen Stage 2 state is checked after training.

Stage 3 prints question progress, elapsed time, and estimated remaining time
for each role's label build and each epoch's training/development pass. Updates
appear on the first/last question, every `print_every` questions, or after 30
seconds at the next question boundary. Each epoch also prints both losses and
early-stopping patience. The `Stage 3a policy: ...` coverage summary means policy
labels are complete; development labels come next, before head training. It
lists the case statuses, how many rows have a route to their next goal, and how
many cheapest first steps those rows have; the 3b summary adds the rows moved
away from the pretrained head's favorite. With the default external-development
setting, all four benchmark logs supply development questions, and their label
build can be substantial. Keep the same output directory and configuration to
reuse completed questions after interruption, including in a new Kaggle session
when `hf_repo` mirrors the folder (see "Surviving Kaggle session limits"). After
every completed training/development epoch, `decision_head_pretrain_progress.pt`
(3a) or `decision_head_finetune_progress.pt` (3b) atomically saves current and
best head weights, Adam state, question/row shuffle RNG, losses, and
early-stopping bookkeeping. With `resume=True`, head optimization continues from
the last completed epoch of the phase in progress; an interrupted partial epoch
repeats from its start. Failed checkpoint writes preserve the previous
completed epoch. Keep the training and development settings and selected device
consistent when comparing a resumed run to an uninterrupted run. The pretrained
head is saved as `decision_head_pretrained.pt` when Stage 3a finishes, and the
fine-tuned head as `decision_head_completed.pt` when Stage 3b finishes.
Fine-tuning starts from the pretrained weights with `decision_finetune_lr` for up
to `decision_finetune_epochs`; `patience` counts epochs in each phase.

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
  created on the first upload if it does not exist. The traced Stage 4 rows
  that earlier revisions mirrored (`final_trajectories.jsonl`) are not
  downloaded.
- **Push.** The dataset cell's first upload is required, so a bad token,
  repository id, or disabled Internet fails in the first minute. The dataset
  phase uploads after each log it reads, and each later stage uploads when it
  completes. Inside long loops the run uploads every `hf_sync_minutes`
  (default 15): during Stage 1-2 training (with the model's training state),
  after Stage 1 folds, during the Stage 3 label build, and after Stage 3 epochs;
  each merged role uploads immediately. Uploads skip unchanged
  files by content hash, never include `.tmp` files, and remove per-question
  label files, a completed model's Stage 1-2 training state, the dataset build's
  per-log records, and an earlier revision's
  `final_trajectories.jsonl` from the mirror once they are gone locally (the
  dataset repository's history keeps earlier versions).
  A failed periodic upload prints a warning and retries at the next sync
  point; local files are never discarded. `WORK.push()` uploads on demand,
  and `WORK.sync.describe()` reports the upload state.
- **Resume.** In a new session, download the same input logs, keep the same
  configuration, and run the cells. Inputs are identified by filename, size,
  and SHA-256, so a new directory or mtime keeps the same run (a contract
  that bound the logs to their mtime, from before content hashing, is rejected
  as a different contract). The logs may be downloaded to a different directory than
  last session: point `train_file` and `test_files` there. The Stage 3
  fingerprint leaves those paths out too; the records' content and their uids,
  which name each log's filename stem, identify the logs. Stages 0-2 load
  their completed checkpoints; Stage 1 also reuses completed
  `teacher_fold_<n>.pt` and `teacher_final.pt` files whose question roles
  match exactly, and a teacher or the snapshot model that was training
  continues from its saved training state (next item), so an interrupted
  Stage 1 or 2 repeats at most about `hf_sync_minutes` of training. Stage 3
  reuses completed questions and continues from the last completed epoch of
  the phase it was in; fine-tuning labels are bound to the exact pretrained head
  they were rolled out from. Stage 4 restarts if interrupted.
- **Training state.** A teacher or the snapshot model trains for hours (a
  teacher epoch over 1,280 questions took about 17 minutes on Kaggle), longer
  than a session may have left. At the first development check after every
  `hf_sync_minutes`, its current and best weights, Adam state, random states,
  position in the epoch, and history are saved atomically to
  `training_progress/<model>.pt` (`teacher_fold_<n>.pt`, `teacher_final.pt`,
  `snapshot.pt`) and uploaded. A session that finds the file restores it and
  continues with the epoch's next batch; the epoch's batch order is rebuilt from
  its seed, so on the same device the model ends exactly as an uninterrupted
  run would. The file is bound to a hash of the questions and targets the
  model trains on; one saved for other training is reported and replaced. It
  is removed, locally and on the Hub, once the model's completed checkpoint is
  saved.
- **One session per repository.** Two sessions writing the same mirror would
  overwrite each other; use a different `hf_repo` for a different run
  (`QUICK_PILOT` appends `_pilot` to both the output folder and the mirror).
- **Datasets.** With `dataset_id` set, the compact records that the dataset
  phase reads from the logs are stored separately, under a version, and never
  rebuilt once uploaded; the run mirror carries everything else, every model
  checkpoint and label included (see the next section).

### Datasets and dataset IDs

The dataset is what the input logs alone determine: every eligible question's
candidates, similarities, baselines, cross-evaluation measurements, and
correctness labels, with each log's audit counts (`compact_records.pt`). The
notebook's dataset cell (`WORK.build_datasets()`, the first workflow cell)
builds it, and **building it trains, loads, and runs no neural network**: it
reads the logs and nothing else. Everything a model produces is training
output, made by the stage that trains that model:

| Model output | Made by | Why it is not part of the dataset |
| --- | --- | --- |
| Out-of-fold scores and SAFE/MAX labels | Stage 1 (`teacher_*.pt`) | SAFE/MAX follow the cross-fitted teachers' ranking |
| Frozen snapshot predictor | Stage 2 (`snapshot_completed.pt`) | Trained on the SAFE/MAX labels |
| Stage 3a labels | Stage 3a (`decision_labels/pretrain_*.pt`) | Every goal is the frozen predictor's recognition |
| Stage 3b labels | Stage 3b (`decision_labels/finetune_*.pt`) | Rolled out with the pretrained head |

So Stage 1 is the first cell that trains, and each later stage builds its labels
from the model before it; they are checkpointed in the output folder and, with
`hf_repo`, in the run mirror (see "Surviving Kaggle session limits"). The split
and data audit are not part of the dataset either: Stage 0 recomputes them from
the records and the seed in every session, and they name the logs' paths.

**Checkpoints.** The build saves each finished log to
`dataset_progress/<n>_<log>.pt` (its records and audit counts) and uploads it at
once, so a session that ends during the build reads only the logs it had not
finished. A saved log is bound to the dataset schema, the record settings, and
the content of that log and of every log before it, since exact duplicates of an
earlier log's questions are excluded; a file saved for other logs or settings is
reported and replaced. Once `compact_records.pt` holds every log, the per-log
files are removed, locally and on the Hub. The build's first upload is required,
so a bad token or repository fails before any log is read. Without `dataset_id`
the same files go to the run mirror.

**Store.** Set `dataset_id` (a name such as `adaptive-v10`) and `dataset_repo`
(a private Hub dataset repository id that can hold many ids, and must differ
from `hf_repo`; the token is the one it uses) to keep the dataset apart from the
run mirror, version it, and never rebuild one that the Hub already holds. Both
are runtime-only, outside the run contract and every fingerprint. The dataset
store holds `compact_records.pt`, `dataset_manifest.json`, and, while a build is
in progress, `dataset_progress/*`. Every model checkpoint, label file, and
training state is in the run mirror, so keep `hf_repo` set to continue training
in a new session; the workflow says so when it is missing.

**Version.** A dataset lives at `<dataset_repo>/<dataset_id>/<version>/`. The
version is the first 12 hex digits of a SHA-256 over what decides the records:
the record settings that `parse_record` reads (`k`, `zero_shots`, `repeats`),
the dataset schema, and each input log's filename, size, and content hash, in
configuration order. Training settings and the implementation revision do not
change the records, so they never make a new version: another `teacher_epochs`,
learning rate, or revision downloads the same dataset. Another record setting or
other log content under the same id is another version beside the first, which
stays untouched; the workflow prints which settings differ (`config.repeats`,
the input logs, ...). Equal numbers are equal settings (`5` and `5.0`).

**What a session does** (when `Workflow(CFG)` is created, then
`build_datasets()`):

| The Hub holds, for this id and version | Result |
| --- | --- |
| Nothing | The logs are read and the dataset is uploaded as it progresses: before the first log, after each log, and sealed at the end. Records already in the output folder (a run built before the dataset store, pulled from its run mirror) are uploaded under the version, not read again |
| A partial dataset (`complete: false`) | Its per-log files are downloaded and only the remaining logs are read |
| A complete dataset | `compact_records.pt` is downloaded and its size and SHA-256 verified; no log is read and nothing is uploaded |

Finishing a build writes `dataset_manifest.json` with `complete: true`, the
records' size and SHA-256, and the record and audit counts, and uploads it
together with any file the Hub still lacks in one commit, so the Hub never shows
a complete dataset with a file missing. A sealed dataset is read-only: later
uploads go to the run mirror only.

**Safeguards.** A dataset is never overwritten or silently replaced:
`resume=False` against an uploaded version, an output folder that holds another
dataset id or version, and records that differ from the uploaded hash all stop
with an explanation (use an empty output folder to download, or another
`dataset_id` to keep a rebuild as its own dataset). Hub reads retry three times;
if the Hub still cannot be read, the run stops with an error rather than
treating an outage as "not generated yet" and rebuilding.

**Without the logs.** A session may attach no input logs. The version cannot be
hashed from content then, so the Hub must hold exactly one complete version of
the id built with these record settings and these log filenames; its recorded
logs (name, size, hash) stand in for them in the run contract. Zero or several
matches stop with an error asking for the logs. The compact records are all
Stages 0-4 read.

**Datasets from dataset-store schema 1.** Until schema 2 the dataset also held
the model output of Stages 1-3a (teachers, snapshot predictor, Stage 3 labels,
the pretrained head) and the training state of a build in progress, under a
version hashed from the run's whole contract, and `build_datasets()` trained
those models. Such a version belongs to exactly one run. When a run that holds
no training output of its own yet finds the schema 1 version of its contract
under its id, it downloads those model files into the output folder once; the
stages validate them as their own and continue, and the run mirror keeps them
from then on. A session that switched to this code with the schema 1 build
still in its working folder keeps those files the same way: its old
`dataset_manifest.json` is recognized as this run's and replaced, and its
records go to the schema 2 version without reading the logs again. The schema 1
version stays untouched, and the records are stored again under a schema 2
version. Delete the old folder on the Hub once the run mirror holds the
checkpoints.

**Housekeeping.** Delete a `<dataset_id>/<version>` folder in the repository to
drop a version. Run one session per dataset id and version, as with the run
mirror, and use a new `dataset_id` for experiments (`QUICK_PILOT` appends
`-pilot`). The Hub's repository history keeps earlier commits.

### Stage 3 label storage

Stage 3 saves only what each head trains on (label-file schema 6): one array
per question and phase, indexed by case (`scenario * in-budget states + state`).

- Stage 3a `options`: for every case, a bit mask of its cheapest first steps,
  packed into `ceil(actions / 8)` bytes (two for the default eleven actions);
  all zero when the case has no row.
- Stage 3b `choice`: for every case, the chosen action in one byte; 255 when the
  case has no row.

A default-geometry question takes about **98 KiB** (3a) and **50 KiB** (3b) with
file overhead, so 2,000 policy questions plus the pooled test-file development
questions take roughly 590 MB on disk. Everything else is a deterministic
function of these values and the fingerprinted contract:

| Head input | Rebuilt from |
| --- | --- |
| Row source | The question's nonzero cases |
| Observation | The question's measurements, permuted by `scenario`, at `state` |
| Frozen features | The observation, re-encoded by the frozen predictor, plus the acquired-item masks |
| Valid-action mask | The budget-filtered action table at `state` |
| Target (3a) | An even split over the case's `options` bits |
| Target (3b) | A one-hot at the case's `choice` |

When a question is labeled, every row's rebuilt observation is checked against
the digest taken at label time, and each question's 3a labels store the SHA-256
of those observations. Reused questions are re-verified against that hash, so an
observation-code or data change cannot silently alter training inputs. Stage 3b
labels must choose available actions for exactly the cases that have a 3a row.
On CPU the re-encoded features match label-time features bit for bit; on a GPU
they can differ by about 2e-7 from floating-point roundoff.

**Head input.** The head reads the frozen 128-dimensional hidden vector plus 13
plain facts of the visible state: which of the eight candidates and which of the
five evaluators are acquired (141 inputs). These masks were already inputs of the
frozen network, so nothing hidden enters. They are added because an action's
meaning depends on the state ("add R3" opens source 3 in one state and completes
it in another) and the hidden vector, trained to predict correctness, is not
guaranteed to keep them.

Distances, goal ranks, and the 48,960 case dispositions per question are not
stored; `decision_label_coverage.json` keeps their counts for each phase and
role. To inspect one question, `WORK.decision_label_audit("policy", position)`
(or `decision_question_audit(...)`) rebuilds its cases with their statuses, goal
ranks, and cheapest first steps and checks them against the saved 3a labels.
Run it on the labeling device; another device can round probabilities
differently and fail the check.

**Memory.** The route search works on arrays: a few boolean and float tables of
states x actions x goals, about 27 MB of head features for the six orders, and
no per-case dictionaries. Labeling one default-geometry question peaks near
100 MiB (3a) and 80 MiB (3b) of Python/NumPy allocations. During head training
only the label arrays stay in RAM: about 380 MB for the 3a stores and 190 MB for
the 3b stores of the full run (training and development questions). While 3b
labels are built the 3a stores are also loaded, to check that the two phases
cover the same rows, and released afterwards. Only one question's re-encoded
features and training tensors occupy GPU memory at a time. Before writing,
Stage 3 prints free disk space and an upper bound for its label files, and
warns if the bound does not fit.

**Recognizing the earlier code.** The notebook embeds its implementation, so a
Kaggle copy uploaded before these changes keeps its old behavior. Stage 0 of
the current workflow prints the implementation revision and the label schema
(`Implementation revision 10; Stage 3 labels every presence state with its
cheapest routes to the next goal and pretrains on them (Stage 3a), then
fine-tunes on the route its own continuations make cheapest (Stage 3b). Compact
labels (schema 6: a bit mask per state/order case for 3a, one byte for 3b)`)
before any stage trains, so the running code can be checked in the first minute
rather than hours later. The same line ends with `Stages 1-2 train on every
presence state under every zero-shot order (8,160 x 6 views per question),
assembled on the device and never stored; each keeps the checkpoint with the
lowest development loss (AP is only a reported metric), stops 0.25 epochs after
it`. A copy that prints revision 9 stops Stages 1-2 only at epoch ends; a lower
revision plans a single label per state with its own tie rules, or still
requires a one-shot candidate's evaluator. None can continue a revision-10 run.
The dataset cell before Stage 0 prints `Implementation revision 10, dataset
schema 2: the dataset is the compact records of the input logs`; a copy that
prints `The labels come from trained models, so the build trains them` instead
still trains Stages 1-3a inside its datasets cell.

A `torch.save` iostream error followed by `unexpected pos` is a checkpoint
write failure, commonly caused by exhausted disk space or a storage quota.
It is not evidence of a GPU-memory failure. Failed writes report the affected
path, free space, and original exception, remove the partial temporary file,
and preserve completed files. Free filesystem space alone does not rule out a
quota or I/O problem. Constructing `Workflow(CFG)` on an existing run deletes
the interrupted `.tmp` files and prints the reclaimed size before Stage 0 writes
its reports, so a full disk needs no manual cleanup.

The Stage 3 implementation reuses budget-specific structure/action tables
across questions as arrays, constructs observations in NumPy batches, reuses
each scenario's observations for both inference and visible-state grouping,
and evaluates the scalar goal rules in arrays. Training reads the accumulated
loss back once per question and uses a fused Adam update on CUDA.

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
assembly, state enumeration, route search, grouping, and label
serialization still run on the CPU; enabling CUDA does not move Python search
or file I/O to the GPU. The dataset notebook cell reports the selected device,
available GPUs, memory, and CPU threads so the active session can be checked.
Kaggle offers GPU options such as T4 x2 and P100; this workflow uses one
selected device and does not automatically combine two GPUs' memory.
See [NVIDIA's Kaggle setup instructions](https://docs.nvidia.com/datascience/deployment/latest/platforms/kaggle/).

The default `decision_batch_size=128` takes about 320 small optimizer steps per
training question, which dominate an epoch. For a new CUDA experiment, try
`decision_batch_size=1024` to reduce the number of small head updates, then
compare development loss and checkpoint quality. It keeps every training row but
changes update count, gradient grouping, and the optimization trajectory; use a
**new `output_dir`** when changing it. It does not accelerate label
construction. The default head is too small to assume that adding distributed
workers, mixed precision, or a second GPU improves total runtime. Reduced device
transfers and loss readbacks follow the synchronization guidance in
[PyTorch's performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

`stage3_performance_report.json`, `stage3_full_dataset_performance_report.json`
and `stage3_local_cuda_smoke_report.json` are **historical** measurements of
earlier revisions' planner, compared against older commits; they say nothing
about the labels described here. `benchmark_adaptive_stage3.py` now measures
the current implementation.

### Stage 4 reports

Offline reports compare the complete-pool baseline, configurable fixed
sequences, and the two learned heads: `supervised_pretrained` (Stage 3a) and
`supervised`, the fine-tuned Stage 3b head. Defaults for five evaluators and three
zero-shot candidates are:

| Method | Fixed acquisition order |
| --- | --- |
| `fixed` (`interleaved`) | ZS1, R1, OS1, ZS2, R2, OS2, ZS3, R3, OS3, R4, OS4, R5, OS5 |
| `fixed_evaluators_first` | ZS1, R1, R2, R3, R4, R5, ZS2, ZS3, OS1, OS2, OS3, OS4, OS5 |
| `fixed_zero_shots_first` | ZS1, ZS2, ZS3, R1, OS1, R2, OS2, R3, OS3, R4, OS4, R5, OS5 |

Edit `fixed_acquisition_orders` in the Kaggle configuration cell to select
these built-ins or add a method such as
`"fixed_custom": ["ZS1", "R1", "OS1", "ZS2", "R2", "OS2", "ZS3"]`.
`Rj` adds the evaluator of source j; `OSj` generates its one-shot candidate.
Every sequence starts with the already generated `ZS1`, and unseen zero-shots
arrive in slot order. Duplicates, out-of-range items, and a source that the
similarity order has not reached are rejected: the nearest unused source may
be opened with `Rj` or `OSj`, and a used source's other half may follow at any
time, so a one-shot may come before, after, or without its evaluator. A custom
sequence may be a
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
recognition without interpreting the legacy reason string; `candidate_mask`
and `evaluator_mask` give the final acquired state. Full-pool reference
runs record recognition at their observed full state as well.
Pass `continue_after_max=False` to stop that call at predicted MAX, or change
the configuration default. This runtime-only choice does not change the
Stage-3 labels or training fingerprints. With the default full-pool budget,
a continuing call has no API savings against the full reference.

The returned `ranked_candidates` list sorts present candidates by their current
Stage 2 correctness probabilities, and `selected` is their correctness argmax.
Stage 1 defines offline reference goals; it does not define this runtime list.
The inference bundle contains the frozen Stage 2 predictor and both action
heads (`load_inference_bundle(path, variant="supervised_pretrained")` returns
the pretrained one; the default is the fine-tuned head), not the Stage 1 teacher. These two rankings may disagree.

Continuation diagnostics (one set per learned head, under
`continuation[method]`) keep the best-ever prefix in
`mean_rank_prefix_reached` and `per_rank_reach`. They separately report the
ending prefix in `mean_final_rank_prefix` and `per_rank_final_recognition`,
all-goal recognition in `all_goals_recognized_rate`, and acquired-pool
completion in `acquisition_complete_rate`. The continuation stops once every
reference goal holds, and a state with every candidate can still lack
evaluators, so `acquisition_complete_rate` can be lower than
`all_goals_recognized_rate`. These are averaged over zero-shot
orders within each question, then equally across questions. An earlier
recognition peak does not establish that those goals still hold at the end.

Each ACQUISITION section opens with its question counts: MAX present,
no MAX, and the two kinds of no-MAX question. A question has a MAX only when
the teacher's top-ranked candidate is correct. Otherwise it is a **teacher
miss** (a correct candidate ranks below a wrong top) or **all wrong** (no
candidate in the pool is correct). Every question is rolled out, and its
saving is the full pool's calls minus the calls used when its rollout
stopped, so savings are never limited to questions with a MAX. Four tables
follow, then one notes block.

1. **API calls versus accuracy.** One row per method, the full pool
   included: mean API calls per question, the share saved, top-1, exact MAX
   (the share of MAX-present questions whose returned candidate is the MAX),
   the share of rollouts stopped at a predicted MAX, and the stop precision
   (the share of those stops that returned the MAX; `--` when a method never
   stops).
2. **Paired differences**, first method minus second, each with a 95%
   bootstrap interval over questions: every method against the full pool,
   then each learned head against each fixed order and the fine-tuned head
   against the pretrained one. Columns are saved calls
   per question, top-1, and exact MAX (over MAX-present questions). Read
   savings and exact MAX side by side to weigh calls saved against MAX
   found; an interval that contains zero is no evidence of a difference.
3. **API calls saved by question group**, as a share of each group's
   full-pool calls: all, MAX present, no MAX, teacher miss, and all wrong,
   plus the share of the method's saved calls that came from no-MAX
   questions. Empty groups show `--`.
4. **MAX recognition outcomes.** Each rollout falls in one row: a MAX stop
   that returned the MAX; a MAX stop on another candidate whose top-1 was
   kept or lost against the full pool in the same zero-shot order; no MAX
   stop although a MAX exists (missed); or no MAX stop and no MAX. Each row
   gives its share of rollouts, its calls saved per question, and its top-1
   lost and gained. The **Total** row adds up to all rollouts, table 1's
   saved calls, and table 2's top-1 change. Every MAX stop on a no-MAX
   question is false; on an all-wrong question it cannot cost accuracy.

The JSON keeps the printed values and more. `policies[method].call_savings`
has five question groups (`all`, `max_present`, `no_max`, `teacher_miss`,
`all_wrong`), each with `top1_correct`, stop rates, and, per cost unit
(`solver_calls` and `total_calls`), calls used and saved, the saved share,
`saved_per_benchmark_question` (groups add up to `all`), savings divided into
correct-MAX, wrong-MAX, and other stops (`mean_saved_by_*_stop`), and the
savings conditional on predicted-MAX stops. `predicted_max_stop_precision` is
`null` for a group without any MAX. `comparisons["A vs B"]` holds table 2's
`saved_calls`, `top1`, and `exact_max` intervals, and
`recognition_outcomes[method]` holds table 4. `decision_training` records each
head's best epoch, development loss, and label coverage.

Total API calls include candidate generations, repeated baseline/cross
measurement solves, and their grading calls. With default settings the
complete pool uses 233 solver calls and 225 graders: **458 API calls**.
Retrieval and target-answer grading are excluded. This accounting follows
the recorded-pool simulator, not observed live traffic. Costs and savings
are averaged across zero-shot orders within each question before aggregating,
so a question is counted once. Empty MAX-present cohorts have null metrics.
Trajectories, recomputed on demand by `WORK.rollout_audit` (see Storage
below), name each acquired evaluator/candidate and its incremental cost.
Every prediction check, including the initial ZS1 and the final state, records
available candidate IDs, the selected candidate, the four MAX/SAFE signals,
and the threshold. These make the reason for stopping inspectable.

Fixed comparison orders affect evaluation only. They are excluded from
training/checkpoint fingerprints, so matching checkpoints of the same
revision can be reused when changing orders and rerunning reports.

Reports keep answer correctness, exact reference-MAX recovery, no-MAX results,
costs and savings, and temporary loss of earlier recognition goals separately.
They also expose exhaustive status counts, actionable coverage by target
rank, evaluator masks, and action names. The run saves teacher/snapshot/head
checkpoints, an inference bundle, compact rollout outcomes, and `results.json`.

**Storage.** Stage 4 rolls out each question once per method and zero-shot
order. With `use_test_files_for_dev_and_audit=True` the pooled audit is the
union of the benchmark files, so each benchmark report reuses the audit's
rollouts instead of repeating them; the continuation diagnostic is shared the
same way. Evaluation rollouts are untraced. As each finishes, its outcome is
appended to `final_rollouts.jsonl`, and the rows are reduced to what the
reports need: per-question means over zero-shot orders (the first step of
`policy_summary`), stop reasons, and per-order correctness for the paired
bootstrap. Each report is then aggregated from those per-question inputs in
its own question order, so `results.json` is byte-identical to the earlier
per-report computation, bootstrap intervals included. Stage 4 prints question
progress, elapsed time, and estimated remaining time like Stage 3.

Each line of `final_rollouts.jsonl` holds only what the models decide:

| Field | Meaning |
| --- | --- |
| `uid`, `method`, `zero_shot_order` | The rollout's question, method, and zero-shot permutation |
| `candidate_mask`, `evaluator_mask` | Final acquired candidate slots (under that permutation) and evaluators |
| `selected` | Returned candidate ID |
| `reason` | Stop reason |
| `final_max_recognized` | All four MAX/SAFE signals pass at the final state |

Everything else is derived. Costs, call counts, and savings follow from the
masks; correctness and exact MAX recovery from `selected` with the records and
teacher labels. Because evaluation stops at the first recognition,
`first_max_recognition` is the final state whenever `reason` is
`predicted_max` (for `full`, whenever `final_max_recognized`). Ranked lists
and trajectories need the models again: `WORK.rollout_audit(name, position)`
(or `question_rollout_audit(...)`) repeats one question's rollouts for every
method and order with tracing (a fraction of a second at the default
geometry), checks them against the saved lines, and returns the rows earlier
revisions wrote. `name` is `"audit"` or a
benchmark's file stem, and `position` indexes that report's questions. Run it
on the reporting device with the reporting configuration: other hardware can
round probabilities differently, and changed fixed orders need a new
`report()` first.

Measured on a synthetic default-geometry evaluation with 1,879 test questions
(the full run's test-file count) in four benchmarks, six zero-shot orders,
five methods, and a small trained model, one local CPU process per version:

| Stage 4 | Before | After |
| --- | --- | --- |
| Rollouts | 112,740, traced (pooled audit, then each benchmark again) | 56,370, untraced |
| Per-rollout file | `final_trajectories.jsonl`, 658 MiB (6.1 KB per row) | `final_rollouts.jsonl`, 11.1 MiB (about 205 bytes per line) |
| Python heap per question | 474 KiB of traced rows, for every question of a report at once | 14 KiB of summary inputs |
| Peak RSS increase during `report()` | 1,230 MiB | 142 MiB |
| Time | about 20 minutes | about 9 minutes |

`results.json` was byte-identical. About 105 MiB of the remaining peak was the
per-report snapshot diagnostic, which then held sampled observations and
logits. It now streams per-question sums over every reachable state and, under
`every_presence_state`, over all 8,160 presence states, each under every
zero-shot order; like the rollouts, each question's sums are computed once and
shared by the pooled audit and its benchmark report. The old writer also needed
room for two copies of its file when an earlier one was present.

Earlier revisions wrote `final_trajectories.jsonl`: every traced row with its
group hash, ranked list, call-count dictionaries, and per-step signals, about
6 KB of JSON and 18 KB of Python objects per row. All five methods' rows of a
report were held in RAM before summarizing, and pooled-audit questions were
rolled out and written twice. Once the new file is complete, an old
`final_trajectories.jsonl` in the output folder is deleted, and the mirror
drops it at the next upload. Stage 0's version line contains `Stage 4 rolls
out each question once and saves compact outcomes to final_rollouts.jsonl, not
trajectories`; a notebook copy without it still writes the traced file.

### Versions

This implementation uses **revision 10**, **contract schema 4**,
**inference-bundle schema 5**, **decision-dataset schema 5** (the label
fingerprint, without input log paths), **label-file schema 6**, and
**dataset-store schema 2** (the dataset holds only the compact records, and its
version hashes only the record settings and the logs; schema 1 also held the
model output of Stages 1-3a, which a run adopts as described in "Datasets and
dataset IDs"; neither changes any label, checkpoint or contract, so neither
needed a new revision). Revision 10
stops Stages 1-2 at the first development check `view_patience` epochs after the
best one, inside an epoch if need be; revision 9 counted whole epochs there,
checked at epoch ends, so its teachers trained every epoch. Revision 9
replaces the single planned label per state with two phases. An evaluator and a
one-shot candidate are separate acquisitions per source (eleven actions instead
of seven), offered for a source in use or the nearest unused one, so 1,092
states (up from 576) are reachable from the first candidate. Stage 3 labels
every presence state except the complete one with its cheapest first steps
(Stage 3a) and then fine-tunes on the step that the pretrained head's own
continuations make cheapest (Stage 3b). The head also reads the acquired-item
masks, and the report evaluates both heads. Revision 8 let a one-shot candidate
be generated without its own evaluator and labeled all 8,160 presence states.
Revision 7 selects Stage 1 and Stage 2 checkpoints by development loss instead
of development AP (Stage 1) and Top-1 then Brier (Stage 2); AP stays a reported
metric. Revision 6 trains Stages 1-2 on exhaustive views instead of random
snapshots. Every earlier output folder is rejected as a different contract, and
older action/state bundles and checkpoints are rejected. Use the new default
output directory `/kaggle/working/adaptive_analogical_shared_training_v10_run` and
a new `hf_repo` mirror for a fresh run; old runs are not migrated. The notebook's `QUICK_PILOT` checks execution, not final model
quality. Recorded outcomes cannot establish live provider behavior or the
results of newly generated candidates.
