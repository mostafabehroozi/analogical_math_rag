# Adaptive analogical training: three supervised stages

The executable workflow is in `adaptive_analogical_training.py`. The self-contained
`adaptive_analogical_training_kaggle.ipynb` embeds the same definitions. Both read
recorded Layer-1 `_run_log.json` files. Training and offline evaluation make no
provider calls and cannot establish what a new live generation would produce.

## What adapts

This workflow learns how much analogical evidence to acquire, and which
candidate to generate next. It does not fine-tune the mathematical solver.
The first network defines a reference priority, the second estimates what
the currently visible evidence supports, and the head learns economical
actions toward the specified recognition goals.

Full evidence is the chosen reference for label construction. It is not a
guarantee that a probability is accurate, or that more evidence always raises
confidence. This is why earlier goals must be checked at every intermediate
step and why evaluation measures actual answer correctness separately from
MAX recognition. The neural inputs contain measurement values and masks,
not question or solution text. Before the first evaluator is acquired,
questions with the same structural state are indistinguishable to the head;
its initial action therefore reflects a learned population-level preference.

## Evidence and the three stages

For each target question, the default record has three zero-shot candidates,
five retrieved solved examples, and one one-shot candidate generated from each
retrieved example. A retrieved example can evaluate existing candidates through
Base-CCS and candidate-conditioned CCS. The stored correctness labels are used
for training and offline audit, never as model input during acquisition.

1. **Reference ranking model.** A joint ResNet sees a 223-feature observation
   of the candidate pool and predicts one correctness probability per candidate.
   Its eight outputs have no SAFE/MAX heads. It trains on complete and masked
   observations, while a complete observation for every training question is
   retained in every epoch. Its complete-state probabilities define the
   reference candidate order. Out-of-fold predictions are used for supervised
   training questions.
2. **Partial-state predictor.** A second ResNet uses the same observation and
   masking rules. It predicts eight correctness values, eight SAFE memberships,
   eight MAX memberships, and global SAFE/MAX presence: 26 outputs by default.
   Development-only temperature calibration is fitted before the model and its
   128-dimensional representation are frozen.
3. **Supervised acquisition head.** A small `128 → 64 → 7` head learns which
   valid item to acquire next. It receives only the frozen representation of
   visible evidence. There is no STOP output, DQN reward, replay buffer, or
   Bellman target in this training stage.

SAFE is the leading uninterrupted run of actually correct candidates in the
first model's order. MAX is its first member. If the first-ranked candidate
is wrong, SAFE and MAX are both empty, even if a lower-ranked candidate is
correct. These are historical supervision definitions, not deployment truth.

## Observations and augmentation

Every observation contains presence masks alongside similarities, baselines,
and CCS values. A missing measurement therefore differs from a measured zero.
Training independently hides candidates and evaluators, and also samples
missing individual measurements and aligned slot permutations. It can retain a
one-shot candidate while hiding its source evaluator as a training-only
augmentation. Runtime acquisitions never generate a one-shot candidate before
its source evaluator is available. Candidate identity, source provenance,
CCS rows, and correctness/SAFE/MAX labels move together under permutations.

The first model's loss and the second model's candidate-specific losses exclude
absent candidates. The complete-state ranking is an **offline guide** for
constructing labels. It is not passed to the deployed acquisition head.

## Acquisition actions and costs

The initial acquisition state contains one zero-shot candidate and no
retrieved evaluator. Retrieval is sorted by descending cosine similarity,
with original log order breaking ties. The seven default actions are:

| Action | Effect |
| --- | --- |
| `add_next_retrieved` | Acquire the next retrieved evaluator in similarity order. |
| `add_zero_shot` | Generate the next zero-shot candidate; its result is not chosen by the head. |
| `one_shot_source_1` through `one_shot_source_5` | Generate a one-shot candidate from that already active source. |

Invalid actions, including an unavailable one-shot source or an exhausted
budget, are masked. With five repeated solves per CCS estimate, the default
complete pool costs 233 solver calls: eight candidate generations, 25 baseline
solves, and 200 candidate-conditioned solves. The initial zero-shot candidate
costs one call. `cost_unit="total_calls"` additionally counts evaluator
grading calls according to the configured cost model.

## How the decision dataset is built

For each `policy` or `dev` question, the builder uses the first model's
complete-state ranking, historical SAFE/MAX labels, and the frozen second
model's predictions on acquisition states. For three zero-shot candidates it
enumerates all six consistent generation orders. The ordinary action graph
contains 189 structural states per order with the default five evaluators,
before any budget cap. A limited training-only sample of source-hidden states
adds robustness where valid future actions and goals can be defined.

An acceptable first state has the exact reference MAX present and its global
MAX, MAX-i, global SAFE, and SAFE-i probabilities all at least 0.5. When MAX
is absent, the first-ranked candidate must instead be present and have a
partial-state correctness probability strictly greater than the frozen
second model's full-state probability for every lower-ranked candidate.

For later ranks, the candidate must be present. By default, it must also
beat every lower-ranked candidate's full-state probability. This later-rank
comparison can be disabled, but the no-MAX first-rank comparison always
applies. A later SAFE member additionally needs SAFE-i at least 0.5. Earlier
goal conditions must remain satisfied at **every step** toward a later goal;
an action that turns an earlier required light off is not on an acceptable
path.

The builder searches valid additions for the lowest additional call cost to
the current goal. Across still unseen historical zero-shot orders, it must
choose the same action whenever the model's visible observation is the same.
This constraint applies at every future step, not just the next acquisition.
It can branch only after acquired measurements distinguish those orders;
candidate IDs or knowledge of the next zero-shot result cannot distinguish
them. The original reference ranking, including tied scores, remains fixed
when candidate slots are permuted.

It first maximizes the probability of reaching the goal, then minimizes
expected search cost and then expected action count. Search cost includes
calls in unsuccessful orders and ends at goal attainment, violation of an
earlier goal, or action exhaustion. These are offline search boundaries,
not deployment stopping rules. Equally best actions receive a shared soft
target; a concrete tied continuation uses the lowest action index.
States already satisfying the current goal move to the
next priority without a redundant acquisition label. States with no
reachable acceptable goal receive no invented action label; their count is
reported, including exhausted states with unmet goals. Artificial source-hidden
rows are capped at 10% of ordinary labeled rows and receive weight 0.1;
ordinary rows receive weight 1.0.

Each saved decision row records the visible-state key, question ID, frozen
representation, valid-action mask, soft action target, expected path cost,
reachability, per-action reach/cost, and goal rank for audit. The goal rank, reference ranking,
historical truth, and future measurements are **not** action-head inputs.
Rows from one question and every zero-shot order stay in one question split.

## Training, evaluation, and artifacts

The four internal roles are `supervised` (60%), `policy` (20%), `dev` (10%),
and `audit` (10%). The first two models train on `supervised`. The head trains
on `policy`, with checkpoint selection by masked supervised loss on `dev`.
`audit` and external benchmarks are reserved for reporting. A bundle from
the prior DQN schema cannot be resumed into this version 3 experiment; use a
new output directory.

Offline reports compare three methods on the same recorded zero-shot orders:

1. **Complete pool:** acquire all measurements, then select the second
   model's highest-correctness candidate.
2. **Fixed order:** `ZS1 → R1 → OS1 → ZS2 → R2 → OS2 → ZS3 → R3 → OS3 → R4
   → OS4 → R5 → OS5`.
3. **Supervised head:** choose each valid next acquisition from the frozen
   representation and trained seven-action head.

For the fixed and learned methods, the application ends acquisition when
global MAX and SAFE and the currently selected candidate's MAX-i and SAFE-i
all reach 0.5, or when no budgeted action remains. This is separate from
decision-head training. The same observable rule is applied on no-MAX
questions, so a false predicted MAX can end a run and is reported as an
error. Metrics include actual selected-answer correctness on all questions,
exact reference-MAX recovery where MAX exists, no-MAX results separately,
mean and high-percentile calls, and percentage saved from full-pool cost.
An offline continuation diagnostic checks progress through later ranks
without applying the evaluation stop rule. It covers the same zero-shot
orders and reports results at the question level. A path that loses an
earlier goal is invalid; later recovery cannot inflate its reported progress.
The fixed-order baseline stops if its next scheduled step is unaffordable.

The run saves a contract, data audit and split manifest, teacher fold and
completed checkpoints, a frozen snapshot checkpoint, decision datasets for
`policy` and `dev`, a completed head checkpoint, `inference_bundle.pt`,
`results.json`, and `final_trajectories.jsonl`. Completed stages resume only
under the same version 3 configuration, implementation revision, and data
contract. Revision 2 corrects planning under unseen zero-shot orders; an
earlier completed run needs a new output directory rather than reusing its
old labels or checkpoints. The notebook's
`QUICK_PILOT` checks execution cheaply; it is not a paper-quality run.

Cached evaluation reveals recorded results. It cannot reconstruct different
candidate generations, prompt-dependent counterfactuals, or real provider
failures. In particular, an offline planner can use complete historical
information to create labels that the deployed head cannot know exactly.
Held-out accuracy and cost, followed by live validation, determine whether
those labels teach a useful acquisition policy.
