# Reinforcement learning in this project: from the broad view to the training loop

This guide is for an AI engineer who knows supervised learning and wants to understand the reinforcement learning (RL) part of this repository step by step. The implementation is in [`adaptive_analogical_training.py`](adaptive_analogical_training.py), and Stage 3 of [`adaptive_analogical_training_kaggle.ipynb`](adaptive_analogical_training_kaggle.ipynb) runs it. The notebook contains its own copy of the implementation.

## Level 0: The broad picture

The RL component answers one question:

> Given the evidence collected so far, what should we acquire next to improve the final answer without spending too many calls?

The RL network does not solve the math problem or judge candidate correctness directly. A supervised network estimates which available candidate looks best. A fixed application rule decides whether there is enough evidence to stop. When more evidence is needed, the RL network chooses the next acquisition.

```text
Recorded complete question
        ↓
Supervised teacher learns from complete evidence
        ↓
Supervised snapshot predictor learns from partial evidence
        ↓
Freeze the snapshot predictor
        ↓
At each partial snapshot:
  predictor ranks available answers
  → fixed rule checks whether to stop
  → if continuing, RL head chooses what to acquire
  → new evidence changes the snapshot
  → repeat
```

## Level 1: How RL differs from supervised learning

In supervised learning, you normally train on pairs such as `(features, correct_label)`. You ask: **What is the right prediction for this input?**

Here, the question is sequential: **What should I do now, knowing that my choice changes the evidence I will have for the next decision?**

An evaluator might cost calls now and provide no immediate improvement. But its result could reveal that the current favorite answer is weak, or enable a useful candidate on the next step. We therefore care about the **eventual answer and total cost**, rather than whether one isolated acquisition looks good.

| RL term | Meaning in this project |
|---|---|
| **Agent** | The small `ActionHead` neural network |
| **Environment** | A cached historical question whose recorded evidence can be revealed |
| **Observation** | The partial evidence acquired so far, encoded by a frozen network |
| **Action** | One bundle of new candidate or evaluator evidence |
| **Reward** | Acquisition cost penalty, plus a reward for a correct final answer |
| **Episode** | Decisions for one question, from the initial evidence until stopping |
| **Policy** | The rule that selects an action from the network's values |

The cached record holds complete evidence and correctness labels so training can simulate outcomes and score the final answer. The agent receives only **currently revealed** evidence. See [`observation()`](adaptive_analogical_training.py#L699).

## Level 2: What an episode looks like

The default initial state already has one zero-shot candidate, `ZS1`, and one active evaluator, `R1`. The model examines the evidence available at that point.

If the fixed stop rule is satisfied, it returns the highest-ranked available candidate. Otherwise, the RL head chooses an acquisition. The environment reveals the corresponding recorded measurements, the supervised predictor processes the new partial snapshot, and the cycle repeats.

For example:

```text
ZS1 + R1 are available
    ↓
RL chooses "activate R2"
    ↓
R2 measurements are revealed
    ↓
Predictor updates its candidate ranking
    ↓
Still insufficient evidence
    ↓
RL chooses "acquire R2's one-shot candidate"
    ↓
Predictor ranks that candidate first; stop rule fires
    ↓
Return that answer and score whether it was correct
```

An earlier action can be useful because it enables a later one. That is the central reason to use RL here.

## Level 3: What exactly can the agent choose?

With the default five retrieved sources, the head emits **seven numbers**, one for each action:

| Output | Action |
|---:|---|
| 0 | Activate the next evaluator in retrieval order |
| 1 | Acquire the next zero-shot candidate |
| 2–6 | Acquire the one-shot candidate for retrieved sources 1–5 |

These actions acquire **bundles**. Activating an evaluator reveals its baseline measurements and its CCS measurements against candidates already acquired. Adding a candidate reveals its generation result and CCS measurements against active evaluators. See [`raw_next_state()`](adaptive_analogical_training.py#L664).

An action may be unavailable because its candidate was already acquired, its source has not been activated, the pool is exhausted, or its resulting cost exceeds the budget. [`valid_actions()`](adaptive_analogical_training.py#L684) builds a Boolean mask. The code applies that mask during exploration, greedy choice, **and** calculation of future training targets.

The seven outputs are **Q-values**, not probabilities. A Q-value estimates the future reward from taking that action and then continuing well.

## Level 4: What enters the RL neural network?

The supervised snapshot predictor takes a partial-evidence feature vector. With current defaults, it has **223 inputs**: masks showing what exists or was attempted, observed similarity and evaluation measurements, candidate information, spending, and remaining budget. Missing measurements are represented distinctly from observed zero values.

That predictor produces a **128-dimensional hidden vector** `h` and **26 prediction outputs**. The prediction outputs help rank candidates and apply the stop rule. The RL head receives `h`, **before** that final prediction layer; it does not receive the 26 output probabilities as its learned input.

Before RL training, the predictor is put in evaluation mode and frozen. Only this small head is optimized:

```text
128-dimensional frozen h
        ↓
Linear(128, 64) → ReLU → Linear(64, 7)
        ↓
seven action Q-values
```

That is 8,711 trainable parameters under the default dimensions. See [`FrozenPredictor`](adaptive_analogical_training.py#L917) and [`ActionHead`](adaptive_analogical_training.py#L1012).

## Level 5: How does an action receive a reward?

The environment charges for the **incremental** acquisition. Under default solver-call accounting, with `n` acquired candidates, `k` active evaluators, and five probe attempts:

```text
cost(n, k) = n + 5 × k × (n + 1)
```

The initial `ZS1 + R1` state represents **11 calls**. The complete eight-candidate, five-evaluator pool represents **233 calls**. These are accounting units for cached bundles, not measured runtime or API prices. See [`acquisition_cost()`](adaptive_analogical_training.py#L351).

Every action gets a small negative reward proportional to its extra cost. When the episode stops, the environment adds **1 if the answer actually returned is correct**, otherwise 0:

```text
reward ≈ −0.20 × incremental_cost / 233
         + terminal_answer_correct
```

Thus acquiring another evaluator from the initial state costs 10 calls and gives about `−0.0086` immediately. It may still be a good action if it helps produce a correct final answer.

The code trains two variants. The **baseline** uses the cost and final-correctness reward above. The **refined** variant additionally rewards having a correct currently selected answer earlier across the available budget. That extra term is controlled by `early_quality_weight = 0.10`. The exact implementation is [`CachedEnvironment.step()`](adaptive_analogical_training.py#L978).

Correctness labels are used by this training environment to calculate reward. They are **not** exposed to the action head in its observation.

## Level 6: How does a final result teach an earlier action?

Suppose activating `R2` costs reward `−0.0086`. After seeing `R2`, the best next action has an estimated future value of `0.82`. The first action's learning target is approximately:

```text
−0.0086 + 1.0 × 0.82 = 0.8114
```

So an action with an immediate cost can have a high **total future value**.

This is the Bellman idea. For a recorded transition `(h, action, reward, next_h, done)`:

```text
target = reward                                      if done
target = reward + γ × best valid future Q-value      otherwise
```

Here `γ = 1.0` by default. The head is trained to bring its Q-value for the **action actually taken** closer to this target. The loss is Huber loss, called `smooth_l1_loss` in the code. See [`bellman_target()`](adaptive_analogical_training.py#L1042) and the update in [`fit_dqn()`](adaptive_analogical_training.py#L1200).

This is **Deep Q-Network learning**, or DQN: a neural network approximates a table of “how valuable is action `a` in observation `h`?”

## Level 7: Why replay, exploration, and two Q-networks?

Three mechanisms make the training loop practical:

1. **Exploration.** Early in training, the head's Q-values are unreliable. With probability `ε`, the agent chooses a random *valid* action; otherwise it chooses the highest-Q valid action. `ε` declines from about 1.0 to 0.05 over the first 30% of training steps. During evaluation, action choice is greedy.
2. **Replay buffer.** Each experience stores `(h, action, reward, next_h, done, next_valid_mask)`. Training samples random batches from this buffer, reusing and mixing experiences from different questions. Replay cannot create outcomes that the logs do not contain.
3. **Target network.** The *online* head receives gradient updates. A separate *target* copy supplies the future Q-value in the learning target and is refreshed every 500 optimizer updates. This avoids changing both sides of the learning equation on every update.

The current default is **standard masked DQN**. An optional `double_dqn=True` setting makes the online head choose the next action while the target head values it. The architecture stays the same. The training loop is at [`fit_dqn()`](adaptive_analogical_training.py#L1174).

## Level 8: Where does supervised training fit?

The RL head can work only after a predictor can interpret partial evidence. The notebook therefore proceeds in stages:

1. **Prepare records.** Parse eligible complete historical pools and divide questions into supervised, policy, development, and audit roles.
2. **Train a full-evidence teacher.** It learns candidate rankings. Together with offline correctness labels, those rankings create retrospective `SAFE` and `MAX` supervised targets.
3. **Train the partial-snapshot predictor.** It learns from snapshots that reveal only some candidates and measurements, then is frozen.
4. **Train the RL action head.** It practices acquisition sequences on the separate policy questions using the cached environment.
5. **Evaluate.** Compare the learned acquisition policy with evaluator-first, candidate-first, random, and cheapest-action policies on held-out questions.

`SAFE` and `MAX` are **supervised targets for the predictor**, not RL rewards or guarantees that an answer is correct. The application uses predicted values and thresholds to decide when to stop; the DQN has no STOP action. See [`teacher_labels()`](adaptive_analogical_training.py#L505), [`stopping_reason()`](adaptive_analogical_training.py#L945), and the [workflow stages](adaptive_analogical_training.py#L1470).

## Level 9: What is actually selected and measured?

Training runs both reward variants for 20,000 environment steps by default. Every 1,000 steps, it runs the current head on development questions and retains the best checkpoint. One precise code detail: [`policy_rank()`](adaptive_analogical_training.py#L1168) ranks **Top-1 answer accuracy first**, then uses early quality minus cost as a tie breaker. Some prose in the repository calls this selection “utility”; the current implementation is this lexicographic ranking.

The report examines answer accuracy, mean and tail acquisition cost, confident-stop mistakes, forced stops, and question-level trajectories. A high Q-value by itself does not show success. The meaningful test is whether the resulting policy chooses correct answers at acceptable cost on held-out questions, compared with the fixed policies.

There is also a firm limit to the experiment: this environment **reveals recorded outcomes from complete historical pools**. It cannot invent a new candidate, prompt response, or evaluator result that was never recorded. A cached evaluation can establish behavior under that simulation; it cannot by itself establish live API performance.

## One sentence to retain

> The supervised predictor says what the current evidence suggests; the application decides whether to answer; when it needs more evidence, the RL head learns which valid acquisition is most valuable for the eventual answer after its cost.

For a slower read alongside the code, see the repository's [RL foundations lesson](RL_FOUNDATIONS_AND_ADAPTIVE_ACQUISITION.md). This guide explains the checked-out implementation; it does not claim that a full Kaggle training run has demonstrated improvement.
