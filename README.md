# Analogical Math RAG

Research pipeline for retrieval-based analogical mathematical reasoning. The
main entry point is `main_experiment.ipynb`; runtime behavior is controlled by
`config.py`, and implementation modules live in `src/`.

## Setup

Use Python 3.9 or newer and install the declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

Open `main_experiment.ipynb` with `analogical_math_rag/` as the working
directory. The code intentionally imports `config` and `src.*` from that
directory.

## Project map

- `config.py` — providers, models, feature flags, paths, and experiment sizes.
- `main_experiment.ipynb` — environment setup and experiment entry point.
- `src/orchestration.py` — top-level Layer 1/Layer 2 pipeline coordination.
- `src/pipeline_steps.py` — retrieval, adaptation, solving, simplification, and
  validation stages.
- `src/api_manager.py` — provider clients, retry behavior, and rate limiting.
- `src/evaluation.py` — answer evaluation and aggregate metrics.
- `src/benchmark_data.py` — Hugging Face benchmark schemas, downloads, and
  question/ground-truth normalization.
- `src/layer1_grouping.py` - independent all-combinations few-shot generation
  and target-answer evaluation without the Layer 1 CCS matrix.
- `src/layer1_base_execution.py`, `src/layer2_analysis.py`, and
  `src/layer2_integration.py` — cached execution and offline analysis.
- `src/*_dataset_builder.py` — optional dataset-construction workflows.
- `src/utils.py`, `src/batching.py`, `src/parallel_utils.py`,
  `src/context_logger.py`, and `src/hf_sync.py` — shared infrastructure.

## Before a run

1. Select `numina_hard`, `math500`, `gsm8k`, `aime25`, or `aime26` with
   `TARGET_BENCHMARK`, or set `TARGET_BENCHMARKS` to an ordered unique list
   such as `["math500", "gsm8k"]`. A non-empty combined list overrides the
   scalar setting and is not capped by `BENCHMARK_MAX_QUESTIONS`.
2. Configure the selected provider and its model names/credentials.
3. Keep `BATCH_PROCESSING_ENABLED` and `QUESTION_PARALLEL_API_ENABLED` disabled
   for the first small validation run.
4. Set `BENCHMARK_MAX_QUESTIONS` to a small number before a full experiment.
5. Replace the Hugging Face placeholder values or set
   `PERSIST_RESULTS_ONLINE` to `False`.

## OpenRouter

Set `OPENROUTER_API_KEY` as an environment variable or Kaggle Secret. In the
notebook control panel, set the model names for the roles you will use and
select `"openrouter"` with `API_PROVIDER_ADAPTATION`, `API_PROVIDER_SOLVER`,
`API_PROVIDER_EVALUATOR`, or `API_PROVIDER_SIMPLIFICATION`. For example:

```python
CONFIG["OPENROUTER_MODEL_NAME_FINAL_SOLVER"] = "author/model-slug"
CONFIG["OPENROUTER_PROVIDER_ROUTING"] = {
    "order": ["provider-a", "provider-b"],
    "allow_fallbacks": True,
}
CONFIG["OPENROUTER_MODEL_ROUTING"] = {
    "author/model-slug": {"only": ["provider-a"]},
}
CONFIG["OPENROUTER_MODEL_FALLBACKS"] = {
    "author/model-slug": ["other/model-slug"],
}
experiment_configurations[0]["API_PROVIDER_SOLVER"] = "openrouter"
```

Replace the example model IDs and provider slugs with entries from OpenRouter's
catalog. The model-specific routing object replaces the default routing object
for that model. An experiment may override either routing dictionary without
changing another experiment's requests. Optional global and per-role
`OPENROUTER_REASONING_EFFORT` settings control reasoning where the model
supports it. OpenRouter uses the existing request retry, quota, and provider
delay settings.

## Kaggle Benchmarks models

Inside a Kaggle notebook, the `kaggle` provider sends calls through the
`kaggle_benchmarks` model proxy and uses the notebook's free benchmarking
budget; no API key is needed. Set the role model names to keys of
`kaggle_benchmarks.llms` (`list(kbench.llms)` shows them) and select
`"kaggle"` for any role:

```python
CONFIG["KAGGLE_MODEL_NAME_FINAL_SOLVER"] = "google/gemini-2.5-flash"
CONFIG["KAGGLE_MODEL_NAME_EVALUATOR"] = "google/gemini-2.5-flash"
experiment_configurations[0]["API_PROVIDER_SOLVER"] = "kaggle"
experiment_configurations[0]["API_PROVIDER_EVALUATOR"] = "kaggle"
```

`KAGGLE_REASONING_EFFORT` (and its per-role overrides) accepts `"none"`,
`"low"`, `"medium"` or `"high"`. The library currently drops `temperature`
for proxy models, so sampled attempts (Pass@N) use the model default.
`KAGGLE_MODEL_QUOTAS` paces requests (default 60 per minute).

## Provider batch API

`API_BATCHING_ENABLED` (globally or per experiment) makes OpenRouter and
Kaggle send concurrent calls as one batch; other providers ignore it. It can be
switched per provider and per model:

```python
CONFIG["API_BATCHING_ENABLED"] = True        # every model of both providers
CONFIG["API_BATCHING_ENABLED"] = False       # off
CONFIG["API_BATCHING_ENABLED"] = ["kaggle"]  # every Kaggle model, OpenRouter direct
CONFIG["API_BATCHING_ENABLED"] = {
    "kaggle": True,                          # every Kaggle model
    "openrouter": ["openai/gpt-4o-mini"],    # only these OpenRouter models
}
```

Anything not selected is sent directly with the usual retries and parallelism.
It is a transport switch and works alongside the two existing parallel levels:

- `BATCH_PROCESSING_ENABLED` / `BATCH_MAX_WORKERS`: questions run in parallel.
- `QUESTION_PARALLEL_API_ENABLED` / `QUESTION_PARALLEL_MAX_WORKERS`: calls run
  in parallel inside a question.

Every call those workers make at the same time waits up to
`API_BATCH_COLLECT_SECONDS` for others, and up to `API_BATCH_MAX_SIZE` of them
are submitted together. Each call still receives its own response, and a failed
call is retried on its own in a later batch. At most
`API_BATCH_MAX_CONCURRENT_BATCHES` batches are in flight per provider; calls
that arrive meanwhile form the next, larger batch. Because at most
`BATCH_MAX_WORKERS × QUESTION_PARALLEL_MAX_WORKERS` calls are outstanding,
those settings bound the batch size.

- **Kaggle**: a batch is one `task.evaluate` over the prompts with
  `API_BATCH_KAGGLE_N_JOBS` concurrent workers.
- **OpenRouter**: a batch is one
  [Batch API](https://openrouter.ai/docs/batch-quickstart) job, billed at a
  discount. The job is polled every `API_BATCH_POLL_SECONDS`. OpenRouter allows
  up to 24 hours per job, so a job still running after
  `API_BATCH_TIMEOUT_SECONDS` is abandoned and its calls retried, and OpenRouter
  may still bill it. Provider routing may use only `only` and
  `allow_fallbacks`, and `OPENROUTER_MODEL_FALLBACKS` cannot be used.

```python
{
    "experiment_name": "Layer1_Grouping_K5_Sizes_2",
    # ...
    "BATCH_PROCESSING_ENABLED": True, "BATCH_SIZE": 5, "BATCH_MAX_WORKERS": 5,
    "QUESTION_PARALLEL_API_ENABLED": True, "QUESTION_PARALLEL_MAX_WORKERS": 2,
    "API_BATCHING_ENABLED": True, "API_BATCH_MAX_SIZE": 10,
}
```

### Models OpenRouter will not batch

Only OpenRouter models with a batch endpoint can be batched; they are listed at
[openrouter.ai/models?variant=batch](https://openrouter.ai/models?variant=batch).
For any other model, creating the batch fails with HTTP 400, and pinning a
provider cannot help. HTTP 404 means the model has batch endpoints, but none of
them matches the `only` provider pin. When OpenRouter refuses a model in either
way, or the model's routing uses settings the Batch API does not accept, the
run stops with `BatchRefusedError` before the current question's log is saved.
With `BATCH_PROCESSING_ENABLED`, it stops before the current question batch is
committed, which in distributed mode is also its checkpoint. Otherwise each
failed question would be saved as a FAILURE log, and a resumed run counts saved
logs as done. A rerun with a fixed configuration therefore redoes those
questions. Until the kernel restarts, later calls for the refused model and
routing fail at once, without another batch request.

The error names the model, its provider routing and the effective
`API_BATCHING_ENABLED` value, and lists the fixes:

- leave the model out, e.g. `{"kaggle": True, "openrouter": False}`;
- choose a model from the batch list;
- after a 404, pin a provider that serves the model's batch endpoint, or remove
  the pin;
- or set `API_BATCH_FALLBACK_TO_DIRECT = True`. The run then does not stop:
  that model is sent directly for the rest of the session, with one warning, at
  the standard price.

## Layer-1 grouping companion run

Run grouping as its own experiment so its JSON can later be paired with the
ordinary Layer-1 JSON by `target_query_original_hard_list_idx`:

```python
{
    "experiment_name": "Layer1_Grouping_K5_Sizes_2_3",
    "USE_RETRIEVAL": True,
    "TOP_N_CANDIDATES_RETRIEVAL": 5,
    "APPLY_LAYER1_BASE_EXECUTION": False,
    "APPLY_LAYER1_GROUPING": True,
    "LAYER1_GROUPING_ONLY_MODE": True,
    "LAYER1_GROUP_SIZES": [2, 3],
    "N_PASS_ATTEMPTS": 1,
}
```

For K retrieved samples this makes one solver call and one correctness-
evaluation call for every configured combination. For example, K=5 and sizes
`[2, 3]` creates `C(5,2) + C(5,3) = 20` candidates. Grouping does not run
baseline or candidate-conditioned CCS calls.

Runtime data is written below the configured `local_data/` or Kaggle output
directory; it is not source code and should not be edited by hand.

## Merging-model fine-tuning

`merging_finetuning.ipynb` is the dedicated Kaggle GPU workflow for preparing
merging JSON files, QLoRA fine-tuning Qwen3-4B-Instruct-2507, and comparing base
and adapted binary fusion trees. Install its isolated dependencies from
`requirements-merging-finetuning.txt`. The reusable implementation is in
`src/merging_finetuning.py`.
Run the notebook setup cell before the imports. If it detects loaded packages
whose versions differ from the installed files, restart the kernel/session and
run from the top. This prevents stale Transformers tokenizer registries from
requesting missing modules such as `transformers.models.audioflamingo3`.
Evaluation runs all supported external benchmarks sequentially: MATH-500,
GSM8K, AIME 2025, and AIME 2026. `numina_hard` is excluded because Numina is
the construction source. `EVAL_BENCHMARKS` controls the ordered selection;
`EVAL_QUESTION_LIMIT=None` evaluates every eligible question per benchmark.
Each question is evaluated once, using its first reference, and exact normalized
matches to every fine-tuning split are excluded. Under
`WORK_DIR/evaluations/<benchmark>/`, `population_audit.json` records duplicate
references and construction overlaps, `phase_1_results.json` and
`phase_2_results.json` save raw runs, and `two_phase_evaluation_summary.json`
reports accuracy, candidate/transition/resource metrics, and paired effects.
`WORK_DIR/evaluations/benchmark_reports.json` indexes the separate reports.
The final reporting cell rebuilds metrics from saved runs and `report_config.json`
without model calls. It prints side-by-side accuracy, judged/unknown counts,
coverage, paired gains with bootstrap intervals, corrections/regressions, failures,
and trace resources; full mode also prints candidate and fusion diagnostics.
Both fine-tuning notebooks save the readable output as `evaluation_report.txt`
and print a final comparison table across the separate benchmarks. Full JSON
metrics and raw results remain available. The simplification report compares
direct solving, base simplification, and adapted simplification using a fixed
base solver, with separate copy/change behavior and matched-question contrasts.

### Runs longer than one Kaggle session

A Kaggle session ends after 12 hours and takes `/kaggle/working` with it. Both
fine-tuning notebooks therefore recover every long stage from Hugging Face:
running the notebook again from the top in a new session continues each stage
where it stopped. A read-only "Check Hugging Face for progress" cell prints what
the Hub already holds for training, the finished adapter, and each benchmark
before any GPU work starts.

| Stage | Saved to | Resumes from |
|---|---|---|
| Training | private dataset repo `<user>/<workflow>-qwen3-4b-training-state` (`HF_TRAIN_*`) | the newest Trainer checkpoint: adapter, optimizer, scheduler, RNG and data position |
| Finished adapter | model repo `<user>/<workflow>-qwen3-4b-qlora` (`HF_MODEL_REPO_ID`) | the adapter; training is skipped |
| Evaluation | private dataset repo `<user>/<workflow>-qwen3-4b-evaluation` (`HF_EVAL_*`) | every saved generation and judgment, per question |

Everything is matched by an **identity**, never by name alone, so changed inputs
cannot resume stale work:

- A training checkpoint resumes only when the base model, QLoRA settings (epochs,
  learning rate, LoRA rank, seed, maximum length, ...) and the exact tokenized
  training rows match. Output paths, GPU index and `checkpoint_steps` do not
  count, so saving more often never discards progress.
- A trained adapter carries `training_source.json` (data hashes and QLoRA
  settings). An uploaded adapter from other data or settings stops the notebook;
  the merging notebook accepts adapters uploaded before this file existed with a
  warning because their data cannot be verified.
- An evaluation checkpoint is identified by the adapter, base-model revision,
  data, source files, prompts, generation budgets and judge settings. When
  nothing matches but the Hub holds progress for that benchmark under another
  identity, the notebook prints that identity and the keys that differ (for
  example `code_sha256.merging_finetuning.py`) instead of silently starting over.
  The identity hashes the raw bytes of the listed source files, so keep the
  project files byte-identical across the sessions of one run (a CRLF upload
  and an LF git clone are different bytes), and finish a partly evaluated run
  with the code that started it.

How training recovery works (`src/training_checkpoints.py`):

- After every Trainer save, the new `checkpoint-N` and the best-validation
  checkpoint are published in a single Hub commit that also deletes superseded
  checkpoints, so the repository never holds a half-written checkpoint. At most
  two checkpoint folders are kept online (estimated at roughly 0.2 GB each for
  the default LoRA: adapter plus 8-bit optimizer state; not yet measured on a
  real run).
- A new session restores the newest matching checkpoint at a pinned repository
  revision, verifies every file's size, rewrites Trainer's absolute
  `best_model_checkpoint` path for the new machine, and continues with
  `resume_from_checkpoint`. A local checkpoint wins when it is newer; local
  checkpoints from another configuration are moved to `_untrusted_checkpoints/`
  rather than resumed or deleted.
- An upload failure is reported and retried at the next save instead of ending
  training. Transient Hub errors (rate limits, 5xx, dropped connections) are
  retried with a 5 s / 20 s / 60 s backoff for training and evaluation
  checkpoints and for adapter upload and restore; authentication and permission
  errors still stop immediately.
- Every uploaded checkpoint is a new Hub object, so the training-state repository
  grows with each save. Delete it once the adapter is uploaded.
- `RESUME_CHECKPOINT` remains a manual override: when set, the Hub is not
  consulted for training.

`test_training_checkpoints_trainer.py` runs a real Trainer on a tiny PEFT model and
checks that a session lost mid-training, restored from the (fake) Hub into an empty
directory, finishes with the same weights and best checkpoint as an uninterrupted
run. It needs `torch`, `transformers` and `peft` and is skipped otherwise.

## Simplification-model fine-tuning

`simplification_finetuning.ipynb` trains a Kaggle QLoRA adapter from an
existing Phase 1 `<experiment_name>_run_log.json`. Configure its path in a
Hugging Face dataset repository before running the notebook. Accepted proxies
become simplification targets; rejected and failsafe cases become exact-copy
targets. Repeated questions receive one label: a successful proxy takes
priority, and the loader chooses the largest recorded score gain among
successful proxies. It uses the merging notebook's dependency guard, so run the
setup cell first and restart when it reports stale imports. `HF_TOKEN` and
`AVALAI_API_KEY` come from Kaggle Secrets.

The best adapter is uploaded to its own model repository,
`<user>/simplification-qwen3-4b-qlora` by default, together with
`training_source.json`. That file records the run-log hash, instruction, split,
and QLoRA settings. With `HF_REUSE_ADAPTER_IF_AVAILABLE=True` a later session
restores the adapter and skips training, and a mismatched
`training_source.json` stops the notebook. Use a new `HF_MODEL_REPO_ID` and
`WORK_DIR` for each training configuration. Training itself also survives a lost
session: every Trainer checkpoint is mirrored to
`<user>/simplification-qwen3-4b-training-state` and a rerun resumes from it (see
"Runs longer than one Kaggle session" above).

Evaluation compares direct solving with base and adapted simplification using a
fixed local base solver. With `EVAL_HELDOUT=True` it first evaluates the
labeled construction test split as `heldout`, where copy and change rates can
be read against the targets; judging routes to Numina, and failsafe rows
without a ground truth contribute behavior only. It then evaluates the four
external benchmarks sequentially. `MAX_EVAL_QUESTIONS=None` evaluates all
eligible questions per population; `EVAL_BENCHMARKS` controls the ordered
external selection. Exact normalized matches to every construction split are
excluded, and external copy/change rates are reported as unlabeled behavior.

Evaluation is resumable across Kaggle sessions. Successful generations and
judgments are checkpointed per question and synced to a private dataset
repository, `<user>/simplification-qwen3-4b-evaluation` by default, under
`simplification_evaluations/`. A new session restores them, skips completed
questions, and retries only crashed generations and failed judgments. Truncated
or empty outputs recur under greedy decoding, so they stay final and count as
unknown. The checkpoint identity covers the adapter weights, base-model
revision, run log, rendered prompts, generation budgets, and judge settings.
Increase `protocol_version` in the evaluator cell after changing evaluation
logic that the identity cannot see. Each population has
`population_audit.json`, `evaluation.json`, and `summary.json` under
`WORK_DIR/evaluations/<population>/`; the report index is
`WORK_DIR/evaluations/benchmark_reports.json`.
