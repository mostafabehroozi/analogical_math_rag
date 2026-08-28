# Distributed Kaggle execution

This mode runs one immutable experiment across independent Kaggle notebooks.
Workers never call one another and there is no live master. Hugging Face is
only the durable checkpoint store.

## One-time setup

1. Create the Hugging Face dataset repository once with the owner account.
2. Give each notebook a token that can write to that repository.
3. Store Hugging Face and provider credentials in Kaggle Secrets/environment
   variables. Do not put credentials in the notebook, run manifest, or config
   committed to Git.
4. Use the exact same code, question order, answers, experiment configs,
   `DISTRIBUTED_RUN_ID`, and `DISTRIBUTED_WORKER_COUNT` in every notebook.

## Worker configuration

For five workers, use the same settings except for `DISTRIBUTED_WORKER_ID`:

```python
CONFIG.update({
    "DISTRIBUTED_EXECUTION_ENABLED": True,
    "DISTRIBUTED_RUN_ID": "gsm8k-zs3-os3-ret3-mir5-v1",
    "DISTRIBUTED_WORKER_COUNT": 5,
    "DISTRIBUTED_WORKER_ID": 0,  # 0, 1, 2, 3, or 4
    "BATCH_PROCESSING_ENABLED": True,
    "BATCH_SIZE": 5,
    "BATCH_MAX_WORKERS": 5,
    "QUESTION_PARALLEL_API_ENABLED": True,
    "QUESTION_PARALLEL_MAX_WORKERS": 3,
    "PERSIST_RESULTS_ONLINE": True,
})
```

The provided notebooks read these names directly from Kaggle Secrets (or
environment variables): `DISTRIBUTED_EXECUTION_ENABLED`, `DISTRIBUTED_RUN_ID`,
`DISTRIBUTED_WORKER_COUNT`, `DISTRIBUTED_WORKER_ID`, `HF_SYNC_TOKEN`, and the
selected provider credential (`AVALAI_API_KEY`, `GEMINI_API_KEY`, or
`GEMINI_API_KEYS_JSON`).

Run the normal `run_experiments(...)` call. For question index `i`, ownership
uses contiguous, balanced ranges. For 2,000 questions and five workers, worker
0 owns indices 0..399, worker 1 owns 400..799, and so on. Global indices and
the full answer list are retained, so no question is renumbered.

After every completed batch, the worker atomically uploads only its own direct
result artifacts and status. The next batch does not start until that upload
succeeds. At the soft session limit (11 hours by default), no new
batch starts; an active batch may finish and checkpoint. The API retry deadline
is 11.75 hours by default, leaving time before Kaggle's hard cutoff.

When a session ends, restart a notebook with the same worker id. It downloads
only the immutable manifest and its own latest shard, skips strictly successful
questions, and retries failed/partial questions. Different workers may run at
different times. Do not run the same worker id concurrently in two notebooks.

### Special dataset and simplification workflows

Transformation-dataset construction, merging-dataset construction, and Core
Simplification Phase 1 use the same three concurrency levels: notebook shards,
question batches, and bounded API-call parallelism within each question. Their
sparse accepted datasets are merged by immutable global question index.

For distributed transformation datasets, use `hard_questions` as the target
source and leave `TRANSFORMATION_DS_MAX_TARGETS` and
`TRANSFORMATION_DS_MAX_MEMBERS` as `None`. If a smaller run is needed, slice the
question/answer inputs identically in every notebook before starting a new run.

Core Simplification Phase 2 depends on the complete Phase-1 donor set. Enable
both phase flags in the same experiment configuration: workers build Phase 1
shards, then the single finalizer merges the donors and runs Phase 2. Running
Phase 2 independently on each worker would create different donor/test pairings
and is therefore intentionally not allowed.

### Emergency model rotation during an existing run

Model names remain immutable by default. If an AvalAI model becomes unavailable
after a distributed run has started, explicitly enable the narrow compatibility
mode in every restarted worker and in the finalizer:

```python
CONFIG["DISTRIBUTED_ALLOW_MODEL_ROTATION"] = True

CONFIG["AVALAI_MODEL_NAME_ADAPTATION"] = "meta/llama-3.2-11b-vision-instruct"
CONFIG["AVALAI_MODEL_NAME_FINAL_SOLVER"] = "meta/llama-3.2-11b-vision-instruct"
CONFIG["AVALAI_MODEL_NAME_EVALUATOR"] = "openai/gpt-oss-20b"
```

When applying this patch to a run created by older source code, the worker
automatically adopts the exact `code_fingerprint` from the valid immutable
manifest after verifying that every other difference is limited to the three
approved model names. The actual patched runtime fingerprint is recorded
separately in worker status. An explicit `DISTRIBUTED_CODE_FINGERPRINT` remains
supported, but is not required; if supplied, it must match the manifest exactly.

With rotation enabled, only the three model-name fields above may differ. The
stored manifest is never rewritten, completed questions remain complete, and
failed or pending questions use the active models. Every new or solve-only query
log records its active model configuration. The resulting run is intentionally
mixed-model; keep the original experiment name unchanged so checkpoint artifact
names continue to match.

## Final merge and Layer 2

After all workers report `COMPLETE`, run exactly one finalizer notebook:

Set `DISTRIBUTED_FINALIZER_MODE=true` in that notebook's Kaggle Secrets. Keep
the same run id, worker count, code, questions, answers, and experiment config.

```python
from src.orchestration import finalize_distributed_experiments

finalized = finalize_distributed_experiments(
    experiment_configs=experiment_configs,
    global_config=CONFIG,
    hard_questions=hard_questions,
    hard_solutions=hard_solutions,
    exemplar_data=exemplar_data,
    api_managers=api_managers,
    embedding_model=embedding_model,
    run_layer2=True,
)
```

The finalizer downloads all shards, then validates worker identity, manifest
hash, ownership, question and answer hashes, successful run status, complete
Layer-1 states, and exact question coverage. If a worker was forgotten, is
paused, or has a missing/conflicting artifact, it raises a clear error and does
not publish `MERGE_COMPLETE.json`. Once validation succeeds, it writes the
legacy-compatible merged run log, sparse dataset outputs, and any
`{metadata, queries}` Layer-1 cache in global numeric order, publishes them
atomically, and runs configured Layer 2 or Core Simplification Phase 2 once
against the merged artifacts.

Remote layout:

```text
distributed_runs/<run-id>/manifest.json
distributed_runs/<run-id>/workers/worker-000/{status.json,results,logs}
distributed_runs/<run-id>/workers/worker-001/{status.json,results,logs}
...
distributed_runs/<run-id>/merged/{MERGE_COMPLETE.json,results}
```

If scientific inputs, code, worker count, question order, or answers change,
choose a new run id. The existing manifest is intentionally never overwritten.
The sole exception is the explicit model-rotation mode above, which still rejects
every change outside its three-field allowlist.
