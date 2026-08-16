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
    "PERSIST_RESULTS_ONLINE": True,
})
```

The provided notebooks read these names directly from Kaggle Secrets (or
environment variables): `DISTRIBUTED_EXECUTION_ENABLED`, `DISTRIBUTED_RUN_ID`,
`DISTRIBUTED_WORKER_COUNT`, `DISTRIBUTED_WORKER_ID`, `HF_SYNC_TOKEN`, and the
selected provider credential (`AVALAI_API_KEY`, `GEMINI_API_KEY`, or
`GEMINI_API_KEYS_JSON`).

Run the normal `run_experiments(...)` call. For question index `i`, ownership
is `i % worker_count`; with five workers this is a balanced 20% share when the
question count is divisible by five. Global indices and the full answer list
are retained, so no question is renumbered.

After every completed batch, the worker atomically uploads only its own run
log, Layer-1 cache, and status. The next batch does not start until
that upload succeeds. At the soft session limit (11 hours by default), no new
batch starts; an active batch may finish and checkpoint. The API retry deadline
is 11.75 hours by default, leaving time before Kaggle's hard cutoff.

When a session ends, restart a notebook with the same worker id. It downloads
only the immutable manifest and its own latest shard, skips strictly successful
questions, and retries failed/partial questions. Different workers may run at
different times. Do not run the same worker id concurrently in two notebooks.

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
    run_layer2=True,
)
```

The finalizer downloads all shards, then validates worker identity, manifest
hash, ownership, question and answer hashes, successful run status, complete
Layer-1 states, and exact question coverage. If a worker was forgotten, is
paused, or has a missing/conflicting artifact, it raises a clear error and does
not publish `MERGE_COMPLETE.json`. Once validation succeeds, it writes the
legacy-compatible merged run log and `{metadata, queries}` Layer-1 cache in
global numeric order, publishes them atomically, and runs configured Layer 2
once against the merged cache.

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
