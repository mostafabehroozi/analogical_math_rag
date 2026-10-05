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

The AvalAI adaptation, final-solver, and evaluator model names are runtime
provenance rather than immutable run identity. If a model becomes unavailable,
set the replacement names in every restarted worker and in the finalizer:

```python
CONFIG["AVALAI_MODEL_NAME_ADAPTATION"] = "meta/llama-3.2-11b-vision-instruct"
CONFIG["AVALAI_MODEL_NAME_FINAL_SOLVER"] = "meta/llama-3.2-11b-vision-instruct"
CONFIG["AVALAI_MODEL_NAME_EVALUATOR"] = "openai/gpt-oss-20b"
```

No opt-in flag is required for a model-name-only change when the execution code
is unchanged. A legacy code fingerprint is adopted only after authenticating
its complete source hash against the saved Git revision and comparing worker
source. Tests and unrelated training/report scripts do not affect that proof;
worker modules, configuration, and their local import dependencies do. The
actual runtime fingerprint is recorded separately in worker status.
`DISTRIBUTED_ALLOW_MODEL_ROTATION=True` does not authorize changed worker code.
Do not set `DISTRIBUTED_CODE_FINGERPRINT` manually: workers discard manual pins,
and the Hub helper authenticates source even when creating a new manifest.

Only the three model-name fields above may differ automatically. The stored
manifest is never rewritten, completed questions remain complete, and failed or
pending questions use the active models. Every new or solve-only query log
records its active model configuration. The resulting run is intentionally
mixed-model; keep the original experiment name unchanged so checkpoint artifact
names continue to match.

The optional per-role `AVALAI_REASONING_EFFORT_*` settings use `None` to inherit
`AVALAI_REASONING_EFFORT`. That inherited state is canonically omitted from new
manifests, and a legacy explicit `null` is treated as equivalent to an absent
key during the narrow rotation check. Any explicit per-role reasoning effort,
global reasoning-effort change, or unrelated setting remains a scientific
configuration change and requires a new run id.

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
Model-name rotation retains the existing manifest under the rules above.

## Recovering a code-fingerprint mismatch safely

A mismatch occurs before provider calls. The original revision is inside the
authoritative manifest's `code_fingerprint`, rather than the latest branch HEAD.
If worker behavior changed, resume with the original source or start a separate
run. Setting a fingerprint by hand or overwriting the remote manifest would mix
incompatible results and is not a recovery method.

For `layer1-grouping-k5-size2-v1`, the downloaded manifest identifies revision
`22f35725faf98b6426854260109579026694c6f4`. Its full source hash matches that
commit. Later provider pacing and retry changes affect this AvalAI grouping
worker, so the latest source cannot prove equivalent execution. The run has five
workers and 3,879 questions; input order, answers, corpus, and scientific settings
must still pass the manifest checks after source recovery.

After obtaining the updated repository containing `prepare_distributed_resume.py`,
restart the Kaggle kernel. Run this cell **before importing `config` or `src`**:

```python
import os
import subprocess
import sys
from huggingface_hub import hf_hub_download

active_source = "/kaggle/working/analogical_math_rag"
resume_source = "/kaggle/working/analogical_math_rag-resume-v1"
run_id = "layer1-grouping-k5-size2-v1"
manifest_path = hf_hub_download(
    repo_id="mostafabehroozi/grouping_dist",
    repo_type="dataset",
    filename=f"distributed_runs/{run_id}/manifest.json",
    revision="main",
)
subprocess.run(
    [sys.executable, "-B", f"{active_source}/prepare_distributed_resume.py",
     "--manifest", manifest_path, "--run-id", run_id,
     "--destination", resume_source],
    cwd=active_source, check=True,
)
os.chdir(resume_source)
```

The command authenticates the manifest and saved source before creating a detached
Git worktree. It verifies the new checkout's complete fingerprint, leaves the
active checkout intact, and never changes checkpoints or the Hub. If Git history
is shallow, fetch the saved revision/history and rerun. A source hash containing
uncommitted original edits cannot be recovered from the commit alone and is
rejected. An existing destination is reused only when its source matches exactly.

Use the notebook/setup from the recovered checkout, or your original pasted
setup. Change **both** `os.chdir("/kaggle/working/analogical_math_rag")` lines in
that pasted setup to `os.chdir(resume_source)`. Keep its original provider imports;
the latest notebook's unconditional OpenRouter import is unavailable in this old
revision. Keep your experiment configuration and normal `run_experiments(...)`
cell. Do not import the latest source into this worker's kernel.

To avoid a local manifest left behind by the failed attempt, keep shared data and
embedding paths, but use a fresh local checkpoint base. Immediately after
`setup_kaggle_mode()` and **before** `configure_worker_paths(CONFIG)`, set:

```python
CONFIG["BASE_OUTPUT_DIR"] = "/kaggle/working/resume-v1-state"
CONFIG["DISTRIBUTED_RUN_ID"] = "layer1-grouping-k5-size2-v1"
CONFIG["DISTRIBUTED_WORKER_COUNT"] = 5
CONFIG["DISTRIBUTED_WORKER_ID"] = 0  # each notebook keeps its own original ID, 0..4
CONFIG["DISTRIBUTED_FINALIZER_MODE"] = False
CONFIG["DISTRIBUTED_ALLOW_MODEL_ROTATION"] = False  # disable the old unverified code bridge
CONFIG.pop("DISTRIBUTED_CODE_FINGERPRINT", None)
```

The unchanged remote prefix restores that worker's existing results into the new
local state directory. This source verification does not by itself prove that
the currently loaded question/answer/corpus inputs or settings match; normal
startup validates those before execution. Resume each worker from the same saved
source, with its original ID, and never run that ID concurrently.

To use the latest worker behavior instead, restart all worker kernels on the same
updated checkout and choose one shared new ID, for example
`layer1-grouping-k5-size2-v2-20261005`. Set it before configuring paths. This
creates a separate run and retains v1's remote results; do not copy old shards
into the new run.
