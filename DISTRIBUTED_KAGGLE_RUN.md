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
is unchanged. A saved code fingerprint is adopted only after authenticating
its complete source hash against the saved Git revision and comparing worker
source. The proof covers `config.py`, `src/orchestration.py` (the
`run_experiments` entry module), and every local module they import, directly
or transitively, including imports inside functions. Tests, notebooks, the
fine-tuning and reporting modules under `src`, and other scripts the worker
never imports do not affect it. The actual runtime fingerprint is recorded
separately in worker status and in each run log's `worker_code_history`.
`DISTRIBUTED_ALLOW_MODEL_ROTATION=True` does not authorize changed worker code;
`DISTRIBUTED_ALLOW_WORKER_CODE_CHANGE=True` does (see below).
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
The manifest hashes every Python file, so a restarted notebook that clones a
newer commit almost always has a different fingerprint. That alone does not
block a resume: the worker reconstructs the saved revision from Git history and
compares only the worker import closure described above. Two outcomes exist:

- **Only unrelated files changed** (tests, notebooks, fine-tuning or reporting
  modules, root-level tools). The resume proceeds automatically; the manifest's
  code identity is pinned and nothing needs to be configured.
- **Worker code changed** (the error lists the modules). The run stops unless
  you choose one of the options below.

### Resume with changed worker code (normal case)

Set this in every restarted worker notebook, before `run_experiments(...)`:

```python
CONFIG["DISTRIBUTED_ALLOW_WORKER_CODE_CHANGE"] = True
```

The worker keeps the existing run ID, manifest, and checkpoints, pins the
manifest's code identity, and resumes with the current source. The accepted
change is written to the worker's `status.json` (`worker_code`) and to the
`worker_code_history` of every run log the session touches, so the finalized
run records exactly which code produced which results. Scientific inputs,
settings, question order, answers, and corpus are still validated strictly;
the flag never hides those. The manifest is never rewritten.

Leave the flag off when a run must stay on one exact code revision; then use
the isolated original checkout described next, or start a new run ID. Setting
a fingerprint by hand or overwriting the remote manifest would mix
incompatible results and is not a recovery method.

### Resume `layer1-grouping-k5-size2-v2` after a Kaggle restart

This run's remote manifest identifies original Git revision
`7f6b91217973b08e0f38ac86d5e59b5b12111953`. Since then `src/batching.py`
gained a real worker fix (unchanged simplification proxies no longer halt a
batch), and this guide's provenance recording touched `config.py` and
`src/orchestration.py`; the fine-tuning modules that the old proof also listed
are no longer part of it. Resume by setting
`CONFIG["DISTRIBUTED_ALLOW_WORKER_CODE_CHANGE"] = True` in each worker
notebook as described above. The rest of this section is the alternative for
running the exact original source instead.

A fresh kernel alone cannot provide the original source if startup clones the
newer commit again. Keep the existing run ID and use the authenticated original
checkout for every subsequent worker session.

After cloning the current repository, put this cell **before the pasted setup
cell and before any `config` or `src` import**. Give the notebook read access to
`mostafabehroozi/grouping_dist` through the `HF_SYNC_TOKEN` Kaggle Secret,
`HF_SYNC_TOKEN`/`HF_TOKEN` environment variable, or an existing Hugging Face
login. The token is never printed or stored in the checkout.

```python
%run /kaggle/working/analogical_math_rag/kaggle_resume_v2_bootstrap.py
```

The Kaggle checkout must include this new bootstrap file. Until that checkout
has the updated repository, paste the file's contents into the first notebook
cell instead of using `%run`.

The bootstrap downloads
`distributed_runs/layer1-grouping-k5-size2-v2/manifest.json`, invokes
`prepare_distributed_resume.py` in a separate process, and switches the
notebook to `/kaggle/working/analogical_math_rag-resume-v2` only after the
saved source is verified. It stops if `config` or `src` was already imported.
If the saved Git commit is absent, fetch its history into the active checkout
and rerun the bootstrap in a fresh kernel. A mismatched saved source hash must
be investigated rather than overridden.

In the pasted setup, replace **both** occurrences of
`os.chdir("/kaggle/working/analogical_math_rag")` with:

```python
os.chdir(RESUME_SOURCE)
```

The first occurrence matters because it precedes a `src` import. The second
would otherwise switch the notebook back to the changed source. Keep the
original setup's project imports, experiment configuration, question order,
answers, exemplar data, and model settings. Immediately after
`setup_kaggle_mode()` and before `configure_worker_paths(CONFIG)`, use a clean
local checkpoint base for the recovered worker:

```python
CONFIG["BASE_OUTPUT_DIR"] = "/kaggle/working/resume-v2-state"
CONFIG["DISTRIBUTED_RUN_ID"] = "layer1-grouping-k5-size2-v2"
CONFIG["DISTRIBUTED_WORKER_COUNT"] = 5
CONFIG["DISTRIBUTED_WORKER_ID"] = 0  # Retain this notebook's original ID, 0..4.
CONFIG["DISTRIBUTED_FINALIZER_MODE"] = False
```

Keep shared data and embedding paths at their original locations. The worker
will validate the saved manifest and restore its own remote shard into this
local state directory. Do not run the same worker ID in two notebooks at once.
On every later Kaggle restart, repeat the bootstrap with the same run ID and
worker ID; the verified detached checkout is reused if it still matches. The
normal `run_experiments(...)` cell needs no change. Do not set
`DISTRIBUTED_CODE_FINGERPRINT` manually or edit the remote manifest.

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
