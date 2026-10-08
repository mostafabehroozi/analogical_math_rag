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
`WORK_DIR` for each training configuration.

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
