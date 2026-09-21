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
