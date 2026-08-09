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
- `src/pipeline_steps.py` — retrieval, adaptation, solving, augmentation, and
  validation stages.
- `src/api_manager.py` — provider clients, retry behavior, and rate limiting.
- `src/evaluation.py` — answer evaluation and aggregate metrics.
- `src/layer1_base_execution.py`, `src/layer2_analysis.py`, and
  `src/layer2_integration.py` — cached execution and offline analysis.
- `src/*_dataset_builder.py` — optional dataset-construction workflows.
- `src/utils.py`, `src/batching.py`, `src/parallel_utils.py`,
  `src/context_logger.py`, and `src/hf_sync.py` — shared infrastructure.

## Before a run

1. Select the benchmark and feature flags in `CONFIG`.
2. Configure the selected provider and its model names/credentials.
3. Keep `BATCH_PROCESSING_ENABLED` and `QUESTION_PARALLEL_API_ENABLED` disabled
   for the first small validation run.
4. Set `BENCHMARK_MAX_QUESTIONS` to a small number before a full experiment.
5. Replace the Hugging Face placeholder values or set
   `PERSIST_RESULTS_ONLINE` to `False`.

Runtime data is written below the configured `local_data/` or Kaggle output
directory; it is not source code and should not be edited by hand.
