# Project Cleanup Report

Date: 2026-08-09  
Repository: `analogical_math_rag`  
Branch audited: `analogical-consistency`

## Scope and safety approach

The audit covered every Python module, both notebooks, configuration, root-level
research artifacts, imports, top-level symbol references, dictionary keys, and
the actual Git diff. Cleanup was limited to changes that could be supported by
static reference evidence or a focused behavioral test. Feature modules were
not treated as dead merely because their current configuration flag is `False`.

The Python source was reduced from 17,473 to 17,342 lines (131 net lines
removed) before adding documentation. No notebook, dataset, paper context, or
generated extract was deleted.

## Behavior defects corrected

### Parallel benchmarking model selection

`src/pipeline_steps.py::solve_with_parallel_benchmarking` used `model_name`
without assigning it. The function now selects the final-solver model for
Gemini, AvalAI, or Ollama and raises a clear `TypeError` for an unsupported
manager. A redundant, unused `examples_block` construction and repeated local
prompt import were removed from the same function.

### Layer 2 zero-score fallback

`src/layer2_analysis.py::BlockC.run_for_mask_and_perspective` crashed on its
fallback path because the log message referenced undefined `k_fallback`. The
message now reports the actual selected subset size. The stale docstring was
updated to match the existing behavior: keep all retrieved one-shot candidates
and exclude zero-shot candidates. The fallback flag is normalized from
`numpy.bool_` to a native `bool` for reliable JSON/report serialization.

### Optional Gemini SDK import

`src/api_manager.py` claimed to tolerate missing provider SDKs, but a legacy
`google` namespace could raise `ImportError` rather than
`ModuleNotFoundError`. The guard now handles that environment correctly and
leaves `GeminiAPIManager` able to produce its explicit installation error.

## Unreachable and duplicated definitions removed

`config.py` contained duplicate keys for:

- `EMBEDDINGS_DIR`
- `APPLY_DATASET_CONSTRUCTION`
- `CORE_SIMP_PHASE1_N_ATTEMPTS`

The duplicate core-simplification heading was also removed. All retained values
are identical to the values Python used before the cleanup.

`src/prompts.py` contained earlier definitions of `final_solver_v3`,
`final_solver_v4`, and
`self_sampling_augmentor_simplification_simple_shallow_with_solution` that were
silently overwritten later in the same dictionary. Only the later, previously
effective definitions were retained, so runtime prompt selection is unchanged.

## Dead code and import cleanup

- Consolidated repeated imports in `src/pipeline_steps.py`,
  `src/context_logger.py`, `src/layer2_analysis.py`, `src/hf_sync.py`, and
  `src/api_manager.py`.
- Removed unused imports from API management, evaluation, Layer 1, Layer 2,
  integration, multibranch transformation, orchestration, and utilities.
- Removing the unused `fastapi.logger` import means orchestration no longer
  requires FastAPI merely to import the module.
- Removed unused evaluator model/temperature calculations from
  `src/dataset_builder.py`; evaluation already delegates provider selection to
  `evaluate_single_answer_with_llm`.
- Removed unused evaluation truncation and primary-attempt variables.
- Kept retrieval timing calls but removed the unused variable that repeatedly
  stored and overwrote their return values.
- Removed 42 unnecessary `f` prefixes from constant log/print strings. Text and
  behavior are unchanged.
- Added missing final newlines where the edited files lacked them.

## Project clarity and hygiene added

- `.gitignore` now excludes Python caches, notebook/editor state, local runtime
  data, lint/test caches, virtual environments, and local `.env` secrets.
- `README.md` documents setup, the real notebook entry point, the module map,
  and a safe first-run checklist.
- `requirements.txt` declares the libraries imported by the source and
  notebooks, including all three configured provider clients.
- Generated `__pycache__` directories created during verification were removed.

## Validation completed

- All 21 Python files parse successfully with Python 3.9.
- Pyflakes passes with zero findings.
- Focused Pylint checks pass for unused/repeated imports, unreachable code,
  undefined names, and use-before-assignment.
- Custom AST duplicate-dictionary-key detection passes.
- Both notebooks remain valid JSON.
- `git diff --check` passes.
- A fake-manager smoke test verifies model selection and all requested attempts
  in the parallel benchmarking function without making network calls.
- A synthetic zero-matrix smoke test executes the Layer 2 fallback and verifies
  that it selects only the retrieved one-shot candidates and returns a native
  boolean fallback flag.

## Reviewed and deliberately retained

- `wait_for_final_sync` has no current in-repository caller, but it is a useful
  public shutdown hook for the non-daemon Hugging Face sync thread. It was not
  deleted.
- Feature-specific modules and prompt templates were retained even when their
  default feature flags are off; those flags represent experiment selection,
  not proof that the implementation is obsolete.
- `MULTIBRANCH_TIEBREAK_FAVOR_ORIGINAL` and
  `MULTIBRANCH_TIEBREAK_EPSILON` appear to describe intended behavior more
  fully than the current selector implements. They were not removed or wired
  into selection because that would alter experimental results without a
  specification.
- `main_experiment.ipynb` repeats imports across cells, which allows cells to be
  run independently. The repetitions were not removed.
- The workspace-level `utilities.ipynb`, `extract_proj.txt`,
  `extract_proj_excluded.txt`, `paper_context.txt`, and hard-question JSON were
  inspected but left untouched. The extract files are generated snapshots and
  are now stale relative to the cleaned source, but they may be inputs to the
  paper-writing workflow and live outside the nested Git repository.

## Remaining validation limitation

A full end-to-end experiment was not run because this execution environment
lacks the declared ML/provider dependencies, local datasets/models, and API
credentials. The new requirements file makes the missing environment explicit;
the static gates and isolated behavioral tests do not call external services.
