"""Offline checks for external evaluation selection, routing, and notebook reports."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.benchmark_data import HUGGINGFACE_BENCHMARK_SPECS, benchmark_name_for_target_index
from src.finetuning_evaluation import external_evaluation_benchmarks, load_external_evaluation_benchmark
from src.merging_finetuning import normalize_question, summarize_evaluated_runs
from src.merging_evaluation_checkpoints import MergingEvaluationCheckpoint, canonical_fingerprint
from src.simplification_finetuning import (
    HELDOUT_POPULATION, SIMPLIFICATION_INSTRUCTION, SimplificationEvaluationCheckpoint,
    evaluate_question, load_heldout_population, retryable_failures, summarize_evaluation,
)
from src.utils import save_json_atomic


def prepared(*questions):
    return {"splits": {
        "train": [{"question": q, "question_id": normalize_question(q)} for q in questions],
        "validation": [], "test": [],
    }}


def fake_dataset(*args, **kwargs):
    spec = next(spec for spec in HUGGINGFACE_BENCHMARK_SPECS.values() if spec["path"] == args[0])
    return {spec["question_field"]: ["Question 0?", "Question 1?"], spec["answer_field"]: ["42", "42"]}


def notebook_cell(filename, marker):
    nb = json.loads(Path(filename).read_text(encoding="utf-8"))
    return next("".join(cell["source"]) for cell in nb["cells"]
                if cell["cell_type"] == "code" and marker in "".join(cell["source"]))


class ExternalBenchmarkTests(TestCase):
    def test_default_selection_covers_all_external_benchmarks(self):
        self.assertEqual(external_evaluation_benchmarks(), list(HUGGINGFACE_BENCHMARK_SPECS))
        self.assertNotIn("numina_hard", external_evaluation_benchmarks())

    def test_invalid_or_numina_selection_is_rejected(self):
        for names in (["numina_hard"], ["math500", "math500"], [], "gsm8k", [None]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                external_evaluation_benchmarks(names)

    def test_external_questions_need_not_contain_construction_test_questions(self):
        data = prepared("Question 0?")
        data["splits"]["test"] = [{"question": "Unrelated?", "question_id": "unrelated?"}]
        rows, audit, config = load_external_evaluation_benchmark(data, "math500", {}, load_dataset_fn=fake_dataset)
        self.assertEqual([r["benchmark_index"] for r in rows], [1])
        self.assertEqual(rows[0]["record_id"], "math500:1")
        self.assertEqual(rows[0]["label_kind"], "unlabeled")
        self.assertEqual(audit["accepted_questions_excluded"], 1)
        self.assertEqual(audit["eligible_external_questions"], 1)
        self.assertEqual(config["TARGET_BENCHMARK"], "math500")

    def test_each_source_overrides_stale_target_routing_without_mutating_input(self):
        original = {"TARGET_BENCHMARK": "numina_hard", "TARGET_BENCHMARKS": ["aime25"],
                    "_TARGET_BENCHMARK_FOR_QUERY": "numina_hard", "BENCHMARK_MAX_QUESTIONS": 1}
        for name in external_evaluation_benchmarks():
            rows, _, config = load_external_evaluation_benchmark(prepared(), name, original, load_dataset_fn=fake_dataset)
            self.assertEqual(len(rows), 2)
            self.assertEqual(benchmark_name_for_target_index(config, 0), name)
            self.assertTrue(all(row["record_id"].startswith(name + ":") for row in rows))
        self.assertEqual(original["_TARGET_BENCHMARK_FOR_QUERY"], "numina_hard")
        self.assertEqual(original["BENCHMARK_MAX_QUESTIONS"], 1)

    def test_gsm8k_keeps_its_final_answer_format(self):
        rows, _, _ = load_external_evaluation_benchmark(
            prepared(), "gsm8k", {},
            load_dataset_fn=lambda *args, **kwargs: {"question": ["Q?"], "answer": ["Reasoning\n#### 42"]},
        )
        self.assertEqual(rows[0]["ground_truth"], "Reasoning\n\nFinal Answer: \\boxed{42}")

    def test_every_construction_split_and_duplicate_copy_is_excluded(self):
        data = prepared("Train?")
        data["splits"]["validation"] = [{"question": "Validate?", "question_id": "validate?"}]
        data["splits"]["test"] = [{"question": "Test?", "question_id": "test?"}]
        rows, audit, _ = load_external_evaluation_benchmark(
            data, "aime25", {},
            load_dataset_fn=lambda *args, **kwargs: {
                "problem": ["Train?", "Validate?", "Test?", "New?", " NEW? ", " train? "],
                "answer": [1, 2, 3, 42, 43, 1],
            },
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["benchmark_index"], 3)
        self.assertEqual(rows[0]["benchmark_indices"], [3, 4])
        self.assertEqual(rows[0]["ground_truth"], "42")
        self.assertEqual(audit["accepted_questions_excluded"], 3)
        self.assertEqual(audit["duplicate_benchmark_rows_removed"], 2)
        self.assertEqual(audit["duplicate_rows_with_different_reference_text"], 1)

    def test_both_notebooks_default_to_all_external_benchmarks(self):
        for filename, limit in (("merging_finetuning.ipynb", "EVAL_QUESTION_LIMIT"),
                                ("simplification_finetuning.ipynb", "MAX_EVAL_QUESTIONS")):
            config = notebook_cell(filename, limit + " =")
            self.assertIn(limit + " = None", config)
            self.assertIn("EVAL_BENCHMARKS = external_evaluation_benchmarks()", config)

    def test_merging_defaults_to_minimal_evaluation_with_progress_and_details(self):
        config = notebook_cell("merging_finetuning.ipynb", "EVAL_QUESTION_LIMIT =")
        self.assertRegex(config, r"EVAL_MODE\s*=\s*['\"]minimal['\"]")
        self.assertIn("EVAL_PROGRESS = True", config)
        self.assertIn("EVAL_VERBOSE = True", config)


class NotebookExternalEvaluationTests(TestCase):
    def benchmark_loader(self, events, root):
        def load(data, name, config):
            if events:
                previous = events[-1]
                self.assertTrue((root / previous / self.summary_file).exists(),
                                "Previous benchmark metrics must be saved before the next load")
            events.append(name)
            return load_external_evaluation_benchmark(data, name, config, load_dataset_fn=fake_dataset)
        return load

    def test_merging_saves_each_benchmark_before_next_and_routes_judges(self):
        self.assert_merging_reports("full")

    def test_minimal_merging_saves_separate_base_adapter_comparisons_and_resumes(self):
        self.assert_merging_reports("minimal")

    def assert_merging_reports(self, mode):
        self.summary_file = "two_phase_evaluation_summary.json"
        events, judged = [], []
        tree = {"status": "SUCCESS", "root_node_id": "leaf", "root_solution": "42",
                "trace": [{"node_id": "leaf", "kind": "zero_shot_candidate", "status": "SUCCESS", "text": "42"}]}

        def judge(result, truth, evaluator, config, **kwargs):
            name = config["TARGET_BENCHMARK"]
            judged.append(name)
            correct = name in ("math500", "aime25")
            return {"root_correct": correct, "node_correctness": {"leaf": correct},
                    "judge_status": {"leaf": "SUCCESS"}, "transitions": {}}

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory) / "evaluations"
            namespace = {
                "EVAL_MODE": mode, "EVAL_PROGRESS": False, "EVAL_VERBOSE": False,
                "RUN_PHASE_1": True, "RUN_PHASE_2": True, "PHASE_2_TREE_SIZES": (4, 8),
                "EVAL_BENCHMARKS": external_evaluation_benchmarks(), "EVAL_QUESTION_LIMIT": None,
                "EVALUATION_DIR": root, "WORK_DIR": root.parent, "prepared": prepared(), "CONFIG": {}, "SEED": 42,
                "evaluator": object(), "generator": object(), "embedding_model": object(),
                "exemplar_data": {"questions": [], "solutions": []}, "embedded_exemplars": [],
                "GENERATION": {}, "np": np, "json": json, "save_json_atomic": save_json_atomic,
                "MergingEvaluationCheckpoint": MergingEvaluationCheckpoint,
                "canonical_fingerprint": canonical_fingerprint,
                "evaluation_identity": {"test_protocol": 1}, "HF_TOKEN": None,
                "HF_EVAL_DATASET_REPO_ID": None, "HF_EVAL_REMOTE_PREFIX": "merging_evaluations",
                "HF_EVAL_UPLOAD_ENABLED": False, "HF_EVAL_RESTORE_ENABLED": False,
                "HF_EVAL_DATASET_PRIVATE": True, "HF_EVAL_UPLOAD_EVERY": 10,
                "load_external_evaluation_benchmark": self.benchmark_loader(events, root),
                "retrieve_exemplars_cpu": lambda *args, **kwargs: [
                    {"question": f"Example {i}?", "solution": "42"} for i in range(kwargs["top_k"])
                ],
                "generate_candidate_pool": lambda *args, **kwargs: {
                    "status": "SUCCESS", "candidates": [tree["trace"][0]] * kwargs["count"],
                },
                "run_direct_solution": lambda *args, **kwargs: tree,
                "run_single_candidate_revision": lambda *args, **kwargs: tree,
                "compare_base_and_adapted_candidate_trees": lambda *args, **kwargs: {"base": tree, "adapted": tree},
                "evaluate_tree_trace": judge, "summarize_evaluated_runs": summarize_evaluated_runs,
            }
            exec(notebook_cell("merging_finetuning.ipynb", "def evaluate_population"), namespace)
            # Recover reporting with no inference objects, run lists, helpers, or
            # report index in memory. Saved settings take precedence on restart.
            for name in events:
                (root / name / self.summary_file).unlink()
            recovery = {
                "WORK_DIR": root.parent, "EVAL_BENCHMARKS": external_evaluation_benchmarks(),
                "EVAL_MODE": "changed-after-evaluation", "SEED": -1,
            }
            exec(notebook_cell("merging_finetuning.ipynb", "report_index ="), recovery)
            self.assertEqual(events, external_evaluation_benchmarks())
            self.assertEqual(set(judged), set(events))
            for name in events:
                report = json.loads((root / name / self.summary_file).read_text())
                text = (root / name / "evaluation_report.txt").read_text()
                self.assertIn("Final-answer comparison", text)
                self.assertIn("Bootstrap 95% CI", text)
                self.assertEqual(report["benchmark"], name)
                self.assertEqual(report["evaluated_questions"], 2)
                self.assertEqual(report["phase_1_runs"], 4 if mode == "minimal" else 28)
                self.assertEqual(report["phase_2_runs"], 0 if mode == "minimal" else 16)
                group = "phase_1/one_shot/2/pair_fusion" if mode == "minimal" else "phase_1/none/0/direct_solution"
                metric = report["summaries"][group + "/base"]
                self.assertEqual(metric["accuracy_on_evaluated"], float(name in ("math500", "aime25")))
                if mode == "minimal":
                    comparison = report["minimal_comparison"]
                    for arm in ("base", "adapted"):
                        self.assertEqual(comparison[arm]["accuracy_on_evaluated"],
                                         float(name in ("math500", "aime25")))
                    self.assertEqual(comparison["paired"]["paired_questions"], 2)
                    self.assertEqual(comparison["paired"]["accuracy_delta"], 0.0)
                rows = json.loads((root / name / "phase_1_results.json").read_text())
                self.assertTrue(all(row["benchmark"] == name for row in rows))
            events.clear()
            judged.clear()
            exec(notebook_cell("merging_finetuning.ipynb", "def evaluate_population"), namespace)
            self.assertEqual(events, external_evaluation_benchmarks())
            self.assertEqual(judged, [], "Completed merging questions must skip inference and judgment")

    def test_merging_report_recovery_without_saved_results_has_actionable_error(self):
        with TemporaryDirectory() as directory:
            namespace = {"WORK_DIR": Path(directory), "EVAL_BENCHMARKS": ["math500"]}
            with self.assertRaisesRegex(FileNotFoundError, "No complete saved benchmark results"):
                exec(notebook_cell("merging_finetuning.ipynb", "report_index ="), namespace)
            self.assertFalse((Path(directory) / "evaluations" / "benchmark_reports.json").exists())

    def test_simplification_separate_reports_and_resume_use_external_unlabeled_rows(self):
        self.summary_file = "summary.json"
        events, judged = [], []

        def generator(prompt, **kwargs):
            text = prompt.split("Original question:\n", 1)[1] if prompt.startswith(SIMPLIFICATION_INSTRUCTION) else "42"
            return {"status": "SUCCESS", "text": text}

        def judge(text, truth, evaluator, config):
            name = config["TARGET_BENCHMARK"]
            judged.append(name)
            return {"status": "SUCCESS", "is_correct": name == "math500"}

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), patch(
            "src.evaluation.evaluate_single_answer_with_llm", side_effect=judge
        ):
            root = Path(directory) / "evaluations"
            namespace = {
                "RUN_EVALUATION": True, "EVAL_POPULATIONS": external_evaluation_benchmarks(),
                "MAX_EVAL_QUESTIONS": None, "WORK_DIR": Path(directory), "prepared": prepared(),
                "CONFIG": {}, "evaluator": SimpleNamespace(), "generator": generator, "SEED": 42,
                "SIMPLIFIER_MAX_NEW_TOKENS": 20, "SOLVER_MAX_NEW_TOKENS": 20,
                "EVAL_PROGRESS": False, "EVAL_VERBOSE": False,
                "HELDOUT_POPULATION": HELDOUT_POPULATION, "load_heldout_population": load_heldout_population,
                "load_external_evaluation_benchmark": self.benchmark_loader(events, root),
                "SIMPLIFICATION_INSTRUCTION": SIMPLIFICATION_INSTRUCTION,
                "evaluate_question": evaluate_question, "summarize_evaluation": summarize_evaluation,
                "retryable_failures": retryable_failures, "save_json_atomic": save_json_atomic,
                "SimplificationEvaluationCheckpoint": SimplificationEvaluationCheckpoint,
                "canonical_fingerprint": canonical_fingerprint, "evaluation_identity": {"test_protocol": 1},
                "HF_TOKEN": None, "HF_EVAL_DATASET_REPO_ID": None,
                "HF_EVAL_REMOTE_PREFIX": "simplification_evaluations",
                "HF_EVAL_UPLOAD_ENABLED": False, "HF_EVAL_RESTORE_ENABLED": False,
                "HF_EVAL_DATASET_PRIVATE": True, "HF_EVAL_UPLOAD_EVERY": 10,
            }
            cell = notebook_cell("simplification_finetuning.ipynb", "retryable_failures(case)")
            exec(cell, namespace)
            self.assertEqual(events, external_evaluation_benchmarks())
            self.assertEqual(set(judged), set(events))
            # The final cell works after restart without evaluation objects or
            # in-memory summaries, and upgrades saved metrics from raw outputs.
            recovery = {"WORK_DIR": Path(directory), "EVAL_BENCHMARKS": external_evaluation_benchmarks()}
            captured = StringIO()
            with redirect_stdout(captured):
                exec(notebook_cell("simplification_finetuning.ipynb", "report_index ="), recovery)
            self.assertIn("Benchmark comparison", captured.getvalue())
            for name in events:
                report = json.loads((root / name / "summary.json").read_text())
                text = (root / name / "evaluation_report.txt").read_text()
                self.assertIn("Solver final-answer comparison", text)
                self.assertIn("Failures and reuse", text)
                self.assertEqual(report["benchmark"], name)
                self.assertEqual(report["questions"], 2)
                self.assertEqual(report["behavior"]["unlabeled"]["base"]["exact_copy_rate"], 1)
                self.assertEqual(report["behavior"]["copy"]["questions"], 0)
                self.assertEqual(report["solver"]["accuracy"]["direct"], float(name == "math500"))
            judged.clear()
            events.clear()
            exec(cell, namespace)
            self.assertEqual(judged, [], "All four benchmarks should reuse their own completed results")
