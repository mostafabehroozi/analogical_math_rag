"""Exercise the notebook's actual evaluation loop through interrupted calls."""

import hashlib
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.merging_evaluation_checkpoints import (
    MergingEvaluationCheckpoint, canonical_fingerprint, fingerprint_merging_adapter,
)
from src.merging_finetuning import (
    compare_base_and_adapted_candidate_trees, evaluate_tree_trace,
    generate_candidate_pool, run_direct_solution, run_single_candidate_revision,
)
from src.utils import save_json_atomic


class NotebookPartialResumeTests(TestCase):
    def setUp(self):
        self.generated, self.judged, self.retrievals = [], [], []
        self.retrieval_counts = []
        self.interrupt_generation = None
        self.interrupt_judgment = None
        notebook = json.loads(Path("merging_finetuning.ipynb").read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if "def evaluate_population" in "".join(c["source"]))
        self.loop = cell.split("benchmark_reports = {}", 1)[0]

    def generate(self, prompt, **kwargs):
        if self.interrupt_generation == len(self.generated):
            self.interrupt_generation = None
            raise RuntimeError("session interrupted during inference")
        identity = json.dumps([prompt, kwargs], sort_keys=True)
        self.generated.append(identity)
        return {"status": "SUCCESS", "text": hashlib.sha256(identity.encode()).hexdigest(),
                "input_tokens": 10, "output_tokens": 5, "elapsed_seconds": 1.0}

    def judge(self, text, ground_truth, *args):
        if self.interrupt_judgment == len(self.judged):
            self.interrupt_judgment = None
            raise RuntimeError("session interrupted during judging")
        self.judged.append((ground_truth, text))
        return {"status": "SUCCESS", "is_correct": False, "error_details": None}

    def retrieve(self, question, *args, top_k):
        self.retrievals.append(question)
        self.retrieval_counts.append(top_k)
        return [{"question": f"Example {i}", "solution": f"Proof {i}"} for i in range(top_k)]

    def evaluate(self, root, records=None, limit=None, mode="full", verbose=False):
        records = records or [{"question": "Question?", "ground_truth": "42",
                               "benchmark_index": 3, "source_benchmark": "math500"}]
        namespace = {
            "EVAL_MODE": mode, "EVAL_PROGRESS": False, "EVAL_VERBOSE": verbose,
            "RUN_PHASE_1": True, "RUN_PHASE_2": True, "PHASE_2_TREE_SIZES": (4, 8),
            "SEED": 42, "evaluator": object(), "generator": self.generate,
            "embedding_model": object(), "embedded_exemplars": [],
            "exemplar_data": {"questions": [], "solutions": []}, "GENERATION": {},
            "generate_candidate_pool": generate_candidate_pool,
            "compare_base_and_adapted_candidate_trees": compare_base_and_adapted_candidate_trees,
            "run_direct_solution": run_direct_solution,
            "run_single_candidate_revision": run_single_candidate_revision,
            "retrieve_exemplars_cpu": self.retrieve, "evaluate_tree_trace": evaluate_tree_trace,
            "save_json_atomic": save_json_atomic,
        }
        identity = {"benchmark": "math500", "test_protocol": 1,
                    "evaluation_mode": mode,
                    "run_phase_1": True, "run_phase_2": mode == "full"}
        checkpoint = MergingEvaluationCheckpoint(root, identity)
        exec(self.loop, namespace)
        with patch("src.merging_finetuning.evaluate_single_answer_with_llm", side_effect=self.judge):
            return namespace["evaluate_population"]("math500", records, {}, root, checkpoint, limit)

    def assert_completed_resume(self, root, expected):
        counts = (len(self.generated), len(self.judged), len(self.retrievals))
        self.assertEqual(self.evaluate(root), expected)
        self.assertEqual((len(self.generated), len(self.judged), len(self.retrievals)), counts)
        self.assertEqual(len(self.generated), len(set(self.generated)), "No successful inference should repeat")
        self.assertEqual(len(self.judged), len(set(self.judged)), "No successful judgment should repeat")
        self.assertEqual(len(self.retrievals), 1)

    def test_interrupted_candidate_generation_reuses_partial_pool(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            self.interrupt_generation = 5
            with self.assertRaisesRegex(RuntimeError, "during inference"):
                self.evaluate(root)
            self.assertEqual(len(self.generated), 5)
            p1, p2 = self.evaluate(root)
            self.assertEqual((len(p1), len(p2)), (14, 8))
            self.assert_completed_resume(root, (p1, p2))

    def test_interrupted_judgment_keeps_generated_trees_and_prior_judgments(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            self.interrupt_judgment = 1
            with self.assertRaisesRegex(RuntimeError, "during judging"):
                self.evaluate(root)
            self.assertEqual(len(self.judged), 1)
            result = self.evaluate(root)
            self.assert_completed_resume(root, result)

    def test_larger_question_limit_keeps_completed_rows(self):
        records = [{"question": f"Question {i}?", "ground_truth": str(i),
                    "benchmark_index": i, "source_benchmark": "math500"} for i in range(2)]
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            self.evaluate(root, records, limit=1)
            original_generations, original_judgments = self.generated[:], self.judged[:]
            p1, p2 = self.evaluate(root, records, limit=2)
            self.assertEqual((len(p1), len(p2)), (28, 16))
            self.assertEqual(self.retrievals, ["Question 0?", "Question 1?"])
            self.assertEqual(len(self.generated), len(set(self.generated)))
            self.assertEqual(len(self.judged), len(set(self.judged)))
            self.assertEqual(self.generated[:len(original_generations)], original_generations)
            self.assertEqual(self.judged[:len(original_judgments)], original_judgments)


class NotebookMinimalEvaluationTests(NotebookPartialResumeTests):
    def evaluate(self, root, records=None, limit=None, mode="minimal", verbose=False):
        return super().evaluate(root, records, limit, mode, verbose)

    def test_minimal_compares_only_shared_one_shot_pair_and_judges_roots(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            p1, p2 = self.evaluate(root)
            self.assertEqual((len(p1), p2), (2, []))
            self.assertEqual(self.retrieval_counts, [2])
            self.assertEqual((len(self.generated), len(self.judged)), (4, 2))
            calls = [json.loads(identity) for identity in self.generated]
            self.assertEqual([kwargs["use_adapter"] for _, kwargs in calls], [False, False, False, True])
            self.assertIn("Question: Example 0", calls[0][0])
            self.assertNotIn("Question: Example 1", calls[0][0])
            self.assertIn("Question: Example 1", calls[1][0])
            self.assertNotIn("Question: Example 0", calls[1][0])
            self.assertEqual(calls[2][0], calls[3][0])
            self.assertEqual(calls[2][1]["seed"], calls[3][1]["seed"])
            self.assertEqual([run["arm"] for run in p1], ["base", "adapted"])
            for run in p1:
                self.assertEqual((run["candidate_source"], run["candidate_count"], run["mode"]),
                                 ("one_shot", 2, "pair_fusion"))
                candidates = run["tree"]["trace"][:2]
                self.assertEqual([node["kind"] for node in candidates], ["one_shot_candidate"] * 2)
                for node in candidates:
                    self.assertIn(node["text"], calls[2][0])
                self.assertEqual(set(run["evaluation"]["node_correctness"]), {run["tree"]["root_node_id"]})
            self.assertEqual(p1[0]["tree"]["trace"][:2], p1[1]["tree"]["trace"][:2])
            self.assert_completed_resume(root, (p1, p2))

    def test_interrupted_candidate_generation_reuses_partial_pool(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            self.interrupt_generation = 1
            with self.assertRaisesRegex(RuntimeError, "during inference"):
                self.evaluate(root)
            self.assertEqual(len(self.generated), 1)
            result = self.evaluate(root)
            self.assertEqual((len(self.generated), len(self.judged)), (4, 2))
            self.assert_completed_resume(root, result)

    def test_interrupted_judgment_keeps_generated_trees_and_prior_judgments(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            self.interrupt_judgment = 1
            with self.assertRaisesRegex(RuntimeError, "during judging"):
                self.evaluate(root)
            self.assertEqual(len(self.judged), 1)
            result = self.evaluate(root)
            self.assertEqual((len(self.generated), len(self.judged)), (4, 2))
            self.assert_completed_resume(root, result)

    def test_larger_question_limit_keeps_completed_rows(self):
        records = [{"question": f"Question {i}?", "ground_truth": str(i),
                    "benchmark_index": i, "source_benchmark": "math500"} for i in range(2)]
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            self.evaluate(root, records, limit=1)
            original_generations, original_judgments = self.generated[:], self.judged[:]
            p1, p2 = self.evaluate(root, records, limit=2)
            self.assertEqual((len(p1), p2), (4, []))
            self.assertEqual((len(self.generated), len(self.judged)), (8, 4))
            self.assertEqual(self.retrieval_counts, [2, 2])
            self.assertEqual(self.generated[:len(original_generations)], original_generations)
            self.assertEqual(self.judged[:len(original_judgments)], original_judgments)

    def test_failed_fusion_retries_only_failed_generation_and_its_judgment(self):
        original_generate = self.generate
        attempts = []

        def fail_first_fusion(prompt, **kwargs):
            attempts.append(prompt)
            if len(attempts) == 3:
                return {"status": "GENERATION_FAILURE", "text": None}
            return original_generate(prompt, **kwargs)

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), patch.object(
            self, "generate", side_effect=fail_first_fusion,
        ):
            root = Path(directory)
            p1, _ = self.evaluate(root)
            self.assertEqual(p1[0]["tree"]["status"], "INCOMPLETE")
            self.assertIsNone(p1[0]["evaluation"]["root_correct"])
            self.assertEqual((len(self.generated), len(self.judged)), (3, 1))
            result = self.evaluate(root)
            self.assertEqual((len(attempts), len(self.generated), len(self.judged)), (5, 4, 2))
            self.assert_completed_resume(root, result)

    def test_failed_root_judgment_retries_without_regeneration(self):
        original_judge = self.judge
        attempts = []

        def fail_first_judgment(text, ground_truth, *args):
            attempts.append(text)
            if len(attempts) == 1:
                return {"status": "API_FAILURE", "is_correct": None}
            return original_judge(text, ground_truth, *args)

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), patch.object(
            self, "judge", side_effect=fail_first_judgment,
        ):
            root = Path(directory)
            p1, _ = self.evaluate(root)
            self.assertIsNone(p1[0]["evaluation"]["root_correct"])
            self.assertEqual((len(self.generated), len(self.judged)), (4, 1))
            result = self.evaluate(root)
            self.assertEqual((len(attempts), len(self.generated), len(self.judged)), (3, 4, 2))
            self.assert_completed_resume(root, result)

    def test_verbose_progress_reports_stages_resources_cache_reuse_and_accuracy(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first_output, resumed_output, completed_output = StringIO(), StringIO(), StringIO()
            self.interrupt_judgment = 1
            with redirect_stdout(first_output), self.assertRaisesRegex(RuntimeError, "during judging"):
                self.evaluate(root, verbose=True)
            initial = first_output.getvalue()
            for detail in ("Retrieval: top_k=2", "Exemplar 1:", "Exemplar 2:",
                           "one_shot candidates (one exemplar per answer)/base", "tokens in/out=10/5",
                           "fusion of the shared answer pair/base", "fusion of the shared answer pair/adapted",
                           "judging final answer only", "status=SUCCESS; cached=False"):
                self.assertIn(detail, initial)
            with redirect_stdout(resumed_output):
                result = self.evaluate(root, verbose=True)
            resumed = resumed_output.getvalue()
            self.assertIn("Retrieval: top_k=2; cached=True", resumed)
            self.assertIn("SUCCESS; cached=True; wall=", resumed)
            self.assertIn("judge fusion-1-0 -> incorrect; status=SUCCESS; cached=True", resumed)
            self.assertIn("running final accuracy: base=0/1 (0.0%); adapted=0/1 (0.0%)", resumed)
            with redirect_stdout(completed_output):
                self.assertEqual(self.evaluate(root, verbose=True), result)
            self.assertIn("restored question 1/1", completed_output.getvalue())
            self.assertEqual((len(self.generated), len(self.judged)), (4, 2))


class JudgmentCheckpointTests(TestCase):
    def test_failed_judgment_is_retried_but_successful_false_is_cached(self):
        tree = {"root_node_id": "answer", "trace": [
            {"node_id": "answer", "status": "SUCCESS", "text": "wrong answer"},
        ]}
        cache, saved = {}, []
        with patch("src.merging_finetuning.evaluate_single_answer_with_llm", side_effect=[
            {"status": "API_FAILURE", "is_correct": None},
            {"status": "SUCCESS", "is_correct": False},
        ]) as judge:
            first = evaluate_tree_trace(tree, "42", object(), {}, cache, lambda: saved.append(dict(cache)))
            self.assertIsNone(first["root_correct"])
            self.assertEqual(cache, {})
            second = evaluate_tree_trace(tree, "42", object(), {}, cache, lambda: saved.append(dict(cache)))
            third = evaluate_tree_trace(tree, "42", object(), {}, cache, lambda: saved.append(dict(cache)))
            self.assertIs(second["root_correct"], False)
            self.assertEqual(second, third)
            self.assertEqual(judge.call_count, 2)
            self.assertEqual(len(saved), 1)


class NotebookCheckpointIdentityTests(TestCase):
    def test_actual_identity_builder_tracks_judge_settings_and_excludes_credentials(self):
        notebook = json.loads(Path("merging_finetuning.ipynb").read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][14]["source"])
        source = "def file_sha256" + source.split("def file_sha256", 1)[1]
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter weights")
            (adapter / "tokenizer_config.json").write_text("{}")
            embeddings = Path(directory) / "embeddings.npy"
            embeddings.write_bytes(b"embedding bytes")
            namespace = {
                "hashlib": hashlib, "json": json, "Path": Path,
                "canonical_fingerprint": canonical_fingerprint,
                "fingerprint_merging_adapter": fingerprint_merging_adapter,
                "model": SimpleNamespace(config=SimpleNamespace(_commit_hash="base-revision")),
                "BASE_MODEL_NAME": "base-model", "adapter_path": adapter,
                "exemplar_data": {"questions": ["Example?"], "solutions": ["42"]},
                "embeddings_path": embeddings,
                "prepared": {"manifest": {
                    "sources": [{"sha256": "source-digest"}], "split_question_ids": {},
                }},
                "CONFIG": {"AVALAI_MODEL_NAME_EVALUATOR": "judge", "AVALAI_API_KEY": "secret-test-value"},
                "SEED": 42, "MAX_LENGTH": 4096, "GENERATION": {},
                "RUN_PHASE_1": True, "RUN_PHASE_2": True, "PHASE_2_TREE_SIZES": (4, 8),
                "EVAL_MODE": "full",
            }
            exec(source, namespace)
            original = canonical_fingerprint(namespace["evaluation_identity"])
            self.assertNotIn("secret-test-value", json.dumps(namespace["evaluation_identity"]))
            self.assertIn("api_manager.py", namespace["evaluation_identity"]["code_sha256"])
            self.assertEqual(namespace["evaluation_identity"]["notebook_evaluation_sha256"],
                             canonical_fingerprint(notebook["cells"][18]["source"]))
            for setting in ("AVALAI_REASONING_EFFORT_EVALUATOR", "AVALAI_REASONING_EFFORT",
                            "AVALAI_ENABLE_THINKING", "AVALAI_CHAT_TEMPLATE_KWARGS_ENABLE_THINKING"):
                with self.subTest(setting=setting):
                    namespace["CONFIG"][setting] = "changed"
                    exec(source, namespace)
                    self.assertNotEqual(canonical_fingerprint(namespace["evaluation_identity"]), original)
                    namespace["CONFIG"].pop(setting)
            namespace["EVAL_MODE"] = "minimal"
            exec(source, namespace)
            self.assertIs(namespace["evaluation_identity"]["run_phase_1"], True)
            self.assertIs(namespace["evaluation_identity"]["run_phase_2"], False)
            self.assertEqual(namespace["evaluation_identity"]["phase_2_tree_sizes"], [])
            self.assertNotEqual(canonical_fingerprint(namespace["evaluation_identity"]), original,
                                "Minimal and full protocols must use different checkpoint identities")
