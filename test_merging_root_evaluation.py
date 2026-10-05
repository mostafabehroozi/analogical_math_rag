"""Offline checks for the two-candidate, final-answer-only merging comparison."""

import copy
from unittest import TestCase
from unittest.mock import patch

from src.merging_finetuning import (
    compare_base_and_adapted_candidate_trees,
    evaluate_tree_trace,
    generate_candidate_pool,
)


class RootEvaluationTests(TestCase):
    def test_two_retrieved_candidates_are_shared_for_four_generations_and_two_judges(self):
        question = "MAIN_QUESTION: Find the target value."
        exemplars = [
            {"question": "EXEMPLAR_ONE", "solution": "EXAMPLE_SOLUTION_ONE"},
            {"question": "EXEMPLAR_TWO", "solution": "EXAMPLE_SOLUTION_TWO"},
        ]
        generation_calls = []

        def generate(prompt, **options):
            generation_calls.append((prompt, options))
            if len(generation_calls) <= 2:
                text = f"TARGET_CANDIDATE_{len(generation_calls)}"
            else:
                text = "ADAPTED_FINAL" if options["use_adapter"] else "BASE_FINAL"
            return {"status": "SUCCESS", "text": text}

        pool = generate_candidate_pool(
            question, generate, "one_shot", count=2, retrieved_examples=exemplars,
        )
        arms = compare_base_and_adapted_candidate_trees(
            question, pool["candidates"], generate,
        )
        self.assertEqual(len(generation_calls), 4)
        for index, (prompt, options) in enumerate(generation_calls[:2]):
            self.assertFalse(options["use_adapter"])
            self.assertIn(question, prompt)
            self.assertIn(exemplars[index]["question"], prompt)
            self.assertNotIn(exemplars[1 - index]["question"], prompt)
        base_prompt, base_options = generation_calls[2]
        adapted_prompt, adapted_options = generation_calls[3]
        self.assertEqual(base_prompt, adapted_prompt)
        self.assertEqual(base_options["seed"], adapted_options["seed"])
        self.assertFalse(base_options["use_adapter"])
        self.assertTrue(adapted_options["use_adapter"])
        self.assertIn(question, base_prompt)
        self.assertIn("TARGET_CANDIDATE_1", base_prompt)
        self.assertIn("TARGET_CANDIDATE_2", base_prompt)
        self.assertNotIn("EXEMPLAR_ONE", base_prompt)
        self.assertNotIn("EXEMPLAR_TWO", base_prompt)
        self.assertEqual(arms["base"]["trace"][:2], arms["adapted"]["trace"][:2])
        self.assertEqual(
            arms["base"]["trace"][-1]["parents"], arms["adapted"]["trace"][-1]["parents"],
        )
        preserved = copy.deepcopy(arms)
        progress = []
        with patch("src.merging_finetuning.evaluate_single_answer_with_llm") as judge:
            judge.side_effect = [
                {"status": "SUCCESS", "is_correct": False},
                {"status": "SUCCESS", "is_correct": True},
            ]
            results = {
                arm: evaluate_tree_trace(
                    tree, "reference", object(), {}, root_only=True,
                    progress_callback=lambda node, result, cached: progress.append(
                        (node["node_id"], result["is_correct"], cached)
                    ),
                )
                for arm, tree in arms.items()
            }
            self.assertEqual(judge.call_count, 2)
            self.assertEqual([call.args[0] for call in judge.call_args_list], ["BASE_FINAL", "ADAPTED_FINAL"])
        self.assertFalse(results["base"]["root_correct"])
        self.assertTrue(results["adapted"]["root_correct"])
        for arm, result in results.items():
            self.assertEqual(result["evaluation_scope"], "root_only")
            self.assertEqual(list(result["node_correctness"]), [arms[arm]["root_node_id"]])
            self.assertEqual(result["evaluation_coverage"], 1.0)
            self.assertEqual(result["transitions"], {})
        self.assertEqual(progress, [("fusion-1-0", False, False), ("fusion-1-0", True, False)])
        self.assertEqual(arms, preserved)

    def test_root_failure_is_unknown_and_retried_then_success_is_cached(self):
        tree = self.tree()
        cache = {}
        checkpoints = []
        progress = []
        with patch("src.merging_finetuning.evaluate_single_answer_with_llm") as judge:
            judge.side_effect = [
                {"status": "FAILED", "error": "temporary judge failure"},
                {"status": "SUCCESS", "is_correct": True},
            ]

            def evaluate():
                return evaluate_tree_trace(
                    tree, "reference", object(), {}, cache,
                    checkpoint_callback=lambda: checkpoints.append(copy.deepcopy(cache)),
                    root_only=True,
                    progress_callback=lambda node, result, cached: progress.append(
                        (node["node_id"], result["status"], cached)
                    ),
                )

            failed = evaluate()
            self.assertIsNone(failed["root_correct"])
            self.assertEqual(failed["judge_status"], {"root": "FAILED"})
            self.assertEqual(failed["evaluation_coverage"], 0.0)
            self.assertEqual(cache, {})
            self.assertEqual(checkpoints, [])
            recovered = evaluate()
            cached = evaluate()
            self.assertEqual(judge.call_count, 2)
        self.assertTrue(recovered["root_correct"])
        self.assertEqual(recovered, cached)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(progress, [
            ("root", "FAILED", False), ("root", "SUCCESS", False), ("root", "SUCCESS", True),
        ])
        self.assertEqual(len(cache), 1)

    def test_incomplete_tree_without_root_does_not_judge_successful_candidates(self):
        for root_id in (None, "missing-root"):
            with self.subTest(root_id=root_id):
                tree = self.tree()
                tree["status"] = "INCOMPLETE"
                tree["root_node_id"] = root_id
                tree["trace"][-1]["status"] = "FAILED"
                before = copy.deepcopy(tree)
                progress = []
                with patch("src.merging_finetuning.evaluate_single_answer_with_llm") as judge:
                    result = evaluate_tree_trace(
                        tree, "reference", object(), {}, root_only=True,
                        progress_callback=lambda *args: progress.append(args),
                    )
                    judge.assert_not_called()
                self.assertIsNone(result["root_correct"])
                self.assertEqual(result["node_correctness"], {})
                self.assertEqual(result["judge_status"], {})
                self.assertEqual(result["evaluation_coverage"], 0.0)
                self.assertEqual(result["transitions"], {})
                self.assertEqual(progress, [])
                self.assertEqual(tree, before)

    def test_default_retains_all_node_judgments_and_transition_diagnosis(self):
        progress = []
        with patch("src.merging_finetuning.evaluate_single_answer_with_llm") as judge:
            judge.side_effect = [
                {"status": "SUCCESS", "is_correct": False},
                {"status": "SUCCESS", "is_correct": True},
                {"status": "SUCCESS", "is_correct": True},
            ]
            result = evaluate_tree_trace(
                self.tree(), "reference", object(), {},
                progress_callback=lambda node, result, cached: progress.append((node["node_id"], cached)),
            )
            self.assertEqual(judge.call_count, 3)
        self.assertEqual(result["evaluation_scope"], "all_nodes")
        self.assertEqual(result["node_correctness"], {"left": False, "right": True, "root": True})
        self.assertEqual(result["transitions"], {"corrections": 1, "regressions": 0, "unchanged": 0, "unknown": 0})
        self.assertEqual(progress, [("left", False), ("right", False), ("root", False)])

    @staticmethod
    def tree():
        return {
            "status": "SUCCESS", "root_node_id": "root", "root_solution": "final answer",
            "trace": [
                {"node_id": "left", "kind": "one_shot_candidate", "status": "SUCCESS", "text": "left answer"},
                {"node_id": "right", "kind": "one_shot_candidate", "status": "SUCCESS", "text": "right answer"},
                {"node_id": "root", "kind": "fusion", "status": "SUCCESS", "text": "final answer", "parents": ["left", "right"]},
            ],
        }
