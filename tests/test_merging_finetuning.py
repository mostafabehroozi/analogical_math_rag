import json
import tempfile
import unittest
from pathlib import Path

from src.merging_finetuning import (
    build_evaluation_populations,
    compare_base_and_adapted_trees,
    grouped_split,
    load_merging_datasets,
    normalize_question,
    parse_legacy_merging_prompt,
    prepare_merging_data,
    run_merging_tree,
    retrieve_exemplars_cpu,
    tokenize_splits,
    upload_adapter_to_hub,
)
from src.prompts import EXEMPLAR_FORMAT, create_final_reasoning_prompt, create_merging_prompt


def legacy_prompt(question="What is 1 + 1?", first="Rationale: a\nFinal Answer: 2", second="Rationale: b\nFinal Answer: 3"):
    examples = [
        EXEMPLAR_FORMAT.format(question=question, solution=first),
        EXEMPLAR_FORMAT.format(question=question, solution=second),
    ]
    return create_final_reasoning_prompt(
        question, examples, {"PROMPT_TEMPLATE_FINAL_SOLVER": "final_solver_v3"}
    )


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        ids = [11, 12, 13]
        if add_generation_prompt:
            return ids + [20]
        return ids + [20, 31, 32, 2]


class FakeEmbeddingModel:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class RecordingGenerator:
    def __init__(self, fail_call=None):
        self.calls = []
        self.fail_call = fail_call

    def __call__(self, prompt, **kwargs):
        index = len(self.calls)
        self.calls.append((prompt, kwargs))
        if index == self.fail_call:
            return {"status": "GENERATION_FAILED", "text": "", "input_tokens": 3}
        return {
            "status": "SUCCESS", "text": f"solution-{index}",
            "input_tokens": 3, "output_tokens": 2, "elapsed_seconds": 0.01,
        }


class RecordingHubApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def whoami(self):
        return {"name": "test-user"}

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)


class MergingDatasetTests(unittest.TestCase):
    def test_upload_adapter_creates_private_model_repo_and_uploads_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "best_adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
            api = RecordingHubApi()
            result = upload_adapter_to_hub(
                adapter_dir, "secret-token", private=True, api=api
            )
        self.assertEqual(result["repo_id"], "test-user/merging-qwen3-4b-qlora")
        self.assertEqual(api.created[0]["repo_type"], "model")
        self.assertTrue(api.created[0]["private"])
        self.assertEqual(api.uploaded[0]["repo_id"], result["repo_id"])
        self.assertNotIn("secret-token", repr(api.created) + repr(api.uploaded))

    def test_upload_adapter_requires_saved_peft_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "best_adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "adapter weights"):
                upload_adapter_to_hub(adapter_dir, "secret-token", api=RecordingHubApi())

    def test_parses_legacy_contract_and_builds_cautious_prompt(self):
        parsed = parse_legacy_merging_prompt(legacy_prompt())
        self.assertEqual(parsed["question"], "What is 1 + 1?")
        self.assertEqual(len(parsed["candidate_solutions"]), 2)
        prompt = create_merging_prompt(parsed["question"], parsed["candidate_solutions"])
        self.assertIn("Either candidate may contain incorrect", prompt)
        self.assertNotIn("True Solved Example", prompt)

    def test_rejects_candidate_for_a_different_question(self):
        prompt = legacy_prompt().replace("Question: What is 1 + 1?", "Question: Different?", 1)
        with self.assertRaisesRegex(ValueError, "both examples"):
            parse_legacy_merging_prompt(prompt)

    def test_loader_reports_malformed_and_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            good = {"input_prompt": legacy_prompt(), "label": "Rationale: ok\nFinal Answer: 2"}
            path.write_text(json.dumps([good, good, {"input_prompt": "bad", "label": "x"}]), encoding="utf-8")
            loaded = load_merging_datasets([path])
        self.assertEqual(len(loaded["records"]), 1)
        self.assertEqual(len(loaded["failures"]), 2)
        self.assertEqual(loaded["failures"][0]["reason"], "exact_duplicate")

    def test_preparation_persists_hashes_membership_and_token_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.json"
            rows = [
                {
                    "input_prompt": legacy_prompt(question=f"Question {index}"),
                    "label": f"Rationale: ok\nFinal Answer: {index}",
                }
                for index in range(3)
            ]
            data_path.write_text(json.dumps(rows), encoding="utf-8")
            prepared = prepare_merging_data(
                [data_path], Path(directory) / "out", tokenizer=FakeTokenizer(), max_length=20
            )
            manifest_path = Path(directory) / "out" / "data_manifest.json"
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(persisted["sources"][0]["sha256"]), 64)
        self.assertEqual(set(persisted["split_membership"]), {"train", "validation", "test"})
        self.assertIn("token_statistics", prepared["manifest"])

    def test_grouped_split_never_leaks_question_identity(self):
        records = []
        for question_index in range(10):
            for variant in range(2):
                question = f"Question {question_index}"
                records.append({
                    "question": question,
                    "question_id": normalize_question(question),
                    "record_id": f"{question_index}-{variant}",
                })
        splits = grouped_split(records, seed=42)
        identities = [{item["question_id"] for item in splits[name]} for name in splits]
        self.assertFalse(identities[0] & identities[1])
        self.assertFalse(identities[0] & identities[2])
        self.assertFalse(identities[1] & identities[2])
        self.assertTrue(all(splits.values()))

    def test_tokenizer_masks_prompt_and_keeps_entire_assistant_target(self):
        record = {"record_id": "x", "prompt": "prompt", "label": "answer"}
        tokenized, _ = tokenize_splits(
            {"train": [record], "validation": [record], "test": [record]},
            FakeTokenizer(), max_length=20,
        )
        example = tokenized["train"][0]
        self.assertEqual(example["labels"][:4], [-100] * 4)
        self.assertEqual(example["labels"][4:], [31, 32, 2])

    def test_overlength_is_reported_and_empty_split_stops(self):
        record = {"record_id": "x", "question_id": "q", "prompt": "p", "label": "a"}
        with self.assertRaisesRegex(ValueError, "empty after token filtering"):
            tokenize_splits(
                {"train": [record], "validation": [record], "test": [record]},
                FakeTokenizer(), max_length=2,
            )

    def test_ground_truth_alignment_and_population_exclusion(self):
        def record(question):
            return {"question": question, "question_id": normalize_question(question)}
        prepared = {"splits": {
            "train": [record("train q")], "validation": [record("val q")],
            "test": [record("test q")],
        }}
        populations = build_evaluation_populations(
            prepared,
            ["train q", "val q", "test q", "rejected q"],
            ["a", "b", "c", "d"],
        )
        self.assertEqual(populations["heldout_accepted"][0]["ground_truth"], "c")
        self.assertEqual(
            [item["question"] for item in populations["remaining_benchmark"]],
            ["rejected q"],
        )

    def test_cpu_retrieval_excludes_normalized_exact_question(self):
        retrieved = retrieve_exemplars_cpu(
            " Same   Question ",
            ["same question", "other one", "other two"],
            ["forbidden solution", "s1", "s2"],
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
            FakeEmbeddingModel(),
            top_k=2,
        )
        self.assertEqual([item["question"] for item in retrieved], ["other one", "other two"])
        self.assertNotIn("forbidden solution", [item["solution"] for item in retrieved])


class MergingTreeTests(unittest.TestCase):
    def test_zero_shot_even_tree_routes_leaves_to_base_and_fusions_to_adapter(self):
        generator = RecordingGenerator()
        result = run_merging_tree("q", generator, "zero_shot", zero_shot_n=4)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(result["trace"]), 7)
        self.assertTrue(all(not call[1]["use_adapter"] for call in generator.calls[:4]))
        self.assertTrue(all(call[1]["use_adapter"] for call in generator.calls[4:]))

    def test_odd_solution_is_carried_without_extra_generation(self):
        generator = RecordingGenerator()
        result = run_merging_tree("q", generator, "zero_shot", zero_shot_n=3)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(len(generator.calls), 5)

    def test_retrieved_first_layer_uses_two_distinct_exemplars(self):
        examples = [
            {"question": f"rq{i}", "solution": f"rs{i}"} for i in range(4)
        ]
        generator = RecordingGenerator()
        result = run_merging_tree("main q", generator, "retrieved", retrieved_examples=examples)
        self.assertEqual(result["status"], "SUCCESS")
        first_prompt = generator.calls[0][0]
        self.assertIn("Question: rq0", first_prompt)
        self.assertIn("Question: rq1", first_prompt)
        self.assertFalse(generator.calls[0][1]["use_adapter"])

    def test_mixed_fuses_zero_branch_before_pooling(self):
        examples = [{"question": f"rq{i}", "solution": f"rs{i}"} for i in range(4)]
        generator = RecordingGenerator()
        result = run_merging_tree(
            "main", generator, "mixed", retrieved_examples=examples, zero_shot_n=4
        )
        zero_first_layer = [node for node in result["trace"] if node["node_id"].startswith("zero-fusion-1")]
        self.assertEqual(len(zero_first_layer), 2)
        self.assertEqual(result["status"], "SUCCESS")

    def test_base_control_disables_adapter_for_every_call(self):
        generator = RecordingGenerator()
        run_merging_tree(
            "q", generator, "zero_shot", zero_shot_n=2, fusion_use_adapter=False
        )
        self.assertTrue(all(not kwargs["use_adapter"] for _, kwargs in generator.calls))

    def test_comparison_reuses_leaves_but_regenerates_each_fusion_arm(self):
        generator = RecordingGenerator()
        compared = compare_base_and_adapted_trees(
            "q", generator, "zero_shot", zero_shot_n=2
        )
        self.assertEqual(compared["base"]["trace"][:2], compared["adapted"]["trace"][:2])
        self.assertEqual(len(generator.calls), 4)  # two leaves and one root per arm
        self.assertFalse(generator.calls[2][1]["use_adapter"])
        self.assertTrue(generator.calls[3][1]["use_adapter"])

    def test_failure_stops_and_resume_reuses_successful_nodes(self):
        failing = RecordingGenerator(fail_call=1)
        incomplete = run_merging_tree("q", failing, "zero_shot", zero_shot_n=2)
        self.assertEqual(incomplete["status"], "INCOMPLETE")
        resumed_generator = RecordingGenerator()
        completed = run_merging_tree(
            "q", resumed_generator, "zero_shot", zero_shot_n=2,
            resume_trace=incomplete["trace"],
        )
        self.assertEqual(completed["status"], "SUCCESS")
        self.assertEqual(len(resumed_generator.calls), 2)  # failed leaf plus root

    def test_prompts_never_receive_ground_truth(self):
        secret = "GROUND_TRUTH_SECRET_9482"
        generator = RecordingGenerator()
        run_merging_tree("question only", generator, "zero_shot", zero_shot_n=2)
        self.assertTrue(all(secret not in prompt for prompt, _ in generator.calls))


if __name__ == "__main__":
    unittest.main()
