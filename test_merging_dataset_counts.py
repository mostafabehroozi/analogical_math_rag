"""Offline checks for JSON provenance and always-visible dataset row counts."""

import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest import TestCase

from src.merging_finetuning import load_merging_datasets, prepare_merging_data


def merging_row(question, label="answer"):
    return {
        "input_prompt": (
            f"<Main Question to Solve>{question}</Main Question to Solve>"
            f"<Example 1>Question: {question}\n"
            "Rationale and Answer: first solution</Example 1>"
            f"<Example 2>Question: {question}\n"
            "Rationale and Answer: second solution</Example 2>"
        ),
        "label": label,
    }


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = f"user:{messages[0]['content']}\nassistant:"
        return prompt if add_generation_prompt else prompt + messages[1]["content"]

    def __call__(self, text, add_special_tokens=False):
        # Keep ordinary prompts short and make one answer exceed the limit.
        return {"input_ids": [1] * (50 if text == "overlength answer" else 5)}


class MergingDatasetCountTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_rows(self, name, rows):
        path = self.directory / name
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_per_source_counts_and_original_one_based_rows(self):
        first = self.write_rows("first.json", [
            merging_row("A?"), merging_row("B?"), merging_row("A?"), {},
        ])
        second = self.write_rows("second.json", [
            merging_row("C?"), merging_row("B?"), None, merging_row("D?"),
        ])
        loaded = load_merging_datasets([first, second])
        for source in loaded["sources"]:
            self.assertEqual(
                {key: source[key] for key in (
                    "raw_rows", "extracted_rows", "invalid_rows", "exact_duplicate_rows",
                )},
                {"raw_rows": 4, "extracted_rows": 2, "invalid_rows": 1, "exact_duplicate_rows": 1},
            )
        self.assertEqual([row["source_row"] for row in loaded["records"]], [1, 2, 1, 4])
        self.assertEqual([row["source_index"] for row in loaded["records"]], [0, 1, 0, 3])
        self.assertEqual([row["source_row"] for row in loaded["failures"]], [3, 4, 2, 3])
        self.assertEqual([row["reason"] for row in loaded["failures"]].count("exact_duplicate"), 2)

    def test_counts_print_and_persist_without_tokenizer_on_every_prepare(self):
        source = self.write_rows("source.json", [
            merging_row("A?"), merging_row("B?"), merging_row("C?"),
            merging_row("A?"), {},
        ])
        output_dir = self.directory / "prepared"
        # Repeated preparation also covers the path used before adapter reload.
        for _ in range(2):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                prepared = prepare_merging_data([source], output_dir, tokenizer=None)
            self.assertIn("extracted 3 rows from 5 JSON rows", output.getvalue())
            self.assertIn("invalid: 1; exact duplicates: 1", output.getvalue())
            self.assertIn("source.json: extracted 3 of 5 rows", output.getvalue())
            self.assertIn("train=1, validation=1, test=1", output.getvalue())
            persisted = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
            expected = {
                "json_rows": 5, "extracted_rows": 3, "invalid_rows": 1,
                "exact_duplicate_rows": 1,
                "split_rows": {"train": 1, "validation": 1, "test": 1},
            }
            self.assertEqual(prepared["dataset_statistics"], expected)
            self.assertEqual(persisted["dataset_statistics"], expected)
            self.assertIsNone(prepared["tokenized"])

    def test_tokenized_count_reflects_overlength_training_filter(self):
        rows = [merging_row(f"Question {index}?") for index in range(5)]
        source = self.write_rows("source.json", rows)
        output_dir = self.directory / "prepared"
        with contextlib.redirect_stdout(io.StringIO()):
            initial = prepare_merging_data([source], output_dir)
        excluded_source_index = initial["splits"]["train"][0]["source_index"]
        rows[excluded_source_index]["label"] = "overlength answer"
        source.write_text(json.dumps(rows), encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            prepared = prepare_merging_data([source], output_dir, tokenizer=FakeTokenizer(), max_length=20)
        statistics = prepared["dataset_statistics"]
        self.assertEqual(statistics["split_rows"], {"train": 3, "validation": 1, "test": 1})
        self.assertEqual(statistics["tokenized_split_rows"], {"train": 2, "validation": 1, "test": 1})
        self.assertIn("after overlength filtering: train=2, validation=1, test=1", output.getvalue())
        persisted = json.loads((output_dir / "data_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["dataset_statistics"], statistics)
        self.assertEqual(persisted["token_statistics"]["splits"]["train"]["kept_records"], 2)
