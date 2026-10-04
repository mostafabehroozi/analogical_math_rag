"""Completion boundaries must match generation even when BPE merges there."""

from unittest import TestCase

from src.merging_finetuning import _chat_tokens, tokenize_splits


class BoundaryTokenizer:
    """Small tokenizer that merges newline + R, like a BPE boundary merge."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "<user>" + messages[0]["content"] + "</user>\n<assistant>\n"
        if len(messages) > 1:
            text += messages[1]["content"] + "<eos>\n"
        return self(text, add_special_tokens=False)["input_ids"] if tokenize else text

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        ids = []
        while text:
            if text.startswith("\nR"):
                ids.append(1000)
                text = text[2:]
            elif text.startswith("<eos>"):
                ids.append(1001)
                text = text[5:]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return {"input_ids": ids}


class ChatTokenTests(TestCase):
    def setUp(self):
        self.tokenizer = BoundaryTokenizer()

    def test_boundary_merge_reproduces_old_failure_and_preserves_generation(self):
        messages = [{"role": "user", "content": "Solve."}]
        prompt_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": "Rationale: 2"}], tokenize=True
        )
        self.assertNotEqual(full_ids[:len(prompt_ids)], prompt_ids)
        tokens = _chat_tokens(self.tokenizer, "Solve.", "Rationale: 2")
        self.assertEqual(tokens["prompt_ids"], prompt_ids)
        self.assertEqual(tokens["input_ids"][:len(prompt_ids)], prompt_ids)
        self.assertEqual(tokens["input_ids"][len(prompt_ids):],
                         self.tokenizer("Rationale: 2<eos>\n", add_special_tokens=False)["input_ids"])

    def test_prefix_stable_conversation_retains_existing_tokens(self):
        tokens = _chat_tokens(self.tokenizer, "Solve.", "Answer: 2")
        full_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Solve."}, {"role": "assistant", "content": "Answer: 2"}],
            tokenize=True,
        )
        self.assertEqual(tokens["input_ids"], full_ids)

    def test_actual_template_text_mismatch_still_fails(self):
        class InconsistentTemplate(BoundaryTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                return "changed" + text if len(messages) > 1 else text

        with self.assertRaisesRegex(ValueError, "text prefix"):
            _chat_tokens(InconsistentTemplate(), "Solve.", "Rationale: 2")

    def test_loss_masks_prompt_and_supervises_answer_and_end_token(self):
        record = {"prompt": "Solve.", "label": "Rationale: 2", "record_id": "example"}
        splits = {name: [record] for name in ("train", "validation", "test")}
        tokenized, report = tokenize_splits(splits, self.tokenizer, max_length=200)
        prompt_length = len(_chat_tokens(self.tokenizer, record["prompt"], record["label"])["prompt_ids"])
        for name, rows in tokenized.items():
            row = rows[0]
            self.assertEqual(row["labels"][:prompt_length], [-100] * prompt_length)
            self.assertEqual(row["labels"][prompt_length:], row["input_ids"][prompt_length:])
            self.assertEqual(row["labels"][prompt_length], ord("R"))
            self.assertIn(1001, row["labels"][prompt_length:])
            self.assertEqual(len(row["labels"]), len(row["attention_mask"]))
            self.assertEqual(report["splits"][name]["kept_records"], 1)

    def test_overlength_filter_uses_separately_encoded_length(self):
        short = {"prompt": "Solve.", "label": "Rationale: 2", "record_id": "short"}
        long = {**short, "label": "Rationale: " + "2" * 100, "record_id": "long"}
        length = len(_chat_tokens(self.tokenizer, short["prompt"], short["label"])["input_ids"])
        splits = {name: [short, long] for name in ("train", "validation", "test")}
        tokenized, report = tokenize_splits(splits, self.tokenizer, max_length=length)
        for name, rows in tokenized.items():
            self.assertEqual([row["record_id"] for row in rows], ["short"])
            self.assertEqual(report["splits"][name]["excluded"][0]["record_id"], "long")
