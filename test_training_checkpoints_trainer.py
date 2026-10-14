"""A real Trainer, interrupted and restored from the Hub, ends where an uninterrupted run does.

Runs only where torch, transformers and peft are installed (Kaggle, or a dev venv).
The tiny CPU model keeps it fast; the checkpoint format, rotation, best-model
tracking and data-order resume are the real Trainer's.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

from src.merging_finetuning import CompletionOnlyCollator
from src.training_checkpoints import HubTrainingCheckpoints
from test_training_checkpoints import FakeTrainingHub

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments
    AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    AVAILABLE = False

IDENTITY = {"format_version": 1, "workflow": "trainer-test", "qlora": {"epochs": 3}, "data": {}, "extra": {}}
SESSION_LOST = "session lost"


class FakeTokenizer:
    """Stands in for the real tokenizer, which Trainer saves into every checkpoint."""
    pad_token_id = 0

    def save_pretrained(self, directory):
        (Path(directory) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def rows(count, seed):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(count):
        ids = torch.randint(1, 50, (8,), generator=generator).tolist()
        result.append({"input_ids": ids, "attention_mask": [1] * 8, "labels": [-100, -100] + ids[2:]})
    return result


def tiny_model():
    torch.manual_seed(0)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=2, n_head=2))
    return get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["c_attn"], fan_in_fan_out=True,
        task_type="CAUSAL_LM",
    ))


def make_trainer(output_dir, callbacks=None):
    args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=3, per_device_train_batch_size=2,
        per_device_eval_batch_size=2, learning_rate=5e-3, warmup_steps=2, use_cpu=True,
        eval_strategy="steps", eval_steps=4, save_strategy="steps", save_steps=4,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=1, report_to="none", seed=42, data_seed=42,
        remove_unused_columns=False, disable_tqdm=True, log_level="error",
    )
    return Trainer(
        model=tiny_model(), args=args, train_dataset=rows(12, 1), eval_dataset=rows(4, 2),
        data_collator=CompletionOnlyCollator(FakeTokenizer()),
        callbacks=callbacks,
    )


def trained_weights(trainer):
    return {name: parameter.detach().clone()
            for name, parameter in trainer.model.named_parameters() if parameter.requires_grad}


@skipUnless(AVAILABLE, "torch, transformers and peft are required")
class RealTrainerResumeTests(TestCase):
    def interrupted_run(self, directory, hub, upload_every=1):
        class LoseSession(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 10:
                    raise RuntimeError(SESSION_LOST)

        store = HubTrainingCheckpoints(
            Path(directory) / "session-1", IDENTITY, repo_id="researcher/training-state",
            upload_enabled=True, restore_enabled=True, upload_every=upload_every,
            api=hub, download_fn=hub.download, log=lambda message: None,
        )
        self.assertIsNone(store.restore())
        trainer = make_trainer(Path(directory) / "session-1", [store.trainer_callback(), LoseSession()])
        with self.assertRaisesRegex(RuntimeError, SESSION_LOST):
            trainer.train()
        return store

    def test_a_lost_session_resumes_from_the_hub_and_matches_an_uninterrupted_run(self):
        with TemporaryDirectory() as directory:
            reference = make_trainer(Path(directory) / "reference")
            reference.train()
            expected = trained_weights(reference)
            expected_best = reference.state.best_model_checkpoint

            hub = FakeTrainingHub()
            first = self.interrupted_run(directory, hub)
            pointer = json.loads(hub.files[first.remote_root + "/latest.json"])
            self.assertEqual(pointer["global_step"], 8, "steps 4 and 8 were saved before the crash at 10")
            remote_folders = {name.split("/")[-2] for name in hub.names_under(first.remote_root + "/checkpoint-")}
            self.assertLessEqual(len(remote_folders), 2, "old checkpoints are pruned from the Hub")
            self.assertIn("optimizer.pt", pointer["files"]["checkpoint-8"])
            self.assertIn("rng_state.pth", pointer["files"]["checkpoint-8"])

            # A new Kaggle session: empty working directory, same Hub.
            second = HubTrainingCheckpoints(
                Path(directory) / "session-2", IDENTITY, repo_id="researcher/training-state",
                upload_enabled=True, restore_enabled=True, api=hub, download_fn=hub.download,
                log=lambda message: None,
            )
            resume = second.restore()
            self.assertEqual(Path(resume).name, "checkpoint-8")
            restored_state = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))
            best_now = Path(restored_state["best_model_checkpoint"])
            self.assertEqual(best_now.parent, (Path(directory) / "session-2").resolve(),
                             "the best checkpoint must be found on this machine, not at session 1's path")
            self.assertTrue(best_now.is_dir())
            resumed = make_trainer(Path(directory) / "session-2", [second.trainer_callback()])
            resumed.train(resume_from_checkpoint=resume)

            self.assertEqual(resumed.state.global_step, reference.state.global_step)
            self.assertEqual(Path(resumed.state.best_model_checkpoint).name, Path(expected_best).name)
            self.assertAlmostEqual(resumed.state.best_metric, reference.state.best_metric, places=6)
            actual = trained_weights(resumed)
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                self.assertTrue(torch.allclose(actual[name], expected[name], atol=1e-6),
                                f"{name} differs after resuming from the Hub")
            final = json.loads(hub.files[second.remote_root + "/latest.json"])
            self.assertEqual(final["global_step"], reference.state.global_step)
            self.assertEqual(Path(resumed.state.best_model_checkpoint).parent, Path(directory).resolve() / "session-2")

    def test_a_save_interval_larger_than_one_still_resumes_exactly(self):
        with TemporaryDirectory() as directory:
            reference = make_trainer(Path(directory) / "reference")
            reference.train()
            hub = FakeTrainingHub()
            first = self.interrupted_run(directory, hub, upload_every=2)
            self.assertEqual(json.loads(hub.files[first.remote_root + "/latest.json"])["global_step"], 8)
            second = HubTrainingCheckpoints(
                Path(directory) / "session-2", IDENTITY, repo_id="researcher/training-state",
                restore_enabled=True, api=hub, download_fn=hub.download, log=lambda message: None,
            )
            resumed = make_trainer(Path(directory) / "session-2")
            resumed.train(resume_from_checkpoint=second.restore())
            for name, value in trained_weights(reference).items():
                self.assertTrue(torch.allclose(trained_weights(resumed)[name], value, atol=1e-6))

    def test_resuming_after_training_finished_still_returns_the_best_adapter(self):
        with TemporaryDirectory() as directory:
            hub = FakeTrainingHub()
            store = HubTrainingCheckpoints(
                Path(directory) / "session-1", IDENTITY, repo_id="researcher/training-state",
                upload_enabled=True, api=hub, download_fn=hub.download, log=lambda message: None,
            )
            store.restore()
            finished = make_trainer(Path(directory) / "session-1", [store.trainer_callback()])
            finished.train()
            expected = trained_weights(finished)
            self.assertEqual(json.loads(hub.files[store.remote_root + "/latest.json"])["global_step"],
                             finished.state.max_steps)

            # The session died after the last save but before the adapter upload.
            second = HubTrainingCheckpoints(
                Path(directory) / "session-2", IDENTITY, repo_id="researcher/training-state",
                restore_enabled=True, api=hub, download_fn=hub.download, log=lambda message: None,
            )
            again = make_trainer(Path(directory) / "session-2")
            again.train(resume_from_checkpoint=second.restore())
            self.assertEqual(again.state.global_step, finished.state.max_steps)
            for name, value in expected.items():
                self.assertTrue(torch.allclose(trained_weights(again)[name], value, atol=1e-6))
