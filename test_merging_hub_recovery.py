"""Offline checks for recovering a completed merging adapter from the Hub."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from huggingface_hub.errors import RepositoryNotFoundError
from requests import Response

from src.merging_finetuning import download_adapter_from_hub


class FakeHub:
    def __init__(self, files=None, missing=False):
        self.files = files
        self.missing = missing

    def whoami(self):
        return {"name": "researcher"}

    def model_info(self, repo_id):
        if self.missing:
            response = Response()
            response.status_code = 404
            response.url = "https://huggingface.co/owner/adapter"
            raise RepositoryNotFoundError("repository missing", response=response)
        return SimpleNamespace(
            sha="revision-123",
            siblings=[SimpleNamespace(rfilename=name) for name in self.files],
        )


class HubRecoveryTests(TestCase):
    def test_missing_default_repo_allows_initial_training(self):
        with TemporaryDirectory() as directory:
            result = download_adapter_from_hub(
                Path(directory) / "adapter", "secret", api=FakeHub(missing=True),
                download_fn=lambda **kwargs: self.fail("Unexpected download"),
            )
            self.assertIsNone(result)

    def test_explicit_missing_repo_stops_instead_of_retraining(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not found or is inaccessible"):
                download_adapter_from_hub(
                    Path(directory) / "adapter", "secret", "owner/adapter",
                    api=FakeHub(missing=True),
                )

    def test_complete_adapter_downloads_at_checked_revision(self):
        files = {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"}
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            for name in files:
                (Path(kwargs["local_dir"]) / name).write_text("test")

        with TemporaryDirectory() as directory:
            result = download_adapter_from_hub(
                Path(directory) / "adapter", "secret", api=FakeHub(files),
                download_fn=download,
            )
            self.assertEqual(result["repo_id"], "researcher/merging-qwen3-4b-qlora")
            self.assertEqual(result["revision"], "revision-123")
            self.assertEqual(calls[0]["revision"], "revision-123")
            self.assertEqual(calls[0]["token"], "secret")

    def test_incomplete_remote_adapter_never_downloads(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "complete PEFT adapter"):
                download_adapter_from_hub(
                    Path(directory) / "adapter", "secret",
                    api=FakeHub({"adapter_config.json", "tokenizer_config.json"}),
                    download_fn=lambda **kwargs: self.fail("Unexpected download"),
                )
