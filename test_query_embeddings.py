"""Offline checks of the saved benchmark question vectors (src/query_embeddings.py)."""

import os
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from src import query_embeddings as qe

BERT = "Bert-MLM_arXiv-MP-class_zbMath"
LEAP = "MathLeap-Qwen-8B"
QUESTIONS = {"math500": ["What is $1+1$?", "Find $x$."], "gsm8k": ["Tom has 3 apples."]}


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.asarray([[len(t), 1.0, 0.0] for t in texts], np.float32)


def write(folder, model=BERT):
    from src.corpus_filter import EMBEDDING_MODEL_PRESETS

    preset = EMBEDDING_MODEL_PRESETS[model]
    vectors = {name: np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3) + 1
               for name, texts in QUESTIONS.items()}
    info = {"model_id": preset["hf_name"], "model_revision": preset["revision"],
            "max_seq_length": preset["max_seq_length"]}
    return qe.write_model_folder(folder, info, QUESTIONS, vectors), vectors


class SavedVectorsTest(unittest.TestCase):
    def test_round_trip_and_lookup(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, vectors = write(folder)
            self.assertEqual(manifest["dim"], 3)
            self.assertEqual(manifest["benchmarks"]["math500"]["rows"], 2)
            loaded = qe.load_saved_vectors(folder, {"EXEMPLAR_EMBEDDING_MODEL": BERT})
            self.assertEqual(loaded["counts"], {"gsm8k": 1, "math500": 2})
            model = FakeModel()
            encoder = qe.SavedQueryEncoder(loaded["vectors"], lambda: model, name=BERT)
            out = encoder.encode(["Find $x$.", "Tom has 3 apples."])
            np.testing.assert_array_equal(out, np.stack([vectors["math500"][1], vectors["gsm8k"][0]]))
            self.assertEqual(model.calls, [], "the model is not even loaded when every text is saved")
            np.testing.assert_array_equal(encoder.encode("What is $1+1$?"), vectors["math500"][0])
            # A text without a saved vector goes to the model, which loads once.
            mixed = encoder.encode(["new question", "Find $x$."], normalize_embeddings=True)
            self.assertEqual(model.calls, [["new question"]])
            np.testing.assert_allclose(np.linalg.norm(mixed, axis=1), 1.0, rtol=1e-6)
            self.assertEqual((encoder.hits, encoder.misses), (4, 1))

    def test_another_model_or_a_damaged_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            write(folder, BERT)
            with self.assertRaises(ValueError):
                qe.load_saved_vectors(folder, {"EXEMPLAR_CORPUS": "NuminaMath-1.5", "EXEMPLAR_EMBEDDING_MODEL": LEAP})
            np.save(os.path.join(folder, "gsm8k.npy"), np.zeros((1, 3), np.float32))
            with self.assertRaises(ValueError):
                qe.load_saved_vectors(folder, {"EXEMPLAR_EMBEDDING_MODEL": BERT})

    def test_folder_download_and_the_model_loader(self):
        from src.corpus_filter import EMBEDDING_MODEL_PRESETS
        from src.utils import load_embedding_model

        with tempfile.TemporaryDirectory() as work:
            source = {"repo_id": "r", "repo_type": "dataset", "revision": "abc",
                      "folder": f"benchmark_embeddings/{BERT}"}

            def download(repo_id, repo_type, revision, allow_patterns, local_dir, token):
                self.assertEqual(allow_patterns, [f"benchmark_embeddings/{BERT}/*"])
                write(os.path.join(local_dir, "benchmark_embeddings", BERT))

            config = {"EXEMPLAR_EMBEDDING_MODEL": BERT, "EMBEDDINGS_DIR": work}
            self.assertEqual(EMBEDDING_MODEL_PRESETS[BERT]["benchmark_embeddings_source"]["folder"],
                             f"benchmark_embeddings/{BERT}", "the published folder is pinned")
            with patch.dict(EMBEDDING_MODEL_PRESETS[BERT], benchmark_embeddings_source=source):
                folder = qe.ensure_benchmark_embeddings(config, download_fn=download)
            self.assertEqual(folder, os.path.join(work, "benchmark_embeddings", BERT))
            self.assertTrue(os.path.isfile(os.path.join(folder, "manifest.json")))

            loads = []

            class Recorder:
                def __init__(self, path, **kwargs):
                    loads.append(path)

                def encode(self, texts, **kwargs):
                    return np.ones((len(texts), 3), np.float32)

            module = ModuleType("sentence_transformers")
            module.SentenceTransformer = Recorder
            with patch.dict(sys.modules, {"sentence_transformers": module}):
                encoder = load_embedding_model({**config, "BENCHMARK_EMBEDDINGS_ENABLED": True,
                                                "EMBEDDING_MODEL_PATH": EMBEDDING_MODEL_PRESETS[BERT]["hf_name"]})
                self.assertIsInstance(encoder, qe.SavedQueryEncoder)
                self.assertEqual(loads, [], "no model is loaded up front")
                encoder.encode(["not saved"])
                self.assertEqual(loads, [EMBEDDING_MODEL_PRESETS[BERT]["hf_name"]])
                # Without saved vectors (none local, none published) the loader falls back to the model.
                with patch.dict(EMBEDDING_MODEL_PRESETS[BERT], benchmark_embeddings_source=None):
                    plain = load_embedding_model({"EXEMPLAR_EMBEDDING_MODEL": BERT, "BENCHMARK_EMBEDDINGS_ENABLED": True,
                                                  "EMBEDDINGS_DIR": os.path.join(work, "none"),
                                                  "EMBEDDING_MODEL_PATH": EMBEDDING_MODEL_PRESETS[BERT]["hf_name"]})
                self.assertIsInstance(plain, Recorder)

    def test_retrieval_uses_the_saved_vectors(self):
        import test_corpus_filter  # noqa: F401  (imports src.pipeline_steps without sentence_transformers)
        pipeline_steps = test_corpus_filter.pipeline_steps
        corpus = np.eye(3, dtype=np.float32)
        encoder = qe.SavedQueryEncoder({qe.text_key("target"): np.asarray([0, 0, 1], np.float32)},
                                       lambda: FakeModel())
        result = pipeline_steps.retrieve("target", encoder, ["a", "b", "c"], corpus, 1, question_to_index_map={})
        self.assertEqual(result["retrieved_indices"], [2])
        self.assertEqual(encoder.misses, 0)

    def test_benchmark_question_texts_match_the_target_loader(self):
        from src.benchmark_data import load_target_benchmarks

        def fake_dataset(path, *args, split=None):
            return {"problem": ["  P1 ", "P2"], "answer": ["1", "2"], "question": ["Q1"]}

        corpus = ["c0", "c1", "c2"]
        texts = qe.benchmark_question_texts(["math500", "numina_hard"], hard_indices=[2, 0],
                                            corpus_questions=corpus, load_dataset_fn=fake_dataset)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hard.json")
            open(path, "w").write("[2, 0]")
            import json
            questions, _, names = load_target_benchmarks(
                {"TARGET_BENCHMARKS": ["math500", "numina_hard"], "HARD_QUESTIONS_INDICES_PATH": path},
                {"questions": corpus, "solutions": ["s0", "s1", "s2"]},
                load_json_fn=lambda p: json.load(open(p)), load_dataset_fn=fake_dataset)
        self.assertEqual(texts["math500"] + texts["numina_hard"], questions)


if __name__ == "__main__":
    unittest.main()
