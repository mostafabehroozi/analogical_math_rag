"""Offline checks of the exemplar-corpus filter (src/corpus_filter.py) and every place that reads the corpus."""

import json
import os
import pickle
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np


def _import_without_sentence_transformers(name):
    stubbed = "sentence_transformers" not in sys.modules
    if stubbed:
        module = ModuleType("sentence_transformers")
        module.SentenceTransformer = type("SentenceTransformer", (), {})
        sys.modules["sentence_transformers"] = module
    try:
        return __import__(name, fromlist=["_"])
    finally:
        if stubbed:
            del sys.modules["sentence_transformers"]


pipeline_steps = _import_without_sentence_transformers("src.pipeline_steps")
orchestration = _import_without_sentence_transformers("src.orchestration")
layer1 = _import_without_sentence_transformers("src.layer1_base_execution")
transformation = _import_without_sentence_transformers("src.transformation_dataset_builder")

from config import CONFIG  # noqa: E402
from src import corpus_filter as cf  # noqa: E402
from src.benchmark_data import load_target_benchmarks  # noqa: E402
from src.distributed_execution import sanitize_scientific_config  # noqa: E402
from src.merging_finetuning import retrieve_exemplars_cpu  # noqa: E402
from src.utils import create_numina_hard_question_set  # noqa: E402

ON = {"CORPUS_FILTER_ENABLED": True}


class FakeDataset(dict):
    """Column access like datasets.Dataset (dataset["problem"]) plus column_names and len()."""

    @property
    def column_names(self):
        return list(self)

    def __len__(self):
        return len(next(iter(self.values())))

    def __getitem__(self, key):
        if isinstance(key, int):  # one row, like datasets.Dataset
            return {name: column[key] for name, column in self.items()}
        return super().__getitem__(key)


# A NuminaMath-1.5-like corpus.
ROWS = [
    # problem, source, synthetic, problem_is_valid
    ("If $f(x) = (1-x^2)(x^2+ax+b)$ is symmetric about $x=-2$, the maximum of $f(x)$ is ______.", "cn_k12", False, "Yes"),  # 0
    ("If $f(x)=(1-x^{2})(x^{2}+ax+b)$ is symmetric about $x=-2$, the maximum of $f(x)$ is ____.", "cn_k12", False, "Yes"),  # 1 copy of 0
    ("Find $a+b$ if $a=3$ and $b=4$.", "olympiads", False, "Yes"),        # 2
    ("Find $a-b$ if $a=3$ and $b=4$.", "olympiads", False, "Yes"),        # 3 different operator: not a copy
    ("Find $a+b$ if $a=7$ and $b=9$.", "olympiads", False, "Incomplete"),  # 4 numbers changed: copy of 2 (ignore_numbers)
    ("What is 20! / 10^m for the greatest integer m?", "orca_math", True, "Yes"),  # 5 synthetic
    ("Compute $2^{10}$.", "synthetic_math", True, "Yes"),               # 6 synthetic
    ("Compute $2^{10}$.", "aops_forum", False, "Incomplete"),          # 7 real copy of a synthetic row: kept
    ("   ", "cn_k12", False, "Yes"),                                      # 8 empty
    ("Prove that $n^{n}+(n+1)^{n+1}$ is composite for infinitely many $n$.", "olympiads_ref", False, "Yes"),  # 9
    ("Prove that $n^n + (n + 1)^{n + 1}$ is composite for infinitely many $n$.", "aops_forum", False, "Yes"),  # 10 copy of 9
]


ROWS_TEXT = [r[0] for r in ROWS]


def dataset(rows=ROWS, with_synthetic=True):
    columns = {"problem": [r[0] for r in rows], "solution": [f"s{i}" for i in range(len(rows))],
               "source": [r[1] for r in rows]}
    if with_synthetic:
        columns["synthetic"] = [r[2] for r in rows]
        columns["problem_is_valid"] = [r[3] for r in rows]
    return FakeDataset(columns)


def build(match="ignore_numbers", corpus="NuminaMath-1.5", rows=ROWS, **settings):
    config = {"EXEMPLAR_CORPUS": corpus, "CORPUS_FILTER_DUPLICATE_MATCH": match, **ON, **settings}
    return cf.build_corpus_filter(dataset(rows, with_synthetic=corpus == "NuminaMath-1.5"), config, verbose=False)


class KeyTest(unittest.TestCase):
    def test_reformatted_copies_share_a_key(self):
        same = [
            (ROWS[0][0], ROWS[1][0]),
            ("What is the greatest integer m for which the number 20! / 10^m is an integer?",
             "what is the greatest integer m for which the number 20 ! / 10 ^ m is an integer ?"),
            (r"Define \[P(x) =(x-1^2)\cdots(x-100^2).\] How many $n$? $\textbf{(A) } 4900 \qquad \textbf{(B) } 4950$",
             r"Define $$P(x) =(x-1^2)\cdots(x-100^2).$$" "\nHow many $n$?\n\n$\\textbf{(A) } 4900 \\qquad\\textbf{(B) } 4950$"),
            (ROWS[9][0], ROWS[10][0]),
            ("Fill in: x = \\_\\_\\_\\_.", "Fill in: x = ________."),
        ]
        for a, b in same:
            self.assertEqual(cf.normalized_text_key(a), cf.normalized_text_key(b), (a, b))

    def test_operators_and_numbers_are_kept_apart(self):
        self.assertNotEqual(cf.normalized_text_key("Find $a+b$."), cf.normalized_text_key("Find $a-b$."))
        self.assertNotEqual(cf.normalized_text_key(ROWS[2][0]), cf.normalized_text_key(ROWS[4][0]))
        self.assertEqual(cf.numbers_ignored_key(ROWS[2][0]), cf.numbers_ignored_key(ROWS[4][0]))
        self.assertEqual(cf.numbers_ignored_key("x = 3.5"), cf.numbers_ignored_key("x = 12"))


class FilterTest(unittest.TestCase):
    def test_numinamath_15_rows(self):
        f = build()
        self.assertEqual(f.reason(5), "synthetic")
        self.assertEqual(f.reason(6), "synthetic")
        self.assertIsNone(f.reason(7), "a real row is kept even if a synthetic row has the same text")
        self.assertEqual(f.reason(8), "empty")
        # one row per copy group: the valid one, then the lowest index
        self.assertEqual([f.reason(i) for i in (0, 1)], [None, "duplicate"])
        self.assertEqual([f.reason(i) for i in (2, 4)], [None, "duplicate"])
        self.assertEqual([f.reason(i) for i in (9, 10)], [None, "duplicate"])
        self.assertIsNone(f.reason(3))
        self.assertEqual(f.counts(), {"synthetic": 2, "duplicate": 3, "empty": 1, "kept": 5, "total": 11})
        self.assertEqual(f.summary()["hidden_by_source"]["cn_k12"], 2)

    def test_normalized_text_keeps_number_changed_rows(self):
        f = build("normalized_text")
        self.assertIsNone(f.reason(4))
        self.assertEqual(f.reason(1), "duplicate")

    def test_valid_row_wins_the_group(self):
        rows = [("Solve $x+1=2$.", "cn_k12", False, "Incomplete"), ("Solve $x + 1 = 2$.", "cn_k12", False, "Yes")]
        f = build(rows=rows)
        self.assertEqual([f.reason(0), f.reason(1)], ["duplicate", None])

    def test_numinamath_cot_uses_the_synthetic_sources(self):
        rows = ROWS + [("Rewritten AMC problem.", "synthetic_amc", False, "Yes"), ("GSM train problem 5.", "gsm8k", False, "Yes")]
        f = build(corpus="NuminaMath-CoT", rows=rows)
        self.assertEqual([f.reason(5), f.reason(6), f.reason(11)], ["synthetic"] * 3)
        self.assertIsNone(f.reason(12))

    def test_switches(self):
        self.assertFalse(build(CORPUS_FILTER_ENABLED=False).excluded.any())
        only_synthetic = build(CORPUS_FILTER_EXCLUDE_DUPLICATES=False)
        self.assertEqual(only_synthetic.counts()["duplicate"], 0)
        self.assertEqual(only_synthetic.counts()["synthetic"], 2)
        only_duplicates = build(CORPUS_FILTER_EXCLUDE_SYNTHETIC=False)
        # rows 6 (synthetic, earlier) and 7 (real, Incomplete) are copies: the real row wins
        self.assertEqual([only_duplicates.reason(6), only_duplicates.reason(7)], ["duplicate", None])
        self.assertIsNone(only_duplicates.reason(5), "synthetic rows stay when only duplicates are hidden")
        with self.assertRaises(ValueError):
            build(CORPUS_FILTER_DUPLICATE_MATCH="fuzzy")
        with self.assertRaises(ValueError):
            build(CORPUS_FILTER_ENABLED="yes")

    def test_a_partial_config_keeps_the_old_behaviour_and_config_py_turns_it_on(self):
        self.assertFalse(cf.filter_settings({})["enabled"])
        self.assertEqual(cf.corpus_signature({}), cf.LEGACY_CORPUS_SIGNATURE)
        self.assertTrue(cf.filter_settings(CONFIG)["enabled"])
        self.assertEqual(CONFIG["CORPUS_FILTER_DUPLICATE_MATCH"], "ignore_numbers")

    def test_copies_of_a_target_and_apply(self):
        f = build()
        self.assertEqual(f.copies_of("Find $a + b$ if $a = 100$ and $b = 1$.").tolist(), [2, 4])
        self.assertEqual(f.copies_of("A question that is not in the corpus.").tolist(), [])
        similarities = np.zeros(len(ROWS))
        blocked = f.apply(similarities, ROWS[0][0])
        self.assertEqual(blocked, {"hidden_rows": 6, "target_copies": 2})
        self.assertEqual(np.flatnonzero(np.isfinite(similarities)).tolist(), [2, 3, 7, 9])
        with self.assertRaises(ValueError):
            f.apply(np.zeros(3))

    def test_index_carries_the_filter_through_pickle(self):
        f = build()
        index = cf.build_question_index(dataset()["problem"], f)
        clone = pickle.loads(pickle.dumps(index))
        self.assertEqual(dict(clone), dict(index))
        self.assertIsInstance(clone.corpus_filter, cf.CorpusFilter)


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.data = cf.build_exemplar_data(dataset(), np.eye(len(ROWS), dtype=np.float32), {
            "EXEMPLAR_CORPUS": "NuminaMath-1.5", **ON})

    def retrieve(self, query, query_vector, top_k=len(ROWS), index=None):
        with patch.object(pipeline_steps, "_generate_embeddings", return_value=np.asarray([query_vector], np.float32)):
            return pipeline_steps.retrieve(query, None, self.data["questions"], self.data["embeddings"], top_k,
                                           question_to_index_map=self.data["question_to_index"] if index is None else index)

    def test_hidden_rows_and_target_copies_are_never_retrieved(self):
        vector = np.linspace(1.0, 0.5, len(ROWS))  # prefers the low indices, hidden rows included
        result = self.retrieve("Find $a+b$ if $a=1$ and $b=2$.", vector)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["retrieved_indices"], [0, 3, 7, 9])  # 2 is a copy of the target

    def test_without_the_filter_only_the_exact_question_is_excluded(self):
        plain = {q: rows for q, rows in self.data["question_to_index"].items()}
        result = self.retrieve(ROWS[2][0], np.linspace(1.0, 0.5, len(ROWS)), index=plain)
        self.assertEqual(result["retrieved_indices"], [i for i in range(len(ROWS)) if i != 2])

    def test_merging_cpu_retrieval(self):
        class Model:
            def encode(self, texts):
                return np.asarray([np.linspace(1.0, 0.5, len(ROWS))], np.float32)

        found = retrieve_exemplars_cpu("Compute $2^{12}$.", self.data["questions"], self.data["solutions"],
                                       self.data["embeddings"], Model(), top_k=2, corpus_filter=self.data["corpus_filter"])
        self.assertEqual([item["index"] for item in found], [0, 2])
        found = retrieve_exemplars_cpu("Compute $2^{12}$.", self.data["questions"], self.data["solutions"],
                                       self.data["embeddings"], Model(), top_k=2)
        self.assertEqual([item["index"] for item in found], [0, 1])


class SelectionTest(unittest.TestCase):
    def test_hard_question_selection_never_picks_hidden_rows(self):
        f = build()
        with tempfile.TemporaryDirectory() as out:
            report = create_numina_hard_question_set(dataset(), 5, output_dir=out, corpus_filter=f, random_seed=1)
            self.assertEqual(sorted(report["selected_indices"]), f.allowed_indices().tolist())
            self.assertEqual(report["corpus_filter"]["kept"], 5)
            with self.assertRaisesRegex(ValueError, "usable after the corpus filter"):
                create_numina_hard_question_set(dataset(), 6, output_dir=out, corpus_filter=f)
            with self.assertRaisesRegex(ValueError, "after the corpus filter"):
                create_numina_hard_question_set(dataset(), 1, source_percentages={"orca_math": 100},
                                                output_dir=out, corpus_filter=f)
            with self.assertRaisesRegex(ValueError, "does not occur in this dataset"):
                create_numina_hard_question_set(dataset(), 1, source_percentages={"gsm8k": 50},
                                                output_dir=out, corpus_filter=f)
            unfiltered = create_numina_hard_question_set(dataset(), 11, output_dir=out)
            self.assertEqual(sorted(unfiltered["selected_indices"]), list(range(11)))

    def test_allowed_rows_and_precomputed_embeddings(self):
        rng = np.random.default_rng(3)
        by_text = {text: rng.normal(size=8).astype(np.float32) for text in ROWS_TEXT}
        vectors = np.stack([by_text[text] for text in ROWS_TEXT])  # the "embedding file"

        class Model:  # the encoder that made that file
            def encode(self, questions, **kwargs):
                return np.stack([by_text[q] for q in questions])

        allowed = np.ones(len(ROWS), dtype=bool)
        allowed[3] = False
        f = build()
        with tempfile.TemporaryDirectory() as out:
            common = dict(selection_strategy="farthest", output_dir=out, corpus_filter=f, allowed_rows=allowed,
                          random_seed=7, shuffle_output=False)
            saved = create_numina_hard_question_set(dataset(), 3, embeddings=vectors, **common)
            encoded = create_numina_hard_question_set(dataset(), 3, embedding_model=Model(), **common)
            self.assertEqual(saved["selected_indices"], encoded["selected_indices"])
            self.assertTrue(set(saved["selected_indices"]) <= {0, 2, 7, 9})
            self.assertEqual((saved["usable_rows"], saved["embeddings"]), (4, "precomputed"))
            with self.assertRaisesRegex(ValueError, "one row per dataset row"):
                create_numina_hard_question_set(dataset(), 3, embeddings=vectors[:3], **common)
            with self.assertRaisesRegex(ValueError, "allowed_rows must have"):
                create_numina_hard_question_set(dataset(), 3, embeddings=vectors,
                                                **{**common, "allowed_rows": allowed[:3]})

    def test_numina_hard_targets_must_be_usable_rows(self):
        data = cf.build_exemplar_data(dataset(), np.eye(len(ROWS)), {"EXEMPLAR_CORPUS": "NuminaMath-1.5", **ON})
        config = {"TARGET_BENCHMARK": "numina_hard", "HARD_QUESTIONS_INDICES_PATH": "indices.json"}
        with self.assertRaisesRegex(ValueError, r"2 of the 3 numina_hard targets.*'synthetic': 1.*'duplicate': 1"):
            load_target_benchmarks(config, data, load_json_fn=lambda path: [0, 5, 1])
        questions, _, _ = load_target_benchmarks(config, data, load_json_fn=lambda path: [0, 3])
        self.assertEqual(questions, [ROWS[0][0], ROWS[3][0]])
        off = cf.build_exemplar_data(dataset(), np.eye(len(ROWS)), {"EXEMPLAR_CORPUS": "NuminaMath-1.5"})
        self.assertEqual(len(load_target_benchmarks(config, off, load_json_fn=lambda path: [0, 5, 1])[0]), 3)

    def test_transformation_corpus_targets_skip_hidden_rows(self):
        data = cf.build_exemplar_data(dataset(), np.eye(len(ROWS)), {"EXEMPLAR_CORPUS": "NuminaMath-1.5", **ON})
        records = transformation._target_records(None, None, data, {"TRANSFORMATION_DS_TARGET_SOURCE": "exemplar_corpus"})
        self.assertEqual([r["index"] for r in records], [0, 2, 3, 7, 9])


class CorpusConfigTest(unittest.TestCase):
    def test_apply_exemplar_corpus(self):
        config = {"EXEMPLAR_CORPUS": "NuminaMath-1.5", "DATA_DIR": "d", "EMBEDDINGS_DIR": "e",
                  "EXEMPLAR_CORPUS_NAME": "AI-MO/NuminaMath-CoT", "EXEMPLAR_CORPUS_NAME_HF": "AI-MO/NuminaMath-CoT"}
        cf.apply_exemplar_corpus(config)
        self.assertEqual(config["EXEMPLAR_CORPUS_NAME"], "AI-MO/NuminaMath-1.5")
        self.assertEqual(config["EXEMPLAR_CORPUS_NAME_HF"], "AI-MO/NuminaMath-1.5")
        self.assertEqual(config["EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH"],
                         os.path.join("e", "embedding_NuminaMath-1.5_with_Bert-MLM_arXiv-MP-class_zbMath.npy"))
        self.assertEqual(config["LOCAL_EXEMPLAR_CORPUS_PATH"], os.path.join("d", "NuminaMath-1.5"))
        custom = {"EXEMPLAR_CORPUS": "NuminaMath-1.5", "EXEMPLAR_CORPUS_NAME": "/kaggle/input/my-corpus"}
        cf.apply_exemplar_corpus(custom)
        self.assertEqual(custom["EXEMPLAR_CORPUS_NAME"], "/kaggle/input/my-corpus")
        with self.assertRaises(ValueError):
            cf.apply_exemplar_corpus({"EXEMPLAR_CORPUS": "NuminaMath-2"})
        self.assertEqual(CONFIG["EXEMPLAR_CORPUS"], "NuminaMath-CoT")
        self.assertTrue(CONFIG["EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH"].endswith(
            "embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy"))

    def test_embedding_download_is_checked(self):
        payload = b"embedding bytes"
        preset = cf.EXEMPLAR_CORPUS_PRESETS["NuminaMath-1.5"]
        source = dict(preset["embedding_source"], bytes=len(payload),
                      sha256=__import__("hashlib").sha256(payload).hexdigest())
        calls = []

        def download(repo_id, repo_type, revision, filename, local_dir, token):
            calls.append((repo_id, revision, filename))
            path = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(payload)
            return path

        with tempfile.TemporaryDirectory() as folder, patch.dict(preset, embedding_source=source):
            target = os.path.join(folder, "emb.npy")
            config = {"EXEMPLAR_CORPUS": "NuminaMath-1.5", "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH": target}
            self.assertEqual(cf.ensure_exemplar_embeddings(config, download_fn=download), target)
            self.assertEqual(open(target, "rb").read(), payload)
            self.assertEqual(cf.ensure_exemplar_embeddings(config, download_fn=download), target)
            self.assertEqual(len(calls), 1, "an existing file is not downloaded again")
            os.remove(target)
            with patch.dict(source, sha256="0" * 64), self.assertRaises(ValueError):
                cf.ensure_exemplar_embeddings(config, download_fn=download)
            with self.assertRaises(FileNotFoundError):
                cf.ensure_exemplar_embeddings({"EXEMPLAR_CORPUS": "NuminaMath-CoT",
                                               "EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH": target})

    def test_manifest_hash_is_unchanged_with_the_filter_off(self):
        base = {"TOP_N_CANDIDATES_RETRIEVAL": 5, "EXEMPLAR_CORPUS_NAME": "AI-MO/NuminaMath-CoT"}
        off = {**base, "EXEMPLAR_CORPUS": "NuminaMath-CoT", "CORPUS_FILTER_ENABLED": False,
               "CORPUS_FILTER_EXCLUDE_SYNTHETIC": True, "CORPUS_FILTER_EXCLUDE_DUPLICATES": True,
               "CORPUS_FILTER_DUPLICATE_MATCH": "ignore_numbers"}
        self.assertEqual(sanitize_scientific_config(off), sanitize_scientific_config(base))
        on = sanitize_scientific_config({**off, "CORPUS_FILTER_ENABLED": True})
        self.assertTrue(on["CORPUS_FILTER_ENABLED"])
        self.assertEqual(on["CORPUS_FILTER_DUPLICATE_MATCH"], "ignore_numbers")
        self.assertEqual(sanitize_scientific_config({**off, "EXEMPLAR_CORPUS": "NuminaMath-1.5"})["EXEMPLAR_CORPUS"],
                         "NuminaMath-1.5")


class GuardTest(unittest.TestCase):
    def test_saved_logs_from_another_pool_are_refused(self):
        legacy = [{"target_query_original_hard_list_idx": 0}]
        cf.check_saved_logs(legacy, {"CORPUS_FILTER_ENABLED": False}, "log")
        with self.assertRaisesRegex(RuntimeError, "NuminaMath-CoT, no filter.*hides synthetic"):
            cf.check_saved_logs(legacy, {**ON}, "log")
        stamped = [{"exemplar_corpus": cf.corpus_signature({**ON})}]
        cf.check_saved_logs(stamped, {**ON}, "log")

    def test_attach_builds_the_filter_for_an_old_asset_cell(self):
        rows = dataset()
        data = {"questions": list(rows["problem"]), "solutions": list(rows["solution"]),
                "question_to_index": {q: [i] for i, q in enumerate(rows["problem"])}}
        built = cf.attach_corpus_filter(data, {"EXEMPLAR_CORPUS": "NuminaMath-1.5", **ON},
                                        load_dataset_fn=lambda config: rows)
        self.assertIs(data["question_to_index"].corpus_filter, built)
        self.assertEqual(built.counts()["kept"], 5)
        self.assertIs(cf.attach_corpus_filter(data, {"EXEMPLAR_CORPUS": "NuminaMath-1.5", **ON}), built)
        with self.assertRaisesRegex(ValueError, "Rerun the asset-loading cell"):
            cf.attach_corpus_filter(data, {"EXEMPLAR_CORPUS": "NuminaMath-1.5", "CORPUS_FILTER_ENABLED": False})
        self.assertIsNone(cf.attach_corpus_filter({}, {**ON}))
        self.assertIsNone(cf.attach_corpus_filter({"questions": ["q"]}, {"CORPUS_FILTER_ENABLED": False}))
        with self.assertRaisesRegex(ValueError, "row for row"):
            cf.attach_corpus_filter({"questions": ["other"]}, {**ON}, load_dataset_fn=lambda config: rows)

    def test_run_experiments_preflight(self):
        data = cf.build_exemplar_data(dataset(), np.eye(len(ROWS)), {"EXEMPLAR_CORPUS": "NuminaMath-1.5", **ON})
        config = {"EXEMPLAR_CORPUS": "NuminaMath-1.5", **ON, "TARGET_BENCHMARK": "numina_hard"}
        with tempfile.TemporaryDirectory() as results:
            config["RESULTS_DIR"] = results
            with self.assertRaisesRegex(ValueError, "not per experiment"):
                orchestration._check_exemplar_corpus([{"experiment_name": "e", "CORPUS_FILTER_ENABLED": False}],
                                                     config, data, [ROWS[0][0]])
            with self.assertRaisesRegex(ValueError, "1 numina_hard target"):
                orchestration._check_exemplar_corpus([{"experiment_name": "e"}], config, data,
                                                     [ROWS[0][0], ROWS[5][0]])
            mixed = {**config, "TARGET_BENCHMARK_BY_INDEX": ["math500", "numina_hard"]}
            orchestration._check_exemplar_corpus([{"experiment_name": "e"}], mixed, data, [ROWS[5][0], ROWS[0][0]])
            with open(os.path.join(results, "e_run_log.json"), "w") as handle:
                json.dump([{"target_query_original_hard_list_idx": 0}], handle)
            with self.assertRaisesRegex(RuntimeError, "Mixing them"):
                orchestration._check_exemplar_corpus([{"experiment_name": "e"}], config, data, [ROWS[0][0]])

    def test_layer1_cache_from_another_pool_is_not_reused(self):
        def run(snapshot, config):
            cached = {"metadata": {"last_completed_step": 4, "config_snapshot": {
                "LAYER1_ZERO_SHOT_CANDIDATES_N": 0, **snapshot}}}
            with patch.object(layer1, "_load_cached_state", return_value=cached), \
                    patch.object(layer1, "_execute_retrieval", return_value={"status": "FAILURE"}):
                state = layer1.run_layer1_base_execution(
                    0, "q", "a", None, [], [], None, {"question_to_index": {}}, None, None,
                    {"LAYER1_ZERO_SHOT_CANDIDATES_N": 0, **config})
            return state is cached, state

        self.assertTrue(run({}, {})[0], "a legacy cache stays usable without the filter")
        reused, state = run({}, {**ON})
        self.assertFalse(reused)
        self.assertEqual(state["metadata"]["config_snapshot"]["EXEMPLAR_CORPUS_SIGNATURE"], cf.corpus_signature(ON))
        self.assertTrue(run({"EXEMPLAR_CORPUS_SIGNATURE": cf.corpus_signature(ON)}, {**ON})[0])


if __name__ == "__main__":
    unittest.main()
