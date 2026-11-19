"""Which exemplar-corpus rows the pipeline may use.

NuminaMath contains synthetic rows (LLM-written or rewritten problems) and many
copies of the same problem (reformatted, re-sourced, or with only the numbers
changed). With ``CORPUS_FILTER_ENABLED`` (the default) those rows are hidden from
every process that reads the corpus: retrieval, ``numina_hard`` targets,
hard-question selection, the transformation-dataset corpus targets and the
fine-tuning retrieval. Row indices never change; hidden rows are only skipped,
so logs, embeddings and index files keep pointing at the same rows.

Synthetic rows
    NuminaMath-1.5: the dataset's ``synthetic`` column (it is True exactly for
    the sources metamath, orca_math and synthetic_math). NuminaMath-CoT has no
    such column; its synthetic sources are orca_math, synthetic_amc and
    synthetic_math (synthetic_amc was dropped from 1.5 for that reason).

Duplicate rows
    Problems with the same text key form a group and only one row of each group
    stays (a non-synthetic, valid row first, then the lowest index). The key is
    ``"normalized_text"`` (case, spacing, LaTeX spacing, braces and blank lines
    ignored) or ``"ignore_numbers"`` (the default: also every number replaced,
    so "the same problem with other numbers" is one group). A target question
    is also never given any corpus row with its own key, so copies of a
    benchmark question cannot be retrieved for it.

The filter travels with ``exemplar_data["question_to_index"]`` (an
:class:`ExemplarIndex`), which every retrieval call already receives, and with
``exemplar_data["corpus_filter"]``.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Corpus presets
# --------------------------------------------------------------------------

DEFAULT_EXEMPLAR_CORPUS = "NuminaMath-CoT"

EXEMPLAR_CORPUS_PRESETS: Dict[str, Dict[str, Any]] = {
    "NuminaMath-CoT": {
        "hf_name": "AI-MO/NuminaMath-CoT",
        "revision": None,  # the pipeline has always used the current head
        "local_dir": "NuminaMath-CoT",
        "embedding_file": "embedding_NuminaMath_with_Bert-MLM_arXiv-MP-class_zbMath.npy",
        "embedding_source": None,  # restored with the Hugging Face workspace
        "rows": 859494,
        "synthetic_column": None,
        "synthetic_sources": ("orca_math", "synthetic_amc", "synthetic_math"),
    },
    "NuminaMath-1.5": {
        "hf_name": "AI-MO/NuminaMath-1.5",
        "revision": "1b05109f9e5c1ad06c0663519502416c30b300f8",  # the embedded rows
        "local_dir": "NuminaMath-1.5",
        "embedding_file": "embedding_NuminaMath-1.5_with_Bert-MLM_arXiv-MP-class_zbMath.npy",
        "embedding_source": {
            "repo_id": "mostafabehroozi/numinamath-2025",
            "repo_type": "dataset",
            "revision": "7efce386a5bbaf50039d3f66ac76b87066f177e9",
            "filename": "embeddings/embedding_NuminaMath-1.5_with_Bert-MLM_arXiv-MP-class_zbMath.npy",
            "bytes": 2753172608,
            "sha256": "bd494892fb3978557aa679591d3c21283bc966730976be61ac9d45e9a89d9756",
        },
        # Embedding files made with other models (EXEMPLAR_EMBEDDING_MODEL); the keys above are the BERT file.
        "other_embeddings": {
            "MathLeap-Qwen-8B": {
                "embedding_file": "embedding_NuminaMath-1.5_with_MathLeap-Qwen-8B.npy",
                "embedding_source": {
                    "repo_id": "mostafabehroozi/numinamath-2025",
                    "repo_type": "dataset",
                    "revision": "2645d854331504e53fe48dce751afd46b5f462b8",
                    "filename": "embeddings/embedding_NuminaMath-1.5_with_MathLeap-Qwen-8B.npy",
                    "bytes": 7341793408,
                    "sha256": "0dfe92fee277c7a29236d13eb0caa6af61bb4d0fe24c6b15199e8643880f5cba",
                },
            },
        },
        "rows": 896215,
        "synthetic_column": "synthetic",
        "synthetic_sources": ("metamath", "orca_math", "synthetic_math"),
    },
}

# The model that embedded the corpus file must also embed every query, with the same settings.
DEFAULT_EMBEDDING_MODEL = "Bert-MLM_arXiv-MP-class_zbMath"
EMBEDDING_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "Bert-MLM_arXiv-MP-class_zbMath": {
        "hf_name": "math-similarity/Bert-MLM_arXiv-MP-class_zbMath",
        "revision": None,
        "max_seq_length": None,   # the model's own setting, as always
        "large": False,
    },
    "MathLeap-Qwen-8B": {
        "hf_name": "uw-math-ai/MathLeap-Qwen-8B",
        "revision": "ffe9c2a4a1b7777b890044e40a4e4a4d6346ae23",  # the revision the corpus file was made with
        "max_seq_length": 512,    # the corpus file used 512 tokens (the model's default is 128)
        "large": True,            # 8B parameters: loaded in 16-bit, ~16 GB
    },
}

# Used for a corpus that is not a preset (a custom path) and has no flag column.
KNOWN_SYNTHETIC_SOURCES = frozenset({"metamath", "orca_math", "synthetic_amc", "synthetic_math"})

DUPLICATE_MATCH_MODES = ("ignore_numbers", "normalized_text")
FILTER_CONFIG_KEYS = (
    "CORPUS_FILTER_ENABLED",
    "CORPUS_FILTER_EXCLUDE_SYNTHETIC",
    "CORPUS_FILTER_EXCLUDE_DUPLICATES",
    "CORPUS_FILTER_DUPLICATE_MATCH",
)
CORPUS_CONFIG_KEYS = ("EXEMPLAR_CORPUS",) + FILTER_CONFIG_KEYS

# What logs and caches written before the filter existed were made with.
LEGACY_CORPUS_SIGNATURE = {"corpus": DEFAULT_EXEMPLAR_CORPUS, "filter": "off"}


def corpus_preset(config: Mapping[str, Any]) -> Dict[str, Any]:
    name = config.get("EXEMPLAR_CORPUS") or DEFAULT_EXEMPLAR_CORPUS
    if name not in EXEMPLAR_CORPUS_PRESETS:
        raise ValueError(
            f"EXEMPLAR_CORPUS must be one of {sorted(EXEMPLAR_CORPUS_PRESETS)}, got {name!r}."
        )
    return EXEMPLAR_CORPUS_PRESETS[name]


def embedding_preset(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The chosen embedding model (``EXEMPLAR_EMBEDDING_MODEL``) and its file for the chosen corpus."""
    name = config.get("EXEMPLAR_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    if name not in EMBEDDING_MODEL_PRESETS:
        raise ValueError(
            f"EXEMPLAR_EMBEDDING_MODEL must be one of {sorted(EMBEDDING_MODEL_PRESETS)}, got {name!r}."
        )
    corpus = corpus_preset(config)
    if name == DEFAULT_EMBEDDING_MODEL:
        files = {"embedding_file": corpus["embedding_file"], "embedding_source": corpus.get("embedding_source")}
    else:
        files = corpus.get("other_embeddings", {}).get(name)
        if files is None:
            raise ValueError(
                f"There is no {name} embedding file for {config.get('EXEMPLAR_CORPUS') or DEFAULT_EXEMPLAR_CORPUS}; "
                f"use EXEMPLAR_EMBEDDING_MODEL={DEFAULT_EMBEDDING_MODEL!r} or another corpus."
            )
    return {"name": name, **EMBEDDING_MODEL_PRESETS[name], **files}


def apply_exemplar_corpus(config: Dict[str, Any]) -> Dict[str, Any]:
    """Point the corpus names and paths at ``config["EXEMPLAR_CORPUS"]`` and the
    embedding model and file at ``config["EXEMPLAR_EMBEDDING_MODEL"]``.

    Names the user replaced with a custom location (anything that is not a
    preset's Hugging Face name) are left alone.
    """
    preset = corpus_preset(config)
    embedding = embedding_preset(config)
    preset_names = {p["hf_name"] for p in EXEMPLAR_CORPUS_PRESETS.values()}
    for key in ("EXEMPLAR_CORPUS_NAME_HF", "EXEMPLAR_CORPUS_NAME"):
        if config.get(key) in (None, "", *preset_names):
            config[key] = preset["hf_name"]
    model_names = {p["hf_name"] for p in EMBEDDING_MODEL_PRESETS.values()}
    for key in ("EMBEDDING_MODEL_PATH_HF", "EMBEDDING_MODEL_PATH"):
        if config.get(key) in (None, "", *model_names):
            config[key] = embedding["hf_name"]
    if config.get("DATA_DIR"):
        config["LOCAL_EXEMPLAR_CORPUS_PATH"] = os.path.join(config["DATA_DIR"], preset["local_dir"])
        config["LOCAL_EMBEDDING_MODEL_PATH"] = os.path.join(config["DATA_DIR"], embedding["name"])
    if config.get("EMBEDDINGS_DIR"):
        config["EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH"] = os.path.join(
            config["EMBEDDINGS_DIR"], embedding["embedding_file"]
        )
    return config


def filter_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validated filter settings; ``enabled`` is False when nothing is hidden.

    The default (on) lives in config.py's CONFIG. A partial config without
    ``CORPUS_FILTER_ENABLED`` keeps the behaviour from before the filter existed.
    """
    enabled = config.get("CORPUS_FILTER_ENABLED", False)
    synthetic = config.get("CORPUS_FILTER_EXCLUDE_SYNTHETIC", True)
    duplicates = config.get("CORPUS_FILTER_EXCLUDE_DUPLICATES", True)
    match = config.get("CORPUS_FILTER_DUPLICATE_MATCH", "ignore_numbers")
    for key, value in (("CORPUS_FILTER_ENABLED", enabled), ("CORPUS_FILTER_EXCLUDE_SYNTHETIC", synthetic),
                       ("CORPUS_FILTER_EXCLUDE_DUPLICATES", duplicates)):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be True or False, got {value!r}.")
    if match not in DUPLICATE_MATCH_MODES:
        raise ValueError(f"CORPUS_FILTER_DUPLICATE_MATCH must be one of {DUPLICATE_MATCH_MODES}, got {match!r}.")
    return {
        "enabled": bool(enabled and (synthetic or duplicates)),
        "exclude_synthetic": bool(enabled and synthetic),
        "exclude_duplicates": bool(enabled and duplicates),
        "duplicate_match": match,
    }


def corpus_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact identity of the corpus and filter, stored in logs and caches."""
    settings = filter_settings(config)
    name = config.get("EXEMPLAR_CORPUS") or DEFAULT_EXEMPLAR_CORPUS
    if not settings["enabled"]:
        return {"corpus": name, "filter": "off"}
    return {
        "corpus": name,
        "filter": {
            "synthetic": settings["exclude_synthetic"],
            "duplicates": settings["duplicate_match"] if settings["exclude_duplicates"] else False,
        },
    }


def describe_signature(signature: Optional[Mapping[str, Any]]) -> str:
    signature = signature or LEGACY_CORPUS_SIGNATURE
    rule = signature.get("filter")
    if rule == "off":
        text = "no filter"
    else:
        parts = (["synthetic"] if rule.get("synthetic") else []) + (
            [f"duplicates ({rule['duplicates']})"] if rule.get("duplicates") else [])
        text = "hides " + " and ".join(parts)
    return f"{signature.get('corpus')}, {text}"


# --------------------------------------------------------------------------
# Text keys
# --------------------------------------------------------------------------

_LATEX_NOISE = re.compile(
    r"\\(?:[,;:! ]|quad\b|qquad\b|left\b|right\b|displaystyle\b|textstyle\b|limits\b|[\[\]()])|\$"
)
_BLANKS = re.compile(r"(?:\\_|_)+")
_DROP = re.compile(r"[\s{}.,;:!?'\"`~]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalized_text_key(text: Any) -> str:
    """The problem with case, spacing, LaTeX spacing, braces and punctuation ignored.

    Operators (+ - * / = < > ^ and brackets) are kept, so "a+b" and "a-b" differ.
    """
    value = str(text or "").lower()
    value = _LATEX_NOISE.sub("", value)
    value = _BLANKS.sub("_", value)
    return _DROP.sub("", value)


def numbers_ignored_key(text: Any) -> str:
    """:func:`normalized_text_key` with every number replaced by ``#``."""
    return _NUMBER.sub("#", normalized_text_key(text))


DUPLICATE_KEY_FUNCTIONS = {"normalized_text": normalized_text_key, "ignore_numbers": numbers_ignored_key}


def _key_hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


# --------------------------------------------------------------------------
# The filter
# --------------------------------------------------------------------------

KEPT, SYNTHETIC, DUPLICATE, EMPTY = 0, 1, 2, 3
REASON_NAMES = {SYNTHETIC: "synthetic", DUPLICATE: "duplicate", EMPTY: "empty"}


@dataclass
class CorpusFilter:
    """Rows hidden from the pipeline, and the copies a given question must not see."""

    corpus: str
    settings: Dict[str, Any]
    reasons: np.ndarray                      # uint8 per row: KEPT, SYNTHETIC, DUPLICATE or EMPTY
    sources: Optional[np.ndarray] = None     # per-row source names, for reports
    _sorted_hashes: Optional[np.ndarray] = field(default=None, repr=False)
    _hash_order: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def excluded(self) -> np.ndarray:
        return self.reasons != KEPT

    @property
    def size(self) -> int:
        return int(self.reasons.shape[0])

    @property
    def signature(self) -> Dict[str, Any]:
        return corpus_signature({"EXEMPLAR_CORPUS": self.corpus, **_settings_to_config(self.settings)})

    def allowed_indices(self) -> np.ndarray:
        return np.flatnonzero(self.reasons == KEPT)

    def is_excluded(self, index: int) -> bool:
        return bool(self.reasons[int(index)] != KEPT)

    def reason(self, index: int) -> Optional[str]:
        return REASON_NAMES.get(int(self.reasons[int(index)]))

    def counts(self) -> Dict[str, int]:
        return {name: int((self.reasons == code).sum()) for code, name in REASON_NAMES.items()} | {
            "kept": int((self.reasons == KEPT).sum()), "total": self.size}

    def copies_of(self, text: Any) -> np.ndarray:
        """Every corpus row whose duplicate key equals the key of ``text``."""
        if self._sorted_hashes is None:
            return np.zeros(0, dtype=np.int64)
        key_fn = DUPLICATE_KEY_FUNCTIONS[self.settings["duplicate_match"]]
        target = _key_hash(key_fn(text))
        lo = int(np.searchsorted(self._sorted_hashes, target, side="left"))
        hi = int(np.searchsorted(self._sorted_hashes, target, side="right"))
        return np.sort(self._hash_order[lo:hi])

    def apply(self, similarities: np.ndarray, target_query: Any = None) -> Dict[str, int]:
        """Set hidden rows and copies of ``target_query`` to -inf, in place."""
        if similarities.shape[0] != self.size:
            raise ValueError(
                f"The corpus filter covers {self.size} rows but the similarity array has "
                f"{similarities.shape[0]}; rebuild exemplar_data from the loaded corpus."
            )
        similarities[self.excluded] = -np.inf
        copies = self.copies_of(target_query) if target_query is not None else np.zeros(0, dtype=np.int64)
        if copies.size:
            similarities[copies] = -np.inf
        return {"hidden_rows": int(self.excluded.sum()), "target_copies": int(copies.size)}

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"corpus": self.corpus, "rule": describe_signature(self.signature), **self.counts()}
        if self.sources is not None:
            out["hidden_by_source"] = {
                str(source): int(count) for source, count in zip(*np.unique(self.sources[self.excluded],
                                                                              return_counts=True))
            }
        return out

    def report(self) -> str:
        c = self.counts()
        lines = [f"Exemplar corpus {self.corpus}: {c['kept']:,} of {c['total']:,} rows usable "
                 f"({describe_signature(self.signature)})"]
        if self.settings["enabled"]:
            lines.append(f"  hidden: {c['synthetic']:,} synthetic, {c['duplicate']:,} duplicates, {c['empty']:,} empty")
            if self.sources is not None and c["total"] > c["kept"]:
                by_source = self.summary()["hidden_by_source"]
                lines.append("  hidden by source: " + ", ".join(
                    f"{name} {count:,}" for name, count in sorted(by_source.items(), key=lambda kv: -kv[1])))
        return "\n".join(lines)


def _settings_to_config(settings: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CORPUS_FILTER_ENABLED": bool(settings["enabled"]),
        "CORPUS_FILTER_EXCLUDE_SYNTHETIC": bool(settings["exclude_synthetic"]),
        "CORPUS_FILTER_EXCLUDE_DUPLICATES": bool(settings["exclude_duplicates"]),
        "CORPUS_FILTER_DUPLICATE_MATCH": settings["duplicate_match"],
    }


def _column(dataset: Any, name: str) -> Optional[List[Any]]:
    names = getattr(dataset, "column_names", None)
    if names is not None and name not in names:
        return None
    try:
        return list(dataset[name])
    except (KeyError, TypeError, ValueError):
        return None


def build_corpus_filter(
    dataset: Any = None,
    config: Optional[Mapping[str, Any]] = None,
    *,
    questions: Optional[Sequence[Any]] = None,
    sources: Optional[Sequence[Any]] = None,
    synthetic: Optional[Sequence[Any]] = None,
    problem_valid: Optional[Sequence[Any]] = None,
    verbose: bool = True,
) -> CorpusFilter:
    """Build the filter for a loaded corpus (a ``datasets.Dataset`` or plain columns)."""
    config = dict(config or {})
    settings = filter_settings(config)
    corpus = config.get("EXEMPLAR_CORPUS") or DEFAULT_EXEMPLAR_CORPUS
    preset = EXEMPLAR_CORPUS_PRESETS.get(corpus, {})
    started = time.time()

    if questions is None:
        questions = _column(dataset, "problem")
        if questions is None:
            raise ValueError("The exemplar corpus has no 'problem' column.")
    if sources is None and dataset is not None:
        sources = _column(dataset, "source")
    if synthetic is None and dataset is not None and preset.get("synthetic_column", "synthetic"):
        synthetic = _column(dataset, preset.get("synthetic_column") or "synthetic")
    if problem_valid is None and dataset is not None:
        problem_valid = _column(dataset, "problem_is_valid")
    n = len(questions)
    for name, column in (("source", sources), ("synthetic", synthetic), ("problem_is_valid", problem_valid)):
        if column is not None and len(column) != n:
            raise ValueError(f"Column {name!r} has {len(column)} rows; 'problem' has {n}.")

    source_array = None if sources is None else np.asarray([str(s).strip().lower() for s in sources], dtype=object)
    if synthetic is not None:
        is_synthetic = np.asarray([bool(v) for v in synthetic], dtype=bool)
    elif source_array is not None:
        names = set(preset.get("synthetic_sources", KNOWN_SYNTHETIC_SOURCES))
        is_synthetic = np.asarray([s in names for s in source_array], dtype=bool)
    else:
        is_synthetic = np.zeros(n, dtype=bool)
    invalid = np.zeros(n, dtype=bool) if problem_valid is None else np.asarray(
        [str(v).strip().lower() != "yes" for v in problem_valid], dtype=bool)

    reasons = np.zeros(n, dtype=np.uint8)
    sorted_hashes = hash_order = None
    if settings["enabled"]:
        if settings["exclude_synthetic"]:
            reasons[is_synthetic] = SYNTHETIC
        if settings["exclude_duplicates"]:
            key_fn = DUPLICATE_KEY_FUNCTIONS[settings["duplicate_match"]]
            keys = [key_fn(q) for q in questions]
            empty = np.asarray([not k for k in keys], dtype=bool)
            reasons[empty & (reasons == KEPT)] = EMPTY
            hashes = np.fromiter((_key_hash(k) for k in keys), dtype=np.int64, count=n)
            # Inside each key group keep one row: non-synthetic, then valid, then lowest index.
            order = np.lexsort((np.arange(n), invalid, is_synthetic, hashes))
            candidates = order[reasons[order] == KEPT]
            first = np.ones(candidates.size, dtype=bool)
            first[1:] = hashes[candidates[1:]] != hashes[candidates[:-1]]
            reasons[candidates[~first]] = DUPLICATE
            hash_order = np.argsort(hashes, kind="stable")
            sorted_hashes = hashes[hash_order]

    result = CorpusFilter(corpus=corpus, settings=settings, reasons=reasons, sources=source_array,
                          _sorted_hashes=sorted_hashes, _hash_order=hash_order)
    if verbose:
        print(result.report() + f"\n  (built in {time.time() - started:.0f}s)", flush=True)
    return result


# --------------------------------------------------------------------------
# exemplar_data
# --------------------------------------------------------------------------

class ExemplarIndex(dict):
    """``question text -> [row indices]`` that also carries the corpus filter.

    Every retrieval call already receives ``question_to_index_map``; carrying
    the filter here means no retrieval path can skip it.
    """

    corpus_filter: Optional[CorpusFilter] = None

    def __init__(self, *args: Any, corpus_filter: Optional[CorpusFilter] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.corpus_filter = corpus_filter

    def __reduce__(self):
        return (self.__class__, (dict(self),), {"corpus_filter": self.corpus_filter})


def build_question_index(questions: Iterable[Any], corpus_filter: Optional[CorpusFilter] = None) -> ExemplarIndex:
    index = ExemplarIndex(corpus_filter=corpus_filter)
    for row, question in enumerate(questions):
        index.setdefault(question, []).append(row)
    return index


def build_exemplar_data(dataset: Any, embeddings: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    """The aligned corpus the pipeline reads: questions, solutions, embeddings, index, filter."""
    questions = list(dataset["problem"])
    solutions = list(dataset["solution"])
    if not (len(questions) == len(solutions) == len(embeddings)):
        raise ValueError(
            "Exemplar corpus/solution/embedding lengths do not match: "
            f"{len(questions)}, {len(solutions)}, {len(embeddings)}. The embedding file must be made "
            f"from the same corpus (EXEMPLAR_CORPUS={config.get('EXEMPLAR_CORPUS', DEFAULT_EXEMPLAR_CORPUS)!r})."
        )
    corpus_filter = build_corpus_filter(dataset, config, questions=questions)
    return {
        "questions": questions,
        "solutions": solutions,
        "embeddings": embeddings,
        # Keep every matching row: duplicate questions must all be excluded when one is a target.
        "question_to_index": build_question_index(questions, corpus_filter),
        "corpus_filter": corpus_filter,
        "corpus_signature": corpus_filter.signature,
    }


def corpus_filter_of(exemplar_data: Optional[Mapping[str, Any]]) -> Optional[CorpusFilter]:
    if not exemplar_data:
        return None
    found = exemplar_data.get("corpus_filter")
    if found is None:
        found = getattr(exemplar_data.get("question_to_index"), "corpus_filter", None)
    return found


def attach_corpus_filter(
    exemplar_data: Dict[str, Any],
    config: Mapping[str, Any],
    *,
    load_dataset_fn: Any = None,
) -> Optional[CorpusFilter]:
    """Make sure ``exemplar_data`` carries the filter that ``config`` asks for.

    ``exemplar_data`` built by :func:`build_exemplar_data` already has it (and is
    checked against ``config``). An older notebook cell that builds the dict by
    hand gets the filter here, from the corpus dataset, so the switch cannot be
    skipped silently.
    """
    current = corpus_signature(config)
    existing = corpus_filter_of(exemplar_data)
    if existing is not None:
        if existing.signature != current:
            raise ValueError(
                f"exemplar_data was built with [{describe_signature(existing.signature)}] but CONFIG now asks for "
                f"[{describe_signature(current)}]. Rerun the asset-loading cell after changing "
                "EXEMPLAR_CORPUS or CORPUS_FILTER_*."
            )
        index = exemplar_data.get("question_to_index")
        if getattr(index, "corpus_filter", None) is not existing:
            exemplar_data["question_to_index"] = ExemplarIndex(
                index if index is not None else build_question_index(exemplar_data.get("questions", [])),
                corpus_filter=existing,
            )
        exemplar_data["corpus_filter"] = existing
        return existing
    questions = exemplar_data.get("questions") if exemplar_data else None
    if not filter_settings(config)["enabled"] or questions is None or len(questions) == 0:
        return None  # nothing to filter (no corpus loaded, e.g. zero-shot runs)

    print("exemplar_data has no corpus filter (an older asset cell); building it from the corpus dataset.", flush=True)
    if load_dataset_fn is None:
        from src.utils import load_exemplar_corpus as load_dataset_fn
    dataset = load_dataset_fn(dict(config))
    if dataset is None:
        raise RuntimeError(
            "CORPUS_FILTER_ENABLED is on but the corpus dataset could not be loaded to build the filter. "
            "Build exemplar_data with src.corpus_filter.build_exemplar_data, or set CORPUS_FILTER_ENABLED=False."
        )
    if len(dataset) != len(questions) or list(dataset["problem"]) != list(questions):
        raise ValueError(
            "The corpus dataset does not match exemplar_data['questions'] row for row, so the filter cannot be "
            "built for it. Load exemplar_data with src.corpus_filter.build_exemplar_data."
        )
    built = build_corpus_filter(dataset, config, questions=questions)
    index = exemplar_data.get("question_to_index")
    exemplar_data["question_to_index"] = ExemplarIndex(
        index if index is not None else build_question_index(questions), corpus_filter=built)
    exemplar_data["corpus_filter"] = built
    exemplar_data["corpus_signature"] = built.signature
    return built


def hidden_target_positions(
    targets: Sequence[str],
    exemplar_data: Mapping[str, Any],
    positions: Optional[Iterable[int]] = None,
) -> List[int]:
    """Target positions whose text is only found in hidden corpus rows."""
    found = corpus_filter_of(exemplar_data)
    index = exemplar_data.get("question_to_index") or {}
    if found is None or not found.settings.get("enabled"):
        return []
    hidden = []
    for position in (range(len(targets)) if positions is None else positions):
        rows = index.get(targets[position])
        if rows is None:
            continue
        # Older asset cells map a question to one row number instead of a list.
        rows = [rows] if isinstance(rows, (int, np.integer)) else list(rows)
        if rows and all(found.is_excluded(row) for row in rows):
            hidden.append(int(position))
    return hidden


def signature_of_exemplar_data(exemplar_data: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    found = corpus_filter_of(exemplar_data)
    if found is not None:
        return found.signature
    if exemplar_data and exemplar_data.get("corpus_signature"):
        return dict(exemplar_data["corpus_signature"])
    return None


# --------------------------------------------------------------------------
# Loading the embedding file
# --------------------------------------------------------------------------

def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_exemplar_embeddings(config: Mapping[str, Any], *, download_fn: Any = None) -> str:
    """Return the embedding file path, downloading the preset's file when it is missing."""
    path = config.get("EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH")
    if not path:
        raise ValueError("EMBEDDED_EXEMPLAR_CORPUS_QUESTIONS_PATH is not set.")
    if os.path.isfile(path):
        return path
    embedding = embedding_preset(config)
    source = embedding.get("embedding_source")
    if not source:
        raise FileNotFoundError(
            f"Embedding file not found: {path}. For {config.get('EXEMPLAR_CORPUS', DEFAULT_EXEMPLAR_CORPUS)} "
            "it comes with the Hugging Face workspace restore; place the matching .npy there."
        )
    if download_fn is None:
        from huggingface_hub import hf_hub_download as download_fn
    folder = os.path.dirname(os.path.abspath(path))
    staging = os.path.join(folder, ".exemplar_embedding_download")
    os.makedirs(staging, exist_ok=True)
    print(f"Downloading the {config.get('EXEMPLAR_CORPUS')} {embedding['name']} embedding file "
          f"({source['bytes'] / 1e9:.2f} GB) from {source['repo_id']} ...", flush=True)
    started = time.time()
    token = config.get("HF_SYNC_TOKEN")
    token = token if token and "YOUR_" not in str(token).upper() else None
    downloaded = download_fn(
        repo_id=source["repo_id"], repo_type=source["repo_type"], revision=source["revision"],
        filename=source["filename"], local_dir=staging, token=token,
    )
    size = os.path.getsize(downloaded)
    if size != source["bytes"] or _sha256(downloaded) != source["sha256"]:
        raise ValueError(f"The downloaded embedding file {downloaded} does not match the expected size/sha256.")
    os.replace(downloaded, path)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Embedding file ready in {time.time() - started:.0f}s: {path}", flush=True)
    return path


# --------------------------------------------------------------------------
# Guards for saved work
# --------------------------------------------------------------------------

def log_signature(run_log: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(run_log.get("exemplar_corpus") or LEGACY_CORPUS_SIGNATURE)


def check_saved_logs(run_logs: Sequence[Mapping[str, Any]], config: Mapping[str, Any], where: str) -> None:
    """Refuse to continue saved logs that were made with another corpus or filter."""
    current = corpus_signature(config)
    for run_log in run_logs or []:
        if not isinstance(run_log, Mapping):
            continue
        saved = log_signature(run_log)
        if saved != current:
            raise RuntimeError(
                f"{where} holds results made with [{describe_signature(saved)}], but this run uses "
                f"[{describe_signature(current)}]. Mixing them would mix two retrieval pools. "
                "Use a new experiment_name, or restore the old settings "
                "(EXEMPLAR_CORPUS / CORPUS_FILTER_ENABLED=False for runs started before the filter existed)."
            )
