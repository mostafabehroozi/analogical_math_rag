"""Saved embeddings of the benchmark questions (BENCHMARK_EMBEDDINGS_ENABLED).

Retrieval embeds each target question with the embedding model. For MathLeap-Qwen-8B
that costs about 77 s per question on a Kaggle CPU and 16 GB of RAM, so the questions
of every supported benchmark are embedded once on a GPU
(``benchmark_embeddings_kaggle.ipynb``) and saved per embedding model:

    benchmark_embeddings/<model name>/manifest.json
    benchmark_embeddings/<model name>/<benchmark>.npy          float32 [questions, dim]
    benchmark_embeddings/<model name>/<benchmark>_texts.json   the questions, row for row

:class:`SavedQueryEncoder` answers ``encode()`` from those vectors, matched by the exact
question text the pipeline embeds, and loads the real model only for a text it does
not have (for example a benchmark added later), so results never depend on whether a
vector was saved.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

MANIFEST_FILE = "manifest.json"
FOLDER_IN_REPO = "benchmark_embeddings"


def text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_embeddings_dir(config: Mapping[str, Any]) -> str:
    from src.corpus_filter import embedding_preset

    if config.get("BENCHMARK_EMBEDDINGS_DIR"):
        return str(config["BENCHMARK_EMBEDDINGS_DIR"])
    return os.path.join(config.get("EMBEDDINGS_DIR") or ".", FOLDER_IN_REPO, embedding_preset(config)["name"])


def ensure_benchmark_embeddings(config: Mapping[str, Any], *, download_fn: Any = None) -> str:
    """The local folder of the chosen model's saved vectors, downloaded from Hugging Face if missing."""
    from src.corpus_filter import embedding_preset

    folder = benchmark_embeddings_dir(config)
    if os.path.isfile(os.path.join(folder, MANIFEST_FILE)):
        return folder
    preset = embedding_preset(config)
    source = preset.get("benchmark_embeddings_source")
    if not source:
        raise FileNotFoundError(f"No saved benchmark embeddings in {folder} and none published for {preset['name']}.")
    if download_fn is None:
        from huggingface_hub import snapshot_download as download_fn
    staging = os.path.join(os.path.dirname(os.path.abspath(folder)), ".benchmark_embedding_download")
    token = config.get("HF_SYNC_TOKEN")
    token = token if token and "YOUR_" not in str(token).upper() else None
    print(f"Downloading the saved {preset['name']} benchmark embeddings from {source['repo_id']} ...", flush=True)
    download_fn(repo_id=source["repo_id"], repo_type=source["repo_type"], revision=source["revision"],
                allow_patterns=[f"{source['folder']}/*"], local_dir=staging, token=token)
    os.makedirs(os.path.dirname(os.path.abspath(folder)), exist_ok=True)
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.replace(os.path.join(staging, source["folder"]), folder)
    shutil.rmtree(staging, ignore_errors=True)
    return folder


def load_saved_vectors(folder: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Read and check one model folder. Returns ``{"vectors": {text_key: vector}, "manifest": ...}``."""
    from src.corpus_filter import embedding_preset

    preset = embedding_preset(config)
    with open(os.path.join(folder, MANIFEST_FILE), encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("model_id") != preset["hf_name"]:
        raise ValueError(f"{folder} holds {manifest.get('model_id')} vectors, but the chosen embedding model is "
                         f"{preset['hf_name']} (EXEMPLAR_EMBEDDING_MODEL={preset['name']!r}).")
    if preset.get("revision") and manifest.get("model_revision") not in (None, preset["revision"]):
        raise ValueError(f"{folder} was made with {preset['hf_name']} revision {manifest.get('model_revision')}, "
                         f"not {preset['revision']}.")
    if preset.get("max_seq_length") and manifest.get("max_seq_length") != preset["max_seq_length"]:
        raise ValueError(f"{folder} used max_seq_length {manifest.get('max_seq_length')}, "
                         f"not {preset['max_seq_length']}.")
    vectors: Dict[str, np.ndarray] = {}
    counts = {}
    for name, item in sorted(manifest["benchmarks"].items()):
        array_path = os.path.join(folder, item["file"])
        if item.get("sha256") and _sha256(array_path) != item["sha256"]:
            raise ValueError(f"{array_path} does not match the sha256 in {MANIFEST_FILE}.")
        array = np.load(array_path).astype(np.float32, copy=False)
        with open(os.path.join(folder, item["texts_file"]), encoding="utf-8") as handle:
            texts = json.load(handle)
        if array.ndim != 2 or array.shape[0] != len(texts) or array.shape[1] != manifest["dim"]:
            raise ValueError(f"{array_path} has shape {array.shape}, expected [{len(texts)}, {manifest['dim']}].")
        for text, row in zip(texts, array):
            vectors[text_key(text)] = row
        counts[name] = len(texts)
    return {"vectors": vectors, "manifest": manifest, "counts": counts}


class SavedQueryEncoder:
    """``encode()`` from saved question vectors; other texts go to the real model, loaded once on first need."""

    def __init__(self, vectors: Dict[str, np.ndarray], load_model: Callable[[], Any], name: str = "", logger: Any = None):
        self.vectors = vectors
        self._load_model = load_model
        self._model = None
        self._lock = threading.Lock()
        self.name = name
        self.logger = logger
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> Any:
        with self._lock:
            if self._model is None:
                message = (f"Loading the {self.name or 'embedding'} model for a text without a saved vector "
                           "(slow on CPU for MathLeap-Qwen-8B).")
                self.logger.warning(message) if self.logger else print(message, flush=True)
                self._model = self._load_model()
                if self._model is None:
                    raise RuntimeError(f"The {self.name} embedding model could not be loaded.")
            return self._model

    def encode(self, sentences: Any, batch_size: int = 32, show_progress_bar: Optional[bool] = None,
               convert_to_numpy: bool = True, normalize_embeddings: bool = False, **kwargs: Any) -> Any:
        single = isinstance(sentences, str)
        texts: List[str] = [sentences] if single else [str(text) for text in sentences]
        rows: List[Optional[np.ndarray]] = [self.vectors.get(text_key(text)) for text in texts]
        missing = [i for i, row in enumerate(rows) if row is None]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            fresh = np.asarray(self.model.encode([texts[i] for i in missing], batch_size=batch_size,
                                                 show_progress_bar=False, convert_to_numpy=True, **kwargs),
                               dtype=np.float32)
            for i, row in zip(missing, fresh):
                rows[i] = row
        out = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), np.float32)
        if normalize_embeddings and out.size:
            out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out[0] if single else out

    def __repr__(self) -> str:
        return f"SavedQueryEncoder({self.name}, {len(self.vectors)} saved texts, hits={self.hits}, misses={self.misses})"


def saved_query_encoder(config: Mapping[str, Any], load_model: Callable[[], Any], logger: Any = None) -> SavedQueryEncoder:
    from src.corpus_filter import embedding_preset

    folder = ensure_benchmark_embeddings(config)
    loaded = load_saved_vectors(folder, config)
    message = f"Saved benchmark embeddings ({embedding_preset(config)['name']}): {loaded['counts']} from {folder}"
    logger.info(message) if logger else print(message, flush=True)
    return SavedQueryEncoder(loaded["vectors"], load_model, name=embedding_preset(config)["name"], logger=logger)


def write_model_folder(folder: str, model_info: Mapping[str, Any], texts: Mapping[str, List[str]],
                       vectors: Mapping[str, np.ndarray], sources: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write one model's vectors in the layout :func:`load_saved_vectors` reads.

    ``model_info`` holds model_id, model_revision, max_seq_length and anything else to
    record; ``texts[name]`` are the questions exactly as the pipeline embeds them.
    """
    os.makedirs(folder, exist_ok=True)
    dims = {np.asarray(vectors[name]).shape[1] for name in texts}
    if len(dims) != 1:
        raise ValueError(f"All benchmarks must have the same dimension, got {dims}.")
    manifest: Dict[str, Any] = {**model_info, "dim": dims.pop(), "benchmarks": {},
                                "lookup": "sha1 of the exact question text the pipeline embeds"}
    for name in sorted(texts):
        array = np.asarray(vectors[name], dtype=np.float32)
        if array.shape[0] != len(texts[name]) or not np.isfinite(array).all():
            raise ValueError(f"{name}: {array.shape[0]} vectors for {len(texts[name])} texts, or NaN values.")
        array_file, texts_file = f"{name}.npy", f"{name}_texts.json"
        np.save(os.path.join(folder, array_file), array)
        with open(os.path.join(folder, texts_file), "w", encoding="utf-8") as handle:
            json.dump(list(texts[name]), handle, ensure_ascii=False)
        manifest["benchmarks"][name] = {
            "rows": len(texts[name]), "file": array_file, "texts_file": texts_file,
            "sha256": _sha256(os.path.join(folder, array_file)), **((sources or {}).get(name) or {}),
        }
    with open(os.path.join(folder, MANIFEST_FILE), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def benchmark_question_texts(names: List[str], *, hard_indices: Optional[List[int]] = None,
                             corpus_questions: Optional[List[Any]] = None,
                             load_dataset_fn: Any = None) -> Dict[str, List[str]]:
    """The question texts of each benchmark, built by the same code as load_target_benchmarks."""
    from src.benchmark_data import HUGGINGFACE_BENCHMARK_SPECS, load_huggingface_benchmark

    out: Dict[str, List[str]] = {}
    for name in names:
        if name in HUGGINGFACE_BENCHMARK_SPECS:
            out[name] = load_huggingface_benchmark(name, load_dataset_fn)[0]
        elif name == "numina_hard":
            if hard_indices is None or corpus_questions is None:
                raise ValueError("numina_hard needs hard_indices and corpus_questions.")
            out[name] = [str(corpus_questions[index]) for index in hard_indices]
        else:
            raise ValueError(f"Unknown benchmark {name!r}.")
    return out
