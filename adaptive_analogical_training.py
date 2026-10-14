"""Kaggle-compatible ranking -> snapshot -> supervised acquisition training.

The companion notebook embeds this module: uploading the notebook alone is enough.
No provider calls are made. Acquisition is simulated from recorded Layer-1 data.
"""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch import nn

# Stages 1-2 train on every presence state and zero-shot order and select checkpoints by development
# loss. A one-shot candidate does not need its own evaluator. Stage 3 labels every presence state
# with its cheapest routes to the next goal (pretraining), then fine-tunes on the route that the
# pretrained head's own continuations make cheapest.
PIPELINE_REVISION = 9


# %% Configuration and compact data
@dataclass
class Config:
    train_file: str = "/kaggle/working/downloaded_files/dir_5/numina_hard_run_log.json"
    test_files: list = field(default_factory=lambda: [
        "/kaggle/working/downloaded_files/dir_1/aime25_run_log.json",
        "/kaggle/working/downloaded_files/dir_2/aime26_run_log.json",
        "/kaggle/working/downloaded_files/dir_3/gsm8k_run_log.json",
        "/kaggle/working/downloaded_files/dir_4/math500_run_log.json",
    ])
    output_dir: str = "/kaggle/working/adaptive_analogical_shared_training_v9_run"
    resume: bool = True             # Reuse completed stages with identical contracts.
    seed: int = 75
    device: str = "auto"
    cpu_threads: int = 2
    k: int = 5
    zero_shots: int = 3
    repeats: int = 5                # Must match the stored CCS denominator.
    use_test_files_for_dev_and_audit: bool = True
    split_fractions: tuple = (.80, .10, .10)  # shared training / dev / audit; used only when False above
    teacher_folds: int = 5
    hidden_dim: int = 128
    residual_blocks: int = 2
    dropout: float = .10
    # Stages 1-2: one epoch visits every presence state of every question under
    # every zero-shot order once (48,960 views per default question).
    batch_size: int = 16384
    learning_rate: float = .0003
    weight_decay: float = 1e-4
    teacher_epochs: int = 10
    snapshot_epochs: int = 10
    # Complete views (6 of 48,960 per question) are repeated to this share of each epoch.
    teacher_complete_share: float = .15
    snapshot_complete_share: float = 0.   # 0: complete views appear once per order, like any other.
    dev_checks_per_epoch: int = 8   # Stages 1-2 development checks, evenly spaced in each epoch.
    patience: int = 15              # Epochs without a better development loss before training stops.
    auxiliary_weight: float = .25
    calibrate: bool = True          # Development-only temperature selection.
    recognition_threshold: float = .50
    later_rank_filter: bool = True
    continue_after_max: bool = True  # Complete the ranked pool after recognition.
    cost_unit: str = "total_calls"  # Includes evaluator graders; solver_calls remains optional.
    max_cost: Optional[float] = None # None = cost of the complete pool.
    decision_hidden: int = 64
    decision_epochs: int = 100       # Stage 3a: pretraining on every cheapest first step.
    decision_lr: float = 1e-4
    decision_finetune_epochs: int = 50  # Stage 3b: fine-tuning from the pretrained weights.
    decision_finetune_lr: float = 3e-5
    decision_batch_size: int = 128
    decision_eval_batch_size: int = 4096  # Frozen dev forwards; no optimizer changes.
    # Evaluation only: built-in order names or explicit [ZS1, R1, OS1, ...] sequences.
    fixed_acquisition_orders: dict = field(default_factory=lambda: {
        "fixed": "interleaved",
        "fixed_evaluators_first": "evaluators_first",
        "fixed_zero_shots_first": "zero_shots_first",
    })
    bootstrap_samples: int = 1000
    print_every: int = 10
    # Runtime-only mirror of output_dir on the Hugging Face Hub: Kaggle keeps its
    # working disk only while a session lives, so uploads make a run resumable.
    hf_repo: Optional[str] = None      # Private dataset repo id such as "user/run"; None disables.
    hf_sync_minutes: float = 15.       # Upload cadence inside long loops; stage ends always upload.
    # Runtime-only versioned dataset store: what the three training phases build (compact records,
    # teacher labels, frozen predictor, Stage 3 labels) is uploaded once under dataset_id and
    # downloaded, never rebuilt, whenever the Hub already holds it for this configuration.
    dataset_id: Optional[str] = None   # Name of the dataset family, e.g. "adaptive-v9"; None disables.
    dataset_repo: Optional[str] = None # Private dataset repo that holds every dataset_id, e.g. "user/datasets".

    @property
    def n(self):
        return self.zero_shots + self.k

    @property
    def action_count(self):
        # Next zero-shot, add evaluator Rj, generate the one-shot candidate of source j.
        return 2 * self.k + 1

    @property
    def head_input_dim(self):
        # The frozen hidden vector plus the plain candidate and evaluator masks.
        return self.hidden_dim + self.n + self.k

    @property
    def input_dim(self):
        # The extra masks distinguish retrieval from evaluator activation and
        # a known similarity from a missing similarity.
        return self.n * (self.k + 3) + 7 * self.k + 3 * self.n * self.k + 4

    @property
    def full_cost(self):
        return acquisition_cost(self.n, self.k, self)

    @property
    def budget(self):
        return self.full_cost if self.max_cost is None else self.max_cost

    def validate(self):
        assert self.k >= 1 and self.zero_shots >= 1 and self.repeats >= 1
        assert self.hidden_dim >= 2 and self.batch_size >= 2 and self.teacher_folds >= 2
        assert self.cost_unit in {"solver_calls", "total_calls"}
        assert 0 < self.recognition_threshold < 1
        assert type(self.continue_after_max) is bool
        assert type(self.use_test_files_for_dev_and_audit) is bool
        if self.use_test_files_for_dev_and_audit:
            if not self.test_files:
                raise ValueError("test_files must be provided when using them for dev and audit.")
        else:
            assert len(self.split_fractions) == 3 and min(self.split_fractions) > 0
            assert abs(sum(self.split_fractions) - 1) < 1e-8
        assert 1 <= self.budget <= self.full_cost
        assert self.teacher_epochs > 0 and self.snapshot_epochs > 0 and self.patience > 0
        assert self.decision_epochs > 0 and self.decision_batch_size >= 2 and self.decision_lr > 0
        assert self.decision_finetune_epochs > 0 and self.decision_finetune_lr > 0
        assert self.decision_eval_batch_size >= 2
        assert self.dev_checks_per_epoch >= 1
        assert 0 <= self.teacher_complete_share < 1 and 0 <= self.snapshot_complete_share < 1
        assert self.print_every > 0
        assert self.hf_sync_minutes > 0
        if self.hf_repo is not None and not re.fullmatch(r"[\w.-]+/[\w.-]+", str(self.hf_repo)):
            raise ValueError("hf_repo must be a Hugging Face dataset repository id such as 'user/name'.")
        if self.dataset_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(self.dataset_id)):
            raise ValueError("dataset_id must be 1-64 letters, digits, '.', '_' or '-' and start with a "
                             "letter or digit, for example 'adaptive-v9'.")
        if self.dataset_repo is not None and not re.fullmatch(r"[\w.-]+/[\w.-]+", str(self.dataset_repo)):
            raise ValueError("dataset_repo must be a Hugging Face dataset repository id such as 'user/name'.")
        if (self.dataset_id is None) != (self.dataset_repo is None):
            raise ValueError("dataset_id and dataset_repo go together: the id names a dataset inside the repository.")
        if self.dataset_repo is not None and self.dataset_repo == self.hf_repo:
            raise ValueError("dataset_repo must differ from hf_repo: the run mirror would download the "
                             "datasets stored beside it.")
        fixed_acquisition_sequences(self)


# Outside the run contract and the Stage 3 fingerprint: runtime controls, the Hub
# mirror, the dataset store, and the input log paths. A log is identified by its content, and its
# filename stem is the benchmark named in each record uid.
NON_CONTRACT_SETTINGS = ("resume", "output_dir", "device", "cpu_threads", "print_every",
                         "fixed_acquisition_orders", "continue_after_max", "decision_eval_batch_size",
                         "hf_repo", "hf_sync_minutes", "train_file", "test_files",
                         "dataset_id", "dataset_repo")


@dataclass
class Record:
    uid: str
    group: str
    benchmark: str
    candidate_ids: list
    evaluator_ids: list
    similarity: np.ndarray
    baseline: np.ndarray
    ccs: np.ndarray
    labels: np.ndarray


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(tmp, path)


def atomic_torch(path, value):
    """Preserve completed artifacts and remove partial writes, including on ENOSPC."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        # A Python file handle exposes filesystem errors that the C++ path writer
        # can otherwise obscure behind an iostream/unexpected-position error.
        with tmp.open("wb") as handle:
            torch.save(value, handle)
        os.replace(tmp, path)
    except (OSError, RuntimeError) as exc:
        try:
            free = f"{shutil.disk_usage(path.parent).free / 1024**3:.3f} GiB free"
        except OSError:
            free = "free space unavailable"
        raise RuntimeError(
            f"Checkpoint write failed: {path} ({free} on the output filesystem). "
            "Check disk space, storage quota, and write access. Completed checkpoints "
            "are preserved; the partial temporary file is removed. Free space and "
            "resume with the same output_dir and configuration. "
            f"Original error: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path):
    # Only load checkpoints created by this workflow, never untrusted .pt files.
    return torch.load(path, map_location="cpu", weights_only=False)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def file_digest(path, chunk=8 * 1024**2):
    """SHA-256 of a file's bytes; identifies an input log independently of its path or mtime."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_group(text):
    text = re.sub(r"\s+", " ", str(text)).strip().casefold()
    if not text:
        raise ValueError("missing_target_text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_log(path):
    """Stream top-level arrays or {'queries': ...} caches; small-file fallback."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        first = ""
        while not first:
            ch = handle.read(1)
            if not ch:
                raise ValueError(f"Empty JSON: {path}")
            if not ch.isspace():
                first = ch
    try:
        import ijson
    except ImportError:
        if path.stat().st_size > 64 * 1024**2:
            raise ImportError("Install ijson before reading large logs: %pip install -q ijson")
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            yield from enumerate(data)
        elif isinstance(data, dict) and isinstance(data.get("queries"), dict):
            yield from data["queries"].items()
        else:
            raise ValueError(f"Unsupported JSON envelope: {path}")
        return
    with path.open("rb") as handle:
        # ijson expects bytes; support a UTF-8 BOM without text re-encoding.
        if handle.read(3) != b"\xef\xbb\xbf":
            handle.seek(0)
        if first == "[":
            yield from enumerate(ijson.items(handle, "item", use_float=True))
        elif first == "{":
            yield from ijson.kvitems(handle, "queries", use_float=True)
        else:
            raise ValueError(f"Unsupported JSON envelope: {path}")


def parse_record(item, fallback, benchmark, cfg):
    if not isinstance(item, dict):
        raise ValueError("not_an_object")
    state = item.get("layer1_base_execution_state", item)
    if not isinstance(state, dict):
        raise ValueError("missing_layer1_state")
    target = state.get("target_query_data", {})
    text = item.get("target_query_text") or target.get("query_text")
    if not text:
        raise ValueError("missing_target_text")
    idx = item.get("target_query_original_hard_list_idx", fallback)
    retrieved = state.get("retrieved_set", [])
    candidates = state.get("candidate_set", {})
    labels = state.get("ground_truth_labels", {})
    if len(retrieved) != cfg.k or not isinstance(candidates, dict):
        raise ValueError("pool_shape")
    # Retrieval actions expose sources in descending similarity order.
    try:
        retrieved = sorted(retrieved, key=lambda row: -float(row["similarity_score"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing_or_invalid_measurement") from None
    eids = [str(x.get("corpus_index", x.get("retrieval_index"))) for x in retrieved]
    if "None" in eids or len(set(eids)) != cfg.k:
        raise ValueError("evaluator_ids")
    candidates = {str(k): v for k, v in candidates.items()}
    labels = {str(k): v for k, v in labels.items()}
    zs, os_by_source = [], {}
    for cid, cand in candidates.items():
        source = str(cand.get("source_exemplar_idx"))
        if cid.startswith("zs_") != (source == "-1"):
            raise ValueError("inconsistent_candidate_source")
        if source == "-1":
            if not re.fullmatch(r"zs_\d+", cid):
                raise ValueError("zero_shot_id")
            zs.append(cid)
        elif source in eids and source not in os_by_source:
            os_by_source[source] = cid
        else:
            raise ValueError("duplicate_or_unknown_source")
    zs.sort(key=lambda s: int(s[3:]))
    if len(zs) != cfg.zero_shots or set(os_by_source) != set(eids):
        raise ValueError("candidate_counts")
    cids = zs + [os_by_source[e] for e in eids]
    y = []
    for cid in cids:
        cand, lab = candidates[cid], labels.get(cid, {})
        if cand.get("generation_status") != "SUCCESS" or not cand.get("candidate_text"):
            raise ValueError("candidate_generation_failed")
        if lab.get("evaluation_status") != "SUCCESS" or type(lab.get("is_correct")) is not bool:
            raise ValueError("unknown_or_failed_label")
        y.append(int(lab["is_correct"]))
    bases = {str(k): v for k, v in state.get("intrinsic_baselines", {}).items()}
    matrix = {str(k): v for k, v in state.get("cross_evaluation_matrix", {}).items()}
    try:
        sim = np.asarray([float(x["similarity_score"]) for x in retrieved], np.float32)
        base = np.asarray([float(bases[e]) for e in eids], np.float32)
        ccs = np.asarray([[float(matrix[c][e]) for e in eids] for c in cids], np.float32)
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing_or_invalid_measurement")
    if not all(np.isfinite(a).all() for a in (sim, base, ccs)):
        raise ValueError("nonfinite_measurement")
    if np.any(np.abs(sim) > 1.00001) or any(np.any((a < 0) | (a > 1)) for a in (base, ccs)):
        raise ValueError("measurement_range")
    if any(not np.allclose(a * cfg.repeats, np.round(a * cfg.repeats), atol=1e-4)
           for a in (base, ccs)):
        raise ValueError("repeat_count_incompatible")
    return Record(f"{benchmark}::{idx}", normalized_group(text), benchmark, cids, eids,
                  sim, base, ccs, np.asarray(y, np.float32))


def load_records(cfg):
    records, reports, seen_groups, seen_ids = [], {}, set(), set()
    paths = [cfg.train_file] + cfg.test_files
    if len({Path(p).stem for p in paths}) != len(paths):
        raise ValueError("Each data file needs a unique filename stem (benchmark identity).")
    for path in paths:
        bench = Path(path).stem
        counts = Counter()
        for index, item in iter_log(path):
            counts["total"] += 1
            try:
                record = parse_record(item, index, bench, cfg)
            except ValueError as exc:
                counts[str(exc)] += 1
                continue
            if record.uid in seen_ids:
                counts["duplicate_id"] += 1
                continue
            if record.group in seen_groups:
                counts["duplicate_or_cross_file_overlap"] += 1
                continue
            seen_ids.add(record.uid)
            seen_groups.add(record.group)
            records.append(record)
            counts["eligible"] += 1
            counts["positive_pool"] += int(record.labels.any())
        reports[bench] = dict(counts)
        if not counts["total"]:
            raise ValueError(f"No records found in {path}; expected list or queries object.")
    return records, reports


def split_records(records, cfg):
    names = [Path(p).stem for p in [cfg.train_file] + cfg.test_files]
    if len(set(names)) != len(names) or set(names) & {"supervised", "policy", "dev", "audit"}:
        raise ValueError("Data filenames need unique stems distinct from supervised/policy/dev/audit.")
    train = [i for i, r in enumerate(records) if r.benchmark == Path(cfg.train_file).stem]
    if len(train) < max(20, cfg.teacher_folds * 3):
        raise ValueError("Too few eligible training questions for teacher folds.")
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(train)
    external = {Path(path).stem: [i for i, r in enumerate(records)
                                 if r.benchmark == Path(path).stem]
                for path in cfg.test_files}
    if cfg.use_test_files_for_dev_and_audit:
        evaluation = [i for ids in external.values() for i in ids]
        if len(evaluation) < 2:
            raise ValueError("test_files must supply at least two eligible questions for dev and audit.")
        result = {"supervised": train, "dev": evaluation, "audit": evaluation.copy()}
    else:
        cuts = np.rint(np.cumsum(cfg.split_fractions)[:-1] * len(train)).astype(int)
        parts = np.split(np.asarray(train), cuts)
        if min(map(len, parts)) < 2:
            raise ValueError("Shared training, dev, and audit each need at least two question groups.")
        result = {name: p.tolist() for name, p in zip(("supervised", "dev", "audit"), parts)}
    if len(result["supervised"]) < cfg.teacher_folds * 3:
        raise ValueError("Too few shared training questions for the configured teacher folds.")
    # The historical policy role is an alias, never a separate training partition.
    result["policy"] = result["supervised"].copy()
    result.update(external)
    return result


def split_protocol(cfg, splits):
    overlap = len(set(splits["dev"]) & set(splits["audit"]))
    return {"training_roles": {"teacher": "supervised", "snapshot": "supervised",
                               "acquisition": "supervised (policy alias)"},
            "use_test_files_for_dev_and_audit": cfg.use_test_files_for_dev_and_audit,
            "dev_audit_overlap_questions": overlap,
            "audit_used_for_model_selection": bool(overlap),
            "external_files_used_for_model_selection": (
                list(cfg.test_files) if cfg.use_test_files_for_dev_and_audit else [])}


def data_digest(records):
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps([r.uid, r.group, r.candidate_ids, r.evaluator_ids]).encode())
        for a in (r.similarity, r.baseline, r.ccs, r.labels):
            h.update(a.tobytes())
    return h.hexdigest()


def acquisition_cost(n, k, cfg):
    return float(acquisition_call_counts(n, k, cfg)[cfg.cost_unit])


def acquisition_call_counts(n, k, cfg):
    probes = cfg.repeats * k * (n + 1)
    return {"generation_calls": n, "measurement_solver_calls": probes,
            "grading_calls": probes, "solver_calls": n + probes,
            "total_calls": n + 2 * probes}


# %% Teacher network, metrics, and cross-fitted labels
class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        self.bn1, self.bn2 = nn.BatchNorm1d(width), nn.BatchNorm1d(width)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout(self.relu(self.bn1(self.fc1(x))))
        return self.relu(x + self.bn2(self.fc2(y)))


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, cfg.hidden_dim),
                                         nn.BatchNorm1d(cfg.hidden_dim), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(cfg.hidden_dim, cfg.dropout)
                                      for _ in range(cfg.residual_blocks)])
        self.output_layer = nn.Linear(cfg.hidden_dim, output_dim)

    def encode(self, x):
        return self.blocks(self.input_layer(x))

    def forward(self, x):
        return self.output_layer(self.encode(x))


def retrieval_order(cfg):
    return list(range(cfg.zero_shots, cfg.n)) + list(range(cfg.zero_shots))


def ranked_indices(scores, order):
    return sorted(order, key=lambda i: -float(scores[i]))


def ap_score(order, labels):
    positives = float(np.sum(labels[order]))
    if not positives:
        return 0.0
    ys = labels[order]
    return float(np.sum(np.cumsum(ys) / np.arange(1, len(order) + 1) * ys) / positives)


def ranking_metrics(records, ids, scores, cfg, tie_order=None):
    if not ids:
        return None
    aps, top, positive = [], [], []
    for idx in ids:
        r = records[idx]
        order = ranked_indices(scores[idx], retrieval_order(cfg) if tie_order is None else tie_order)
        aps.append(ap_score(order, r.labels))
        top.append(float(r.labels[order[0]]))
        positive.append(bool(r.labels.any()))
    pos = np.asarray(positive)
    return {"n": len(ids), "ap": float(np.mean(aps)), "top1": float(np.mean(top)),
            "coverage": float(pos.mean()),
            "conditional_ap": float(np.mean(np.asarray(aps)[pos])) if pos.any() else None}


def heuristic_scores(r, cfg, mode, mask, strategy):
    u = r.ccs.copy() if mode == "Absolute" else np.maximum(0, r.ccs - r.baseline)
    self_mask = np.zeros_like(u)
    for j in range(cfg.k):
        self_mask[cfg.zero_shots + j, j] = 1
    if mask == "Self":
        u *= self_mask
    elif mask == "Others":
        u *= 1 - self_mask
    take = u.sum(axis=1)
    make = np.r_[np.zeros(cfg.zero_shots), u.sum(axis=0)]
    return {"ScoreTake": take, "ScoreMake": make, "Holistic": take + make}[strategy]


def predict_array(model, x, device, batch=1024, hidden=False):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            t = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device)
            values.append((model.encode(t) if hidden else model(t)).cpu().numpy())
    return np.concatenate(values)


def teacher_loss(logits, targets, cfg):
    """Correctness of present candidates only; absent candidates carry no target."""
    present = targets[:, cfg.n:]
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets[:, :cfg.n], reduction="none")
    return ((raw * present).sum(1) / present.sum(1).clamp_min(1)).mean()


def lower_loss(new, old):
    """A development loss is better only when it falls by more than float noise."""
    return new < old - 1e-8


def train_teacher(records, train_ids, val_ids, cfg, device, seed, name):
    """Train on every presence state and zero-shot order; keep the checkpoint with the lowest
    development correctness loss on complete states. AP is a report metric, never a training signal."""
    seed_everything(seed)
    full = State((1 << cfg.n)-1, (1 << cfg.k)-1)
    # The ranking passed to later stages is read from complete states, so selection is too.
    vx = np.stack([observation(records[i], full, cfg) for i in val_ids])
    # Every candidate is present in a complete state, so each carries a correctness target.
    vy = torch.as_tensor(np.stack([np.r_[records[i].labels, np.ones(cfg.n)] for i in val_ids]),
                         dtype=torch.float32)
    # The complete view is the one later stages read, so it gets its own share of training.
    views = SnapshotViews(records, train_ids, cfg, device, complete_share=cfg.teacher_complete_share)
    model = ResNet(cfg.input_dim, cfg.n, cfg).to(device)

    def evaluate(model):
        # The training loss itself, on the complete-state development views.
        score = float(teacher_loss(torch.as_tensor(predict_array(model, vx, device)), vy, cfg))
        return score, f"development loss {score:.4f}"

    fit = train_on_views(model, views, cfg.teacher_epochs, seed, lambda logits, y: teacher_loss(logits, y, cfg),
                         evaluate, lower_loss, cfg, name)
    print(f"{name:<24} | epochs {fit['epochs']:6.2f} | best {fit['best_epoch']:6.2f} | "
          f"development loss {fit['best']:.4f}")
    return {"weights": fit["weights"], "best_epoch": fit["best_epoch"], "history": fit["history"],
            "train_ids": [records[i].uid for i in train_ids],
            "validation_ids": [records[i].uid for i in val_ids]}


def teacher_predict(checkpoint, records, ids, cfg, device):
    model = ResNet(cfg.input_dim, cfg.n, cfg).to(device)
    model.load_state_dict(checkpoint["weights"])
    full = State((1 << cfg.n)-1, (1 << cfg.k)-1)
    x = np.stack([observation(records[i], full, cfg) for i in ids])
    logits = predict_array(model, x, device)
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def teacher_labels(scores, truth, cfg):
    safe, maximum = np.zeros(cfg.n, np.float32), np.zeros(cfg.n, np.float32)
    order = ranked_indices(scores, retrieval_order(cfg))
    for i in order:
        if not truth[i]:
            break
        safe[i] = 1
    if safe.any():
        maximum[order[0]] = 1
    return safe, maximum


def completed_teacher(path, expected, cfg, name):
    """A saved fold/final teacher is reused only when its question roles match exactly."""
    if not (cfg.resume and path.exists()):
        return None
    ckpt = load_checkpoint(path)
    if any(ckpt.get(key) != value for key, value in expected.items()):
        return None
    print(f"{name:<24} | loaded completed checkpoint (best epoch {ckpt.get('best_epoch')})")
    return ckpt


def fit_teachers(records, splits, cfg, device, out, sync=None):
    sup = np.array(splits["supervised"])
    rng = np.random.RandomState(cfg.seed + 1)
    rng.shuffle(sup)
    folds = np.array_split(sup, cfg.teacher_folds)
    oof, provenance = {}, []
    for f, heldout in enumerate(folds):
        others = np.concatenate([p for j, p in enumerate(folds) if j != f])
        rng.shuffle(others)
        nv = max(1, int(.2 * len(others)))
        train_ids, val_ids = others[nv:].tolist(), others[:nv].tolist()
        if not train_ids:
            raise ValueError("Too few training records for nested teacher fold.")
        name, fold_path = f"Teacher fold {f+1}/{cfg.teacher_folds}", out / f"teacher_fold_{f+1}.pt"
        expected = {"train_ids": [records[i].uid for i in train_ids],
                    "validation_ids": [records[i].uid for i in val_ids],
                    "heldout_ids": [records[i].uid for i in heldout]}
        # Each fold completes as an atomic file, so an interrupted Stage 1 resumes per fold.
        ckpt = completed_teacher(fold_path, expected, cfg, name)
        if ckpt is None:
            ckpt = train_teacher(records, train_ids, val_ids, cfg, device, cfg.seed + f, name)
            ckpt["heldout_ids"] = expected["heldout_ids"]
            atomic_torch(fold_path, ckpt)
        predictions = teacher_predict(ckpt, records, heldout.tolist(), cfg, device)
        oof.update({int(i): p for i, p in zip(heldout, predictions)})
        provenance.append({k: ckpt[k] for k in ("train_ids", "validation_ids", "heldout_ids")})
        if sync is not None:
            sync.push(f"Stage 1: teacher fold {f+1}/{cfg.teacher_folds}")
    final_path = out / "teacher_final.pt"
    expected = {"train_ids": [records[i].uid for i in splits["supervised"]],
                "validation_ids": [records[i].uid for i in splits["dev"]]}
    final = completed_teacher(final_path, expected, cfg, "Final teacher")
    if final is None:
        final = train_teacher(records, splits["supervised"], splits["dev"], cfg, device,
                              cfg.seed + 100, "Final teacher")
        atomic_torch(final_path, final)
    scores = teacher_predict(final, records, list(range(len(records))), cfg, device)
    for idx, pred in oof.items():
        scores[idx] = pred
    safe, maximum = zip(*(teacher_labels(p, r.labels, cfg) for p, r in zip(scores, records)))
    return {"teacher": final, "scores": scores, "safe": np.stack(safe),
            "maximum": np.stack(maximum), "folds": provenance}


# %% Partial snapshots and supervised prediction
@dataclass(frozen=True)
class State:
    """Which candidates and evaluators are acquired.

    Any nonempty candidate subset combines with any evaluator subset: a one-shot
    candidate can be generated before, after, or without its own evaluator. Which
    acquisitions may come next is decided by raw_next_state.
    """
    candidate_mask: int = 1         # ZS1 initially exists at runtime.
    evaluator_mask: int = 0         # Arbitrary subset of ordered evaluators.


@dataclass(frozen=True)
class SnapshotState:
    """A supervised state; retrieval, evaluation and generation are independent.

    CCS bits use row-major candidate/evaluator order. Attempted but unobserved
    calls represent failures; unattempted calls have both bits clear.
    """
    candidates: int
    retrieved: int
    evaluators: int
    similarity_observed: int
    baseline_attempted: int
    baseline_observed: int
    ccs_attempted: int
    ccs_observed: int


def bit_array(bits, size):
    return np.asarray([(bits >> i) & 1 for i in range(size)], bool)


def bits_from_array(values):
    return sum(1 << i for i, value in enumerate(values) if value)


def complete_snapshot_state(candidates, retrieved, evaluators, cfg):
    """Construct the fully measured version of any presence state."""
    cells = sum(evaluators << (i * cfg.k) for i in range(cfg.n) if candidates & (1 << i))
    state = SnapshotState(candidates, retrieved, evaluators, retrieved,
                          evaluators, evaluators, cells, cells)
    validate_snapshot_state(state, cfg)
    return state


def validate_snapshot_state(state, cfg):
    """Enforce causal availability. A one-shot candidate needs no evaluator of its own."""
    if min(state.candidates, state.retrieved, state.evaluators,
           state.similarity_observed, state.baseline_attempted,
           state.baseline_observed, state.ccs_attempted, state.ccs_observed) < 0:
        raise ValueError("negative snapshot mask")
    if state.candidates == 0 or state.candidates >= 1 << cfg.n:
        raise ValueError("snapshot needs at least one valid candidate")
    if state.retrieved >= 1 << cfg.k or state.evaluators >= 1 << cfg.k:
        raise ValueError("retrieval/evaluator mask out of range")
    if state.evaluators & ~state.retrieved:
        raise ValueError("evaluator without retrieved source")
    if state.similarity_observed & ~state.retrieved:
        raise ValueError("similarity without retrieved source")
    if state.baseline_attempted & ~state.evaluators or state.baseline_observed & ~state.baseline_attempted:
        raise ValueError("invalid baseline masks")
    legal_cells = sum(state.evaluators << (i * cfg.k)
                      for i in range(cfg.n) if state.candidates & (1 << i))
    if state.ccs_attempted & ~legal_cells or state.ccs_observed & ~state.ccs_attempted:
        raise ValueError("invalid CCS masks")


def as_snapshot_state(state, cfg):
    if isinstance(state, SnapshotState):
        return state
    if not isinstance(state, State):
        raise TypeError("Expected State or SnapshotState")
    # Retrieval and evaluation coincide: a one-shot candidate whose evaluator is
    # absent shows its source slot but none of that source's measurements.
    return complete_snapshot_state(state.candidate_mask, state.evaluator_mask, state.evaluator_mask, cfg)


@lru_cache(maxsize=4)
def presence_states(zero_shots, k):
    """Every nonempty candidate subset with every evaluator subset.

    A one-shot candidate does not need its own evaluator, so these are all the
    states that acquisition, and hiding candidates or evaluators, can produce:
    (2**8 - 1) * 2**5 = 8,160 by default, including states without ZS1.
    """
    return tuple(State(candidates, evaluators)
                 for candidates in range(1, 1 << (zero_shots + k))
                 for evaluators in range(1 << k))


@lru_cache(maxsize=8)
def state_graph(zero_shots, k):
    """Budget-independent, monotone transitions shared by every question."""
    cfg = Config(zero_shots=zero_shots, k=k)
    states = presence_states(zero_shots, k)
    index = {state: j for j, state in enumerate(states)}
    transitions = tuple({action: index[nxt]
                         for action in range(cfg.action_count)
                         if (nxt := raw_next_state(state, action, cfg)) is not None}
                        for state in states)
    return states, transitions


@lru_cache(maxsize=8)
def decision_structure(zero_shots, k, repeats, cost_unit, max_cost):
    """Budget-filtered action/cost tables shared across questions and orders."""
    cfg = Config(zero_shots=zero_shots, k=k, repeats=repeats, cost_unit=cost_unit, max_cost=max_cost)
    states, graph = state_graph(zero_shots, k)
    costs = {state: state_cost(state, cfg) for state in states}
    transitions, valid = {}, {}
    for state, edges in zip(states, graph):
        transitions[state] = {action: states[index] for action, index in edges.items()
                              if costs[state] <= cfg.budget + 1e-6 and costs[states[index]] <= cfg.budget + 1e-6}
        mask = np.zeros(cfg.action_count, bool)
        mask[list(transitions[state])] = True
        mask.setflags(write=False)
        valid[state] = mask
    return states, transitions, valid, costs


@lru_cache(maxsize=8)
def budget_state_index(zero_shots, k, repeats, cost_unit, max_cost):
    """In-budget states in the order every labeling scenario enumerates them."""
    cfg = Config(zero_shots=zero_shots, k=k, repeats=repeats, cost_unit=cost_unit, max_cost=max_cost)
    all_states, _, _, costs = decision_structure(zero_shots, k, repeats, cost_unit, max_cost)
    states = tuple(state for state in all_states if costs[state] <= cfg.budget + 1e-6)
    return states, {state: position for position, state in enumerate(states)}


def reachable_states(cfg):
    """All states reachable from ZS1 under the application actions and budget."""
    seen, pending = {State()}, [State()]
    for state in pending:
        for action in np.flatnonzero(valid_actions(state, cfg)):
            nxt = advance(state, int(action), cfg)
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return tuple(pending)


def present_mask(state, cfg):
    bits = state.candidates if isinstance(state, SnapshotState) else state.candidate_mask
    return bit_array(bits, cfg.n)


def state_cost(state, cfg):
    if isinstance(state, SnapshotState):
        probes = cfg.repeats * (bin(state.baseline_attempted).count("1") + bin(state.ccs_attempted).count("1"))
        return float(bin(state.candidates).count("1") + probes * (2 if cfg.cost_unit == "total_calls" else 1))
    return acquisition_cost(bin(state.candidate_mask).count("1"),
                            bin(state.evaluator_mask).count("1"), cfg)


def used_sources(state, cfg):
    """Bit mask of the sources whose one-shot candidate or evaluator is already present."""
    return state.evaluator_mask | (state.candidate_mask >> cfg.zero_shots)


def opening_source(state, cfg):
    """The nearest source (sources are ordered by similarity) that nothing has used yet, or None."""
    used = used_sources(state, cfg)
    for source in range(cfg.k):
        if not used >> source & 1:
            return source
    return None


def raw_next_state(state, action, cfg):
    """The state after one acquisition, or None when the rules do not allow it.

    Action 0 adds the next zero-shot. Actions 1..k add the evaluator of source j and
    actions k+1..2k generate source j's one-shot candidate; neither needs the other.
    A source already in use (its other half is present) may be completed freely. An unused
    source may be opened only if it is the nearest one in similarity order, as either its
    evaluator or its one-shot candidate.
    """
    action, candidates, evaluators = int(action), state.candidate_mask, state.evaluator_mask
    if action == 0:
        missing = ~candidates & ((1 << cfg.zero_shots) - 1)
        return State(candidates | (missing & -missing), evaluators) if missing else None
    if not 1 <= action <= 2 * cfg.k:
        return None
    evaluator_action = action <= cfg.k
    source = action - 1 if evaluator_action else action - cfg.k - 1
    present = evaluators if evaluator_action else candidates >> cfg.zero_shots
    if present >> source & 1:
        return None
    if not (used_sources(state, cfg) >> source & 1 or source == opening_source(state, cfg)):
        return None
    if evaluator_action:
        return State(candidates, evaluators | 1 << source)
    return State(candidates | 1 << (cfg.zero_shots + source), evaluators)


def valid_actions(state, cfg):
    result = np.zeros(cfg.action_count, bool)
    if state_cost(state, cfg) > cfg.budget + 1e-6:
        return result
    for a in range(cfg.action_count):
        nxt = raw_next_state(state, a, cfg)
        result[a] = nxt is not None and state_cost(nxt, cfg) <= cfg.budget + 1e-6
    return result


def advance(state, action, cfg):
    action = int(action)
    if not 0 <= action < cfg.action_count or not valid_actions(state, cfg)[action]:
        raise ValueError(f"Unavailable action {action} at {state}")
    return raw_next_state(state, action, cfg)


def observation(record, state, cfg):
    """Only observed entries are indexed; no labels/teacher scores enter x."""
    state = as_snapshot_state(state, cfg)
    validate_snapshot_state(state, cfg)
    cm = present_mask(state, cfg)
    rm = bit_array(state.retrieved, cfg.k)
    em = bit_array(state.evaluators, cfg.k)
    sm = bit_array(state.similarity_observed, cfg.k)
    bm = bit_array(state.baseline_observed, cfg.k)
    ba = bit_array(state.baseline_attempted, cfg.k)
    ca = bit_array(state.ccs_attempted, cfg.n * cfg.k).reshape(cfg.n, cfg.k)
    co = bit_array(state.ccs_observed, cfg.n * cfg.k).reshape(cfg.n, cfg.k)
    sources = np.zeros((cfg.n, cfg.k + 1), np.float32)
    for i in np.flatnonzero(cm):
        sources[i, 0 if i < cfg.zero_shots else i - cfg.zero_shots + 1] = 1
    ordinal = np.zeros(cfg.n, np.float32)
    ordinal[cm] = (np.flatnonzero(cm) + 1) / cfg.n
    sim, base = np.zeros(cfg.k, np.float32), np.zeros(cfg.k, np.float32)
    sim[sm], base[bm] = record.similarity[sm], record.baseline[bm]
    ccs = np.zeros((cfg.n, cfg.k), np.float32)
    ccs[co] = record.ccs[co]
    spent = state_cost(state, cfg)
    context = [spent / cfg.full_cost, max(0., cfg.budget - spent) / cfg.full_cost,
                cm.mean(), em.mean()]
    x = np.concatenate([cm, sources.ravel(), ordinal, rm, em, sim, sm,
                        base, bm, ba, ccs.ravel(), co.ravel(), ca.ravel(), context]).astype(np.float32)
    if x.shape != (cfg.input_dim,) or not np.isfinite(x).all():
        raise ValueError("Invalid snapshot observation.")
    return x


@lru_cache(maxsize=8)
def state_observation_template(states, zero_shots, k, repeats, cost_unit, max_cost):
    """Cache state-only bytes/masks; records supply only observed measurements.

    Build through the scalar contract once, retaining its validation, float32
    rounding and budget features. The default 8,160-state catalog needs about
    7 MiB, rather than repeating this Python work for every question and
    historical order.
    """
    cfg = Config(zero_shots=zero_shots, k=k, repeats=repeats,
                 cost_unit=cost_unit, max_cost=max_cost)
    empty = Record("", "", "", [], [], np.zeros(k, np.float32),
                   np.zeros(k, np.float32), np.zeros((cfg.n, k), np.float32),
                   np.zeros(cfg.n, np.float32))
    template = np.stack([observation(empty, state, cfg) for state in states])
    similarity_start = cfg.n * (k + 3) + 2*k
    baseline_start = similarity_start + 2*k
    ccs_start = baseline_start + 3*k
    masks = (template[:, similarity_start+k:similarity_start+2*k].astype(bool),
             template[:, baseline_start+k:baseline_start+2*k].astype(bool),
             template[:, ccs_start+cfg.n*k:ccs_start+2*cfg.n*k].astype(bool))
    template.setflags(write=False)
    for mask in masks:
        mask.setflags(write=False)
    return template, masks, (similarity_start, baseline_start, ccs_start)


def observation_many(record, states, cfg):
    """Batch fully measured presence states; augmented snapshots keep scalar logic."""
    states = tuple(states)
    if not states or not all(isinstance(state, State) for state in states):
        return np.stack([observation(record, state, cfg) for state in states])
    template, masks, offsets = state_observation_template(
        states, cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    xs = template.copy()
    for source, mask, start in zip((record.similarity, record.baseline, record.ccs.ravel()),
                                   masks, offsets):
        block = xs[:, start:start+mask.shape[1]]
        # Index before assignment: NaNs in unobserved cells must stay invisible.
        block[mask] = np.broadcast_to(source, mask.shape)[mask]
    if not np.isfinite(xs).all():
        raise ValueError("Invalid snapshot observation.")
    return xs


def snapshot_target(record, safe, maximum, state, cfg):
    """Scalar reference for SnapshotViews targets."""
    cm = present_mask(state, cfg).astype(np.float32)
    return np.r_[record.labels, safe, maximum,
                 float(np.any(safe * cm)), float(np.any(maximum * cm)), cm].astype(np.float32)


SNAPSHOT_EVAL_BATCH = 16384


class SnapshotViews:
    """Every (question, zero-shot order, presence state) view, assembled per batch.

    Nothing is materialized or written. A shared state table (about 7 MiB for the
    default 8,160 states) holds the state-only observation bytes and masks; each
    question adds its measurements, labels, and SAFE/MAX under each zero-shot order
    (under 1 KiB). Rows equal observation(...) and snapshot_target of the
    permuted record bit for bit. Views are indexed question-major, then
    zero-shot order, then state.
    """
    def __init__(self, records, ids, cfg, device, teacher=None, states=None, complete_share=0.):
        self.cfg, self.device, self.ids = cfg, torch.device(device), list(ids)
        states = presence_states(cfg.zero_shots, cfg.k) if states is None else tuple(states)
        full = State((1 << cfg.n) - 1, (1 << cfg.k) - 1)
        template, masks, (sim, base, ccs) = state_observation_template(
            states, cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
        orders = list(permutations(range(cfg.zero_shots)))
        slots = np.asarray([list(order) + list(range(cfg.zero_shots, cfg.n)) for order in orders])

        def put(values, shape):
            array = np.stack(values) if values else np.zeros((0, *shape))
            return torch.as_tensor(array.astype(np.float32), device=self.device)

        rows = [records[i] for i in self.ids]
        self.template = torch.tensor(template, device=self.device)
        self.masks = tuple(torch.tensor(mask, device=self.device) for mask in masks)
        self.columns = ((sim, cfg.k), (base, cfg.k), (ccs, cfg.n * cfg.k))
        self.values = (put([r.similarity for r in rows], (cfg.k,)),
                       put([r.baseline for r in rows], (cfg.k,)),
                       put([r.ccs[slots].reshape(len(orders), -1) for r in rows],
                           (len(orders), cfg.n * cfg.k)))
        self.labels = put([r.labels[slots] for r in rows], (len(orders), cfg.n))
        self.safe = self.maximum = None
        if teacher is not None:
            self.safe = put([teacher["safe"][i][slots] for i in self.ids], (len(orders), cfg.n))
            self.maximum = put([teacher["maximum"][i][slots] for i in self.ids], (len(orders), cfg.n))
        self.state_count, self.order_count = len(states), len(orders)
        self.per_question = self.state_count * self.order_count
        self.rows = len(self.ids) * self.per_question
        # Training repeats each complete view so complete views make up `complete_share`
        # of an epoch; every partial view is still visited exactly once.
        self.complete_state = states.index(full) if full in states else None
        self.complete_share = complete_share
        self.complete_copies = 1
        if complete_share:
            if self.complete_state is None:
                raise ValueError("complete_share needs the complete state among the views.")
            self.complete_copies = max(1, round(complete_share * (self.state_count - 1)
                                                / (1 - complete_share)))
        self.epoch_per_question = self.per_question + (self.complete_copies - 1) * self.order_count
        self.epoch_rows = len(self.ids) * self.epoch_per_question

    def batch(self, index):
        """Observations and targets for flat view indices on the device."""
        s = index % self.state_count
        o = index // self.state_count % self.order_count
        q = index // self.per_question
        x = self.template[s]
        for (start, width), mask, value in zip(self.columns, self.masks, self.values):
            block = value[q] if value.dim() == 2 else value[q, o]
            x[:, start:start+width] = torch.where(mask[s], block, x[:, start:start+width])
        cm, labels = x[:, :self.cfg.n], self.labels[q, o]
        if self.safe is None:
            return x, torch.cat([labels, cm], 1)
        safe, maximum = self.safe[q, o], self.maximum[q, o]
        return x, torch.cat([labels, safe, maximum,
                             (safe * cm).amax(1, keepdim=True).gt(0).float(),
                             (maximum * cm).amax(1, keepdim=True).gt(0).float(), cm], 1)

    def batches(self, batch_size, rng):
        """One epoch: every view once plus the complete-view repeats.

        Every minibatch takes an even slice of every question's rows. Batches drawn
        from a few questions at a time would train many consecutive steps on the same
        questions, which the model memorizes, raising its loss on the next ones. Each
        question walks one shared random order of its rows from its own random start,
        so the epoch holds a single permutation of one question's rows, not of all rows.
        """
        per, count = self.epoch_per_question, len(self.ids)
        # Rows per question per batch: about batch_size rows in all, at least one per
        # question (so more questions than batch_size rows give larger batches).
        steps = max(1, -(-per // max(1, batch_size // max(1, count))))
        if count == 1:
            steps = max(1, min(steps, per // 2))  # BatchNorm cannot train on a single row.
        order = torch.as_tensor(rng.permutation(per), device=self.device)
        start = torch.as_tensor(rng.randint(0, per, count), device=self.device)
        base = torch.arange(count, device=self.device) * self.per_question
        cuts = [step * per // steps for step in range(steps + 1)]
        for lo, hi in zip(cuts, cuts[1:]):
            local = order[(torch.arange(lo, hi, device=self.device)[None] + start[:, None]) % per]
            if per > self.per_question:  # Slots past the views repeat a complete view per order.
                repeat = local >= self.per_question
                local = torch.where(repeat, (local - self.per_question) % self.order_count
                                    * self.state_count + self.complete_state, local)
            yield self.batch((base[:, None] + local).reshape(-1))

    def ordered(self, batch_size=SNAPSHOT_EVAL_BATCH):
        """Every view in index order, with each row's question position."""
        for lo in range(0, self.rows, batch_size):
            index = torch.arange(lo, min(self.rows, lo + batch_size), device=self.device)
            yield (index // self.per_question, *self.batch(index))


def train_on_views(model, views, epochs, seed, loss_of, evaluate, better, cfg, name):
    """Every epoch visits every view once, plus the views' complete-state repeats;
    development checks are spread through it.

    `evaluate(model)` returns the development loss (lower is better) and its description;
    `better(new, old)` says whether a loss improves on the best so far. The checks only
    choose the checkpoint: training stops at the end of the first epoch that finishes
    `patience` or more whole epochs after the best check.
    """
    if views.rows < 2:
        raise ValueError(f"{name}: too few training views.")
    # One fused update kernel on CUDA: small-model steps are otherwise launch-bound.
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                           fused=views.device.type == "cuda")
    best, best_text, weights, best_epoch, history = None, "", None, 0., []
    checks, position, started = cfg.dev_checks_per_epoch, 0., time.monotonic()
    complete = len(views.ids) * views.order_count * views.complete_copies
    print(f"{name}: {len(views.ids)} questions x {views.order_count} zero-shot orders x "
          f"{views.state_count} presence states = {views.rows:,} views; each complete view "
          f"{views.complete_copies:,}x, so {views.epoch_rows:,} rows per epoch, "
          f"{complete / views.epoch_rows:.1%} complete; {checks} development checks per epoch; "
          f"patience counts epochs.", flush=True)
    for epoch in range(epochs):
        done = seen = check = 0
        total = torch.zeros((), device=views.device)
        epoch_total = torch.zeros((), device=views.device)
        model.train()
        for bx, by in views.batches(cfg.batch_size, np.random.RandomState(seed + epoch)):
            opt.zero_grad(set_to_none=True)
            loss = loss_of(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.detach() * len(bx)
            done, seen = done + len(bx), seen + len(bx)
            due = done * checks // views.epoch_rows
            if due <= check:
                continue
            check, position = due, epoch + done / views.epoch_rows
            score, text = evaluate(model)
            model.train()
            history.append({"epoch": position, "train_loss": float(total) / seen, "dev_loss": score})
            epoch_total += total
            total.zero_()
            seen = 0
            if best is None or better(score, best):
                best, best_text, weights, best_epoch = score, text, cpu_state(model), position
        # Training loss averages this epoch's rows up to its last check.
        waited = position - best_epoch
        print(f"{name} | epoch {position:6.3f} | train loss {float(epoch_total) / max(1, done - seen):.4f} | "
              f"latest {text} | best {best_text} at epoch {best_epoch:.3f} | "
              f"epochs without improvement {waited:.2f}/{cfg.patience} | "
              f"elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True)
        if waited >= cfg.patience:
            break
    return {"weights": weights, "best": best, "best_epoch": best_epoch,
            "epochs": position, "history": history}


def snapshot_row_losses(logits, targets, cfg):
    """The training loss of each view: correctness plus auxiliary SAFE/MAX terms (absent candidates masked)."""
    n = cfg.n
    mask = targets[:, 3*n + 2:]
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets[:, :3*n+2], reduction="none")
    losses = [(raw[:, b*n:(b+1)*n] * mask).sum(1) / mask.sum(1).clamp_min(1) for b in range(3)]
    return losses[0] + cfg.auxiliary_weight * (losses[1] + losses[2] + raw[:, 3*n:].sum(1))


def snapshot_loss(logits, targets, cfg):
    return snapshot_row_losses(logits, targets, cfg).mean()


# `loss` is the sum of snapshot_row_losses, which Stage 2 minimizes and selects its checkpoint on.
SNAPSHOT_SUM_FIELDS = ("views", "top1", "brier", "brier_cells",
                       "safe_predicted", "safe_true_positive", "safe_true",
                       "max_predicted", "max_true_positive", "max_true", "loss")


def snapshot_sums(model, views, temperatures, cfg):
    """Per-question metric sums over every view, streamed; logits are never kept."""
    n = cfg.n
    temperatures = torch.as_tensor(temperatures, dtype=torch.float32, device=views.device)
    sums = torch.zeros((len(views.ids), len(SNAPSHOT_SUM_FIELDS)), dtype=torch.float64,
                       device=views.device)
    model.eval()
    with torch.no_grad():
        for q, x, y in views.ordered():
            logits = model(x) / temperatures
            p = torch.sigmoid(logits)
            mask = y[:, 3*n+2:].bool()
            chosen = torch.where(mask, logits[:, :n], torch.full_like(logits[:, :n], -math.inf)).argmax(1)
            error = torch.where(mask, (p[:, :n] - y[:, :n]) ** 2, torch.zeros_like(p[:, :n]))
            columns = [torch.ones_like(p[:, 0]), y[:, :n].gather(1, chosen[:, None])[:, 0],
                       error.sum(1), mask.sum(1)]
            for col in (3*n, 3*n+1):
                predicted, true = p[:, col] >= cfg.recognition_threshold, y[:, col] > .5
                columns += [predicted, predicted & true, true]
            columns.append(snapshot_row_losses(logits, y, cfg))
            sums.index_add_(0, q, torch.stack([c.double() for c in columns], 1))
    return sums.cpu().numpy()


def summarize_snapshot_sums(sums):
    total = np.asarray(sums, np.float64).reshape(-1, len(SNAPSHOT_SUM_FIELDS)).sum(0)
    value = dict(zip(SNAPSHOT_SUM_FIELDS, total))
    ratio = lambda a, b: float(a / b) if b else None
    return {"snapshots": int(value["views"]), "loss": ratio(value["loss"], value["views"]),
            "top1": ratio(value["top1"], value["views"]),
            "brier": ratio(value["brier"], value["brier_cells"]),
            **{f"{name}_{metric}": ratio(value[f"{name}_true_positive"], value[f"{name}_{base}"])
               for name in ("safe", "max")
               for metric, base in (("precision", "predicted"), ("recall", "true"))}}


def temperature_vector(temperatures, cfg):
    return np.r_[np.repeat(temperatures[:3], cfg.n), temperatures[3:]].astype(np.float32)


TEMPERATURE_GRID = np.geomspace(.25, 4., 41)


def fit_temperatures(model, views, cfg):
    """Development-only temperature per output block, from streamed NLL sums."""
    temps = np.ones(5, np.float32)
    if not cfg.calibrate:
        return temps
    n = cfg.n
    grid = torch.as_tensor(TEMPERATURE_GRID, device=views.device)
    nll = torch.zeros((5, len(grid)), dtype=torch.float64, device=views.device)
    count = torch.zeros(5, dtype=torch.float64, device=views.device)
    model.eval()
    with torch.no_grad():
        for _, x, y in views.ordered():
            logits, y = model(x).double(), y.double()
            mask = y[:, 3*n+2:].bool()
            for block in range(5):
                if block < 3:
                    z, t = logits[:, block*n:(block+1)*n][mask], y[:, block*n:(block+1)*n][mask]
                else:
                    z, t = logits[:, 3*n+block-3], y[:, 3*n+block-3]
                scaled = z[None] / grid[:, None]
                nll[block] += (torch.logaddexp(torch.zeros_like(scaled), scaled) - t * scaled).sum(1)
                count[block] += len(z)
    return TEMPERATURE_GRID[(nll / count[:, None]).argmin(1).cpu().numpy()].astype(np.float32)


def fit_snapshot(records, splits, teacher, cfg, device):
    seed_everything(cfg.seed + 200)
    views = SnapshotViews(records, splits["supervised"], cfg, device, teacher,
                          complete_share=cfg.snapshot_complete_share)
    # Selection and calibration use what the runtime visits: every reachable state.
    dev = SnapshotViews(records, splits["dev"], cfg, device, teacher, reachable_states(cfg))
    model = ResNet(cfg.input_dim, 3*cfg.n+2, cfg).to(device)
    print(f"Snapshot development: {len(dev.ids)} questions x {dev.order_count} zero-shot orders x "
          f"{dev.state_count} reachable states = {dev.rows:,} views.", flush=True)
    uncalibrated = temperature_vector(np.ones(5, np.float32), cfg)

    def evaluate(model):
        # The training loss over all 3n+2 outputs, on the reachable development states. Top-1 and
        # Brier are shown for information; they do not choose the checkpoint.
        metrics = summarize_snapshot_sums(snapshot_sums(model, dev, uncalibrated, cfg))
        return (metrics["loss"],
                f"development loss {metrics['loss']:.4f} (Top-1 {metrics['top1']:.1%}, "
                f"Brier {metrics['brier']:.4f})")

    fit = train_on_views(model, views, cfg.snapshot_epochs, cfg.seed + 201,
                         lambda logits, y: snapshot_loss(logits, y, cfg), evaluate,
                         lower_loss, cfg, "Snapshot")
    model.load_state_dict(fit["weights"])
    temps = fit_temperatures(model, dev, cfg)
    dev_summary = summarize_snapshot_sums(snapshot_sums(model, dev, temperature_vector(temps, cfg), cfg))
    print(f"Snapshot selected at epoch {fit['best_epoch']:.2f}; development loss {fit['best']:.4f} "
          f"({dev_summary['loss']:.4f} calibrated), Top-1 {dev_summary['top1']:.1%}.")
    return {"weights": fit["weights"], "temperatures": temps, "history": fit["history"],
            "best_epoch": fit["best_epoch"], "input_dim": cfg.input_dim,
            "presence_states": views.state_count, "zero_shot_orders": views.order_count,
            "views_per_epoch": views.rows, "rows_per_epoch": views.epoch_rows,
            "complete_copies": views.complete_copies, "dev_summary": dev_summary}


def head_columns(cfg):
    """Observation columns holding the acquired-candidate and active-evaluator masks."""
    evaluators = cfg.n * (cfg.k + 3) + cfg.k  # After candidates, sources, ordinals, and retrieval.
    return np.r_[0:cfg.n, evaluators:evaluators + cfg.k]


class FrozenPredictor:
    def __init__(self, checkpoint, cfg, device):
        self.cfg, self.device = cfg, device
        self.model = ResNet(cfg.input_dim, 3*cfg.n+2, cfg).to(device)
        self.model.load_state_dict(checkpoint["weights"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.temperatures = temperature_vector(checkpoint["temperatures"], cfg)
        self.columns = head_columns(cfg)

    def predict(self, record, state, cfg_override=None):
        h, p = self.predict_many(record, [state], cfg_override)
        return h[0], p[0]

    def encode(self, xs):
        """Frozen hidden vectors on this predictor's device; one copy, few launches."""
        self.model.eval()
        xs = torch.as_tensor(xs, device=self.device)
        with torch.no_grad():
            parts = [self.model.encode(xs[start:start+8192]) for start in range(0, len(xs), 8192)]
        return torch.cat(parts) if parts else torch.empty((0, self.cfg.hidden_dim), device=self.device)

    def features(self, xs):
        """The decision head's input: the frozen hidden vector plus the acquired-item masks.

        The masks are plain facts of the visible state, already inputs of the frozen network.
        An action such as "add R3" means opening source 3 in one state and completing it in
        another, so the head reads which items are acquired directly rather than relying on
        the hidden vector, which was trained to predict correctness, to keep them.
        """
        xs = torch.as_tensor(xs, device=self.device)
        columns = torch.as_tensor(self.columns, device=self.device)
        return torch.cat([self.encode(xs), xs[:, columns].to(torch.float32)], 1)

    def predict_many(self, record, states, cfg_override=None, observations=None):
        """Batch frozen inference; only visible measurements enter the network.

        Returns the head's input features (see `features`) and the output probabilities.
        """
        cfg = cfg_override or self.cfg
        states = tuple(states)
        xs = observations if observations is not None else observation_many(record, states, cfg)
        hs, ps = [], []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(xs), 1024):
                chunk = xs[start:start+1024]
                h = self.model.encode(torch.as_tensor(chunk, device=self.device))
                logits = self.model.output_layer(h).cpu().numpy()
                hs.append(np.concatenate([h.cpu().numpy(), np.asarray(chunk, np.float32)[:, self.columns]], 1))
                ps.append(1 / (1 + np.exp(-np.clip(logits / self.temperatures, -40, 40))))
        features, probabilities = np.concatenate(hs), np.concatenate(ps)
        present = np.stack([present_mask(state, self.cfg) for state in states])
        candidate_probabilities = probabilities[:, :3*self.cfg.n].reshape(-1, 3, self.cfg.n)
        candidate_probabilities[~np.broadcast_to(present[:, None, :], candidate_probabilities.shape)] = 0
        return features, probabilities


def selected_candidate(probabilities, state, cfg):
    return int(np.where(present_mask(state, cfg), probabilities[:cfg.n], -np.inf).argmax())


def max_recognition_signals(probabilities, state, cfg):
    i, n = selected_candidate(probabilities, state, cfg), cfg.n
    return {name: float(probabilities[j]) for name, j in (
        ("global_MAX", 3*n+1), ("global_SAFE", 3*n),
        ("selected_MAX", 2*n+i), ("selected_SAFE", n+i))}


def max_recognized(probabilities, state, cfg):
    return all(value >= cfg.recognition_threshold
               for value in max_recognition_signals(probabilities, state, cfg).values())


def stopping_reason(probabilities, state, cfg):
    if max_recognized(probabilities, state, cfg):
        return "predicted_max"
    if not valid_actions(state, cfg).any():
        full = state.candidate_mask == (1 << cfg.n)-1 and state.evaluator_mask == (1 << cfg.k)-1
        return "pool_exhaustion" if full else "budget"
    return None

def greedy_action(head, features, valid, device):
    if not valid.any():
        raise ValueError("No valid acquisition action.")
    with torch.no_grad():
        logits = head(torch.as_tensor(features[None], device=device))[0].cpu().numpy()
    return int(np.where(valid, logits, -np.inf).argmax())


def action_names(cfg):
    return (["add_zero_shot"] + [f"add_evaluator_R{j+1}" for j in range(cfg.k)] +
            [f"generate_one_shot_from_R{j+1}" for j in range(cfg.k)])


def evaluator_names(mask, cfg):
    return [f"R{j+1}" for j in range(cfg.k) if mask & (1 << j)]


# %% Supervised acquisition labels and frozen-encoder decision head
def permute_zero_shots(record, scores, safe, maximum, order, cfg):
    slots = list(order) + list(range(cfg.zero_shots, cfg.n))
    return (Record(record.uid, record.group, record.benchmark,
                   [record.candidate_ids[i] for i in slots], record.evaluator_ids,
                   record.similarity, record.baseline, record.ccs[slots], record.labels[slots]),
            scores[slots], safe[slots], maximum[slots])

def permuted_reference_order(scores, zero_order, cfg):
    """Move the original ranking intact, including its deterministic score ties."""
    slots = list(zero_order) + list(range(cfg.zero_shots, cfg.n))
    inverse = {original: current for current, original in enumerate(slots)}
    return [inverse[i] for i in ranked_indices(scores, retrieval_order(cfg))]


def goal_condition(rank, order, safe, maximum, probabilities, full_probabilities, state, cfg):
    """The recorded teacher defines a goal, while only the frozen model supplies lights."""
    i = order[rank]
    if not state.candidate_mask & (1 << i):
        return False
    n, threshold = cfg.n, cfg.recognition_threshold
    if rank == 0 and maximum.any():
        return (selected_candidate(probabilities, state, cfg) == i and
                all(probabilities[j] >= threshold for j in
                    (3*n, 3*n+1, n+i, 2*n+i)))
    compare = cfg.later_rank_filter
    if compare and rank < n - 1:
        if not probabilities[i] > max(full_probabilities[j] for j in order[rank+1:]):
            return False
    if rank > 0 and safe[i] and probabilities[n+i] < threshold:
        return False
    return True


def decision_goals(order, safe, maximum, probabilities, full_probabilities, states, cfg):
    """Vectorized form of goal_condition, retaining its presence and tie rules."""
    masks = np.fromiter((state.candidate_mask for state in states), np.int64, len(states))
    present = ((masks[:, None] >> np.arange(cfg.n)) & 1).astype(bool)
    goal = np.empty((len(states), cfg.n), bool)
    n, threshold = cfg.n, cfg.recognition_threshold
    selected = np.where(present, probabilities[:, :n], -np.inf).argmax(axis=1)
    for rank, i in enumerate(order):
        met = present[:, i].copy()
        if rank == 0 and maximum.any():
            met &= (probabilities[:, [3*n, 3*n+1, n+i, 2*n+i]] >= threshold).all(axis=1)
            met &= selected == i
        else:
            if cfg.later_rank_filter and rank < n - 1:
                met &= probabilities[:, i] > max(full_probabilities[j] for j in order[rank+1:])
            if rank > 0 and safe[i]:
                met &= probabilities[:, n+i] >= threshold
        goal[:, rank] = met
    return goal


def visible_state_key(record, state, cfg):
    # Candidate IDs and hidden order are audit information, not model features.
    observed = observation(record, state, cfg)
    return (state.candidate_mask, state.evaluator_mask, hashlib.sha256(observed.tobytes()).hexdigest())


# Case dispositions in precedence order; a plan stores each as an int8 code.
DECISION_STATUSES = ("OUT_OF_BUDGET", "COMPLETE", "EXHAUSTED", "BUDGET", "ACTION", "UNREACHABLE")
NO_CHOICE = 255


class DecisionGraph(NamedTuple):
    states: tuple         # In-budget presence states, in label order.
    succ: np.ndarray      # [states, actions] successor index; -1 where the action is not allowed or affordable.
    delta: np.ndarray     # [states, actions] calls the action adds.
    legal: np.ndarray     # [states, actions] succ >= 0.
    levels: tuple         # Index arrays of the states by number of acquired items, most items first.
    full: int             # Index of the complete state, or -1 when it exceeds the budget.


@lru_cache(maxsize=8)
def decision_graph(zero_shots, k, repeats, cost_unit, max_cost):
    """The in-budget transitions as arrays, shared by every question and zero-shot order.

    Every action adds exactly one item, so a backward pass over `levels` sees all of a
    state's successors before the state itself.
    """
    states, index = budget_state_index(zero_shots, k, repeats, cost_unit, max_cost)
    _, transitions, _, costs = decision_structure(zero_shots, k, repeats, cost_unit, max_cost)
    succ = np.full((len(states), 2 * k + 1), -1, np.int64)
    delta = np.zeros(succ.shape)
    for position, state in enumerate(states):
        for action, nxt in transitions[state].items():
            succ[position, action] = index[nxt]
            delta[position, action] = costs[nxt] - costs[state]
    sizes = np.asarray([bin(state.candidate_mask).count("1") + bin(state.evaluator_mask).count("1")
                        for state in states])
    levels = tuple(np.flatnonzero(sizes == size) for size in sorted(set(sizes.tolist()), reverse=True))
    for array in (succ, delta, *levels):
        array.setflags(write=False)
    legal = succ >= 0
    legal.setflags(write=False)
    return DecisionGraph(states, succ, delta, legal, levels,
                         index.get(State((1 << (zero_shots + k)) - 1, (1 << k) - 1), -1))


def decision_scenarios(record, scores, safe, maximum, predictor, cfg):
    """One question's frozen evidence at every in-budget state under every zero-shot order.

    Returns the head features [orders, states, features], the goals that hold
    [orders, states, goals], and a digest of each visible observation
    [orders, states, 4] (byte-identical observations are the same head input).
    """
    graph = decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    full_cfg = copy.deepcopy(cfg)
    full_cfg.max_cost = None
    features, goals, digests = [], [], []
    for zero_order in permutations(range(cfg.zero_shots)):
        shuffled, _, shuffled_safe, shuffled_maximum = permute_zero_shots(
            record, scores, safe, maximum, zero_order, cfg)
        # Reuse exactly the same observation bytes for inference and visible grouping.
        observations = observation_many(shuffled, graph.states, cfg)
        if isinstance(predictor, FrozenPredictor):
            hidden, probabilities = predictor.predict_many(shuffled, graph.states, observations=observations)
        else:
            hidden, probabilities = predictor.predict_many(shuffled, graph.states)
        _, full_probability = predictor.predict(shuffled, State((1 << cfg.n)-1, (1 << cfg.k)-1), full_cfg)
        goals.append(decision_goals(permuted_reference_order(scores, zero_order, cfg), shuffled_safe,
                                    shuffled_maximum, probabilities, full_probability, graph.states, cfg))
        features.append(np.asarray(hidden, np.float32))
        digests.append(b"".join(hashlib.sha256(x.tobytes()).digest() for x in observations))
    return {"graph": graph, "features": np.stack(features), "goal": np.stack(goals),
            "digest": np.frombuffer(b"".join(digests), np.uint64).reshape(len(goals), len(graph.states), 4)}


def goal_distances(graph, prefix):
    """The fewest calls from every state to a state where goals 0..r all hold, for each rank r.

    `prefix` is [states, goals]. A route ends the first time its destination holds, so a
    state that already holds the prefix is at distance 0 and an unreachable one is infinite.
    Cost depends only on how many candidates and evaluators are present, so every order of
    adding the same items costs the same; the search is over which items to add.
    """
    distance = np.full((prefix.shape[1], len(graph.states)), np.inf)
    for idx in graph.levels:
        onward = graph.succ[idx]
        legal = onward >= 0
        total = graph.delta[idx][None] + distance[:, np.where(legal, onward, 0)]
        best = np.where(legal[None], total, np.inf).min(axis=2)
        distance[:, idx] = np.where(prefix[idx].T, 0., best)
    return distance


def first_steps(graph, distance, achieved):
    """[states, actions]: the allowed first steps that lie on a cheapest route to the next goal.

    The next goal of a state is its first unfinished rank. A state with every goal done, or
    with no route to its next goal, has no cheapest first step.
    """
    goals = distance.shape[0]
    rank = np.minimum(achieved, goals - 1)
    own = distance[rank, np.arange(len(graph.states))]
    onward = distance[rank[:, None], np.where(graph.legal, graph.succ, 0)]
    return (graph.legal & (achieved < goals)[:, None] & np.isfinite(own)[:, None] &
            np.isclose(graph.delta + onward, own[:, None], rtol=0, atol=1e-9))


def visible_leaders(digest):
    """[orders, states]: the first zero-shot order whose observation of the state is byte-identical."""
    leader = np.tile(np.arange(len(digest))[:, None], (1, digest.shape[1]))
    for order in range(1, len(digest)):
        same = (digest[:order] == digest[order]).all(axis=2)
        leader[order] = np.where(same.any(axis=0), same.argmax(axis=0), order)
    return leader


def question_plan(record, scores, safe, maximum, predictor, cfg):
    """Everything both Stage 3 phases derive from one question, over all of its state/order cases.

    A case is a state under one zero-shot order. Its next goal is the first rank whose goals
    do not all hold. ACTION cases have a cheapest route to it and so a set of cheapest first
    steps. Cases that look identical to the head (byte-identical observations) form one
    visible group; the group's informative cases give it its labels, and a group none of
    whose cases is informative (every goal done, or no route to the next goal) gets no
    preference, i.e. every allowed move. Cases without an allowed move get no row.
    """
    scenario = decision_scenarios(record, scores, safe, maximum, predictor, cfg)
    graph, goal, digest = scenario["graph"], scenario["goal"], scenario["digest"]
    orders, states, goals = goal.shape
    prefix = np.logical_and.accumulate(goal, axis=2)
    achieved = prefix.sum(2)
    distance = np.stack([goal_distances(graph, prefix[order]) for order in range(orders)])
    options = np.stack([first_steps(graph, distance[order], achieved[order]) for order in range(orders)])
    finite = np.isfinite(distance[np.arange(orders)[:, None], np.minimum(achieved, goals - 1),
                                  np.arange(states)[None]])
    has_move = graph.legal.any(1)
    status = np.full((orders, states), DECISION_STATUSES.index("ACTION"), np.int8)
    status[~finite] = DECISION_STATUSES.index("UNREACHABLE")
    status[:, ~has_move] = DECISION_STATUSES.index("BUDGET")
    if graph.full >= 0:
        status[:, graph.full] = DECISION_STATUSES.index("EXHAUSTED")
    status[achieved == goals] = DECISION_STATUSES.index("COMPLETE")
    informative = status == DECISION_STATUSES.index("ACTION")
    leader = visible_leaders(digest)
    # Identical observations must give identical frozen features.
    features = scenario["features"]
    for order in range(1, orders):
        shared = np.flatnonzero(leader[order] != order)
        if len(shared) and not np.allclose(features[order, shared], features[leader[order, shared], shared],
                                           atol=1e-6):
            raise ValueError("One visible state has inconsistent frozen features.")
    member = leader[None] == np.arange(orders)[:, None, None]
    group_informative = np.take_along_axis((member & informative[None]).any(1), leader, axis=0)
    rows = informative | (~group_informative & has_move[None])
    targets = np.where(informative[:, :, None], options, graph.legal[None]) & rows[:, :, None]
    if (targets & ~graph.legal[None]).any() or not targets[rows].any(1).all() or targets[~rows].any():
        raise ValueError("Decision label action mapping is incomplete.")
    if (rows & ~has_move[None]).any() or (informative & ~rows).any():
        raise ValueError("Decision label action mapping is incomplete.")
    return {"graph": graph, "features": features, "digest": digest, "prefix": prefix, "achieved": achieved,
            "status": status, "informative": informative, "options": options, "leader": leader,
            "rows": rows, "targets": targets}


def plan_coverage(plan, cfg):
    """One question's case dispositions and row counts, including the cases over budget."""
    graph, status, rows = plan["graph"], plan["status"], plan["rows"]
    orders = status.shape[0]
    all_states = len(presence_states(cfg.zero_shots, cfg.k))
    names = np.asarray(DECISION_STATUSES)[status]
    statuses = Counter(names.reshape(-1).tolist())
    over_budget = orders * (all_states - len(graph.states))
    if over_budget:
        statuses["OUT_OF_BUDGET"] += over_budget
    informative = plan["informative"]
    return {"questions": 1, "presence_states": all_states,
            "state_order_cases": status.size + over_budget, "action_rows": int(rows.sum()),
            "informative_rows": int(informative.sum()),
            "status_counts": dict(statuses),
            "trained_status_counts": dict(Counter(names[rows].tolist())),
            "option_counts": dict(Counter(plan["options"][informative].sum(1).tolist())),
            "actionable_by_target_rank": dict(Counter(str(rank + 1) for rank in plan["achieved"][informative].tolist()))}


def combine_coverage(parts):
    summed = {key: Counter() for key in ("status_counts", "trained_status_counts", "option_counts",
                                         "actionable_by_target_rank")}
    for part in parts:
        for key, counter in summed.items():
            counter.update({str(name): count for name, count in part.get(key, {}).items()})
    # Stage 3b's parts also count the rows it decided; Stage 3a's do not have those keys.
    totals = {key: sum(part.get(key, 0) for part in parts)
              for key in ("state_order_cases", "action_rows", "informative_rows", "changed_rows", "tied_rows",
                          "tied_rows_moved", "no_thread_reaches_rows")
              if any(key in part for part in parts)}
    return {"questions": len(parts),
            "presence_states_per_question": parts[0]["presence_states"] if parts else 0,
            **totals, **{key: dict(counter) for key, counter in summed.items()}}


def head_policy(head, features, graph, device):
    """The head's greedy allowed move (-1 where none) and softmax over the allowed moves."""
    orders, states, width = features.shape
    flat = torch.as_tensor(features.reshape(-1, width))
    device = next(head.parameters(), flat).device  # The head's own device wins over the argument.
    head.eval()
    with torch.no_grad():
        logits = torch.cat([head(flat[start:start+16384].to(device)).float().cpu()
                            for start in range(0, len(flat), 16384)]).numpy().astype(np.float64)
    masked = np.where(graph.legal[None], logits.reshape(orders, states, -1), -np.inf)
    move = np.where(graph.legal.any(1)[None], masked.argmax(-1), -1)
    top = masked.max(-1, keepdims=True)
    weights = np.exp(masked - np.where(np.isfinite(top), top, 0.))
    return move, weights / np.maximum(weights.sum(-1, keepdims=True), 1e-300)


def thread_outcomes(graph, move, prefix):
    """Follow `move` from every state until goals 0..r all hold, for every rank r.

    Returns [orders, goals, states] arrays: whether the thread reaches the goal, and the calls
    it spends (up to the goal, or until the head has no allowed move left).
    """
    orders, states, goals = prefix.shape
    reached, calls = np.zeros((orders, goals, states)), np.zeros((orders, goals, states))
    rows, ranks = np.arange(orders)[:, None, None], np.arange(goals)[None, :, None]
    for idx in graph.levels:
        choice = move[:, idx]
        moving = choice >= 0
        pick = np.where(moving, choice, 0)
        onward = np.where(moving, graph.succ[idx[None], pick], 0)
        step = graph.delta[idx[None], pick]
        hit = prefix[:, idx, :].transpose(0, 2, 1)
        keep = moving[:, None, :]
        reached[:, :, idx] = np.where(hit, 1., np.where(keep, reached[rows, ranks, onward[:, None, :]], 0.))
        calls[:, :, idx] = np.where(hit, 0., np.where(keep, step[:, None, :] + calls[rows, ranks, onward[:, None, :]], 0.))
    return reached, calls


def finetune_choices(plan, head, device, cfg):
    """One action per labeled case: the pretrained head's cheapest continuation among the options.

    Every first step on a cheapest route is tried. After it the pretrained head picks each
    next move (greedily, among allowed moves) until the same next goal holds. Cases the head
    cannot tell apart share one decision, judged on average over them: most threads that
    reach the goal first, then fewest calls, then the pretrained head's own probability. If
    no thread reaches the goal the options are separated by that probability alone. A group
    with no informative case keeps the head's favorite allowed move.
    """
    graph, achieved, leader, informative = plan["graph"], plan["achieved"], plan["leader"], plan["informative"]
    prefix, rows = plan["prefix"], plan["rows"]
    orders, states, goals = prefix.shape
    move, probability = head_policy(head, plan["features"], graph, device)
    reached, calls = thread_outcomes(graph, move, prefix)
    rank = np.minimum(achieved, goals - 1)
    onward = np.where(graph.legal, graph.succ, 0)
    index = (np.arange(orders)[:, None, None], rank[:, :, None], onward[None])
    first_reached, first_calls = reached[index], graph.delta[None] + calls[index]
    member = ((leader[None] == np.arange(orders)[:, None, None]) & informative[None]).astype(np.float64)
    share = np.maximum(member.sum(1), 1)[..., None]
    reach = np.einsum("los,osa->lsa", member, first_reached) / share
    cost = np.einsum("los,osa->lsa", member, first_calls) / share
    union = np.einsum("los,osa->lsa", member, plan["options"].astype(np.float64)) > 0
    candidates = union & graph.legal[None]
    best_reach = np.where(candidates, reach, -1.).max(-1, keepdims=True)
    candidates = candidates & np.isclose(reach, best_reach, rtol=0, atol=1e-9)
    stuck = best_reach[..., 0] <= 1e-9
    best_cost = np.where(candidates, cost, np.inf).min(-1, keepdims=True)
    candidates = np.where(stuck[..., None], union & graph.legal[None],
                          candidates & np.isclose(cost, best_cost, rtol=0, atol=1e-9))
    group_choice = np.where(candidates, probability, -1.).argmax(-1)
    chosen = np.where(informative, np.take_along_axis(group_choice, leader, axis=0), move)
    chosen = np.where(rows, chosen, NO_CHOICE)
    stuck_cases = np.take_along_axis(stuck, leader, axis=0) & informative
    # Rows with several cheapest first steps are the ones Stage 3b really decides; a row with one
    # keeps it. "changed" also counts rows where the head's own favorite move was simply wrong.
    tied = informative & (np.take_along_axis(union.sum(-1), leader, axis=0) > 1)
    return chosen, {"changed_rows": int((rows & (chosen != move)).sum()),
                    "tied_rows": int(tied.sum()), "tied_rows_moved": int((tied & (chosen != move)).sum()),
                    "no_thread_reaches_rows": int(stuck_cases.sum())}


class SupervisedActionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.head_input_dim, cfg.decision_hidden), nn.ReLU(),
                                 nn.Linear(cfg.decision_hidden, cfg.action_count))

    def forward(self, features):
        return self.net(features)

def predictor_fingerprint(predictor):
    digest = hashlib.sha256()
    for name, tensor in sorted(predictor.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(np.asarray(predictor.temperatures).tobytes())
    return digest.hexdigest()


def head_fingerprint(weights):
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rules_fingerprint(cfg):
    """The acquisition rules and costs themselves: every allowed move of every state and what it adds."""
    graph = decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    digest = hashlib.sha256()
    for array in (graph.succ, graph.delta):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def decision_dataset_fingerprint(records, splits, teacher, predictor, cfg):
    """Stage 3 identity: config, rules, records, roles, teacher labels, and frozen predictor.

    The input log paths are left out: the records' content and uids already identify them. The
    rules are hashed as the transition table they produce, so editing them invalidates saved labels
    even within one revision.
    """
    config = {key: value for key, value in asdict(cfg).items() if key not in NON_CONTRACT_SETTINGS}
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema": 5, "revision": PIPELINE_REVISION, "config": config,
                              "rules": rules_fingerprint(cfg),
                              "predictor": predictor_fingerprint(predictor),
                              "data_sha256": data_digest(records),
                              "roles": {role: [records[i].uid for i in splits[role]]
                                        for role in ("policy", "dev")}}, sort_keys=True).encode())
    for role in ("policy", "dev"):
        for idx in splits[role]:
            for key in ("scores", "safe", "maximum"):
                digest.update(np.asarray(teacher[key][idx], dtype=np.float32).tobytes())
    return digest.hexdigest()


def finetune_fingerprint(dataset_fingerprint, pretrained_weights):
    """Stage 3b labels follow from the dataset and from the pretrained head that rolled them out."""
    return hashlib.sha256(json.dumps({"dataset": dataset_fingerprint, "phase": "finetune",
                                      "pretrained": head_fingerprint(pretrained_weights)},
                                     sort_keys=True).encode()).hexdigest()


# One label array per question and phase. Pretraining (3a) stores, for every case, the bit mask of
# its cheapest first steps (two bytes for the default 11 actions; all zero when the case has no
# row). Fine-tuning (3b) stores the single chosen action (NO_CHOICE when the case has no row).
DECISION_LABEL_SCHEMA = 6
HEAD_PHASES = {
    "pretrain": {"title": "Stage 3a", "key": "options", "extras": ("observation_sha256",),
                 "epochs": "decision_epochs", "lr": "decision_lr", "seed": 300,
                 "progress": "decision_head_pretrain_progress.pt"},
    "finetune": {"title": "Stage 3b", "key": "choice", "extras": (),
                 "epochs": "decision_finetune_epochs", "lr": "decision_finetune_lr", "seed": 310,
                 "progress": "decision_head_finetune_progress.pt"}}


def question_paths(records, ids, directory):
    return [directory / f"{position:06d}_{hashlib.sha256(records[idx].uid.encode()).hexdigest()[:12]}.pt"
            for position, idx in enumerate(ids)]


def decision_role_path(out, phase, role):
    return Path(out) / "decision_labels" / f"{phase}_{role}.pt"


def reclaim_derived_stage3_files(out):
    """Delete interrupted writes; completed work is never touched.

    An interrupted atomic write leaves a `.tmp` file that holds no completed work. Removing
    it before a resumed run writes anything lets a run that filled its disk continue.
    """
    out, reclaimed = Path(out), 0
    for pattern in ("*.tmp", "decision_labels/*.tmp", "decision_labels/*/*.tmp"):
        for path in out.glob(pattern):
            if path.is_file():
                reclaimed += path.stat().st_size
                path.unlink()
    return reclaimed


def pack_decision_rows(rows, cfg, keys=("h", "target", "valid")):
    """Stack the head's row arrays from row dictionaries."""
    shapes = {"h": (cfg.head_input_dim, torch.float32), "target": (cfg.action_count, torch.float32),
              "valid": (cfg.action_count, torch.bool)}
    return {key: torch.from_numpy(np.stack([row[key] for row in rows])) if rows else
            torch.empty((0, shapes[key][0]), dtype=shapes[key][1])
            for key in keys}


@lru_cache(maxsize=8)
def decision_valid_table(zero_shots, k, repeats, cost_unit, max_cost):
    """Allowed-action masks of the in-budget states, in label order."""
    return decision_graph(zero_shots, k, repeats, cost_unit, max_cost).legal


def decision_valid(source, cfg):
    table = decision_valid_table(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    return table[np.asarray(source, dtype=np.int64) % len(table)]


def decision_case_count(cfg):
    """In-budget state/order cases of one question: zero-shot orders times in-budget states."""
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    return math.factorial(cfg.zero_shots) * len(states)


def decision_row_observations(record, source, cfg):
    """Rebuild each case's label-time observation from its scenario * states + state source."""
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    template, masks, offsets = state_observation_template(
        states, cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    scenario, position = np.divmod(np.asarray(source, dtype=np.int64), len(states))
    blank = np.zeros(cfg.n, np.float32)
    orders = [permute_zero_shots(record, blank, blank, blank, order, cfg)[0]
              for order in permutations(range(cfg.zero_shots))]
    sources = (np.stack([r.similarity for r in orders]), np.stack([r.baseline for r in orders]),
               np.stack([r.ccs.ravel() for r in orders]))
    xs = template[position]
    for values, mask, start in zip(sources, masks, offsets):
        visible = mask[position]
        # As in observation_many, index before assignment: unobserved cells stay zero.
        xs[:, start:start+visible.shape[1]][visible] = values[scenario][visible]
    if not np.isfinite(xs).all():
        raise ValueError("Invalid snapshot observation.")
    return xs


def label_rows(values, phase):
    """Indices of the cases that have a training row."""
    values = np.asarray(values)
    return np.flatnonzero(values.any(1) if phase == "pretrain" else values != NO_CHOICE)


def decision_targets(values, phase, cfg):
    """Soft targets: pretraining splits each row evenly over its cheapest first steps; fine-tuning is one-hot."""
    if phase == "finetune":
        target = np.zeros((len(values), cfg.action_count), np.float32)
        target[np.arange(len(values)), np.asarray(values, np.int64)] = 1
        return target
    support = np.unpackbits(np.asarray(values, np.uint8), axis=1, count=cfg.action_count).astype(bool)
    return np.where(support, 1 / np.maximum(support.sum(1, keepdims=True), 1), 0).astype(np.float32)


def decision_training_rows(record, source, values, phase, predictor, cfg):
    """One question's head inputs: decoded targets plus re-encoded frozen features."""
    source = np.asarray(source)
    return {"h": predictor.features(decision_row_observations(record, source, cfg)),
            "target": torch.from_numpy(decision_targets(values, phase, cfg)),
            "valid": torch.from_numpy(decision_valid(source, cfg))}


def load_label_file(path):
    """Compact label files hold only tensors, strings, and counts."""
    return torch.load(path, map_location="cpu", weights_only=True)


def build_route_labels(record, scores, safe, maximum, predictor, cfg):
    """Stage 3a labels of one question: every case's cheapest first steps, as packed bit masks."""
    plan = question_plan(record, scores, safe, maximum, predictor, cfg)
    cases = plan["rows"].reshape(-1)
    source = np.flatnonzero(cases)
    observed = decision_row_observations(record, source, cfg)
    rebuilt = np.frombuffer(b"".join(hashlib.sha256(x.tobytes()).digest() for x in observed),
                            np.uint64).reshape(-1, 4)
    if not np.array_equal(rebuilt, plan["digest"].reshape(-1, 4)[source]):
        raise ValueError("Decision rows do not reproduce their visible observations.")
    coverage = plan_coverage(plan, cfg)
    return {"options": torch.from_numpy(np.packbits(plan["targets"].reshape(len(cases), -1), axis=1)),
            "observation_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
            "coverage": coverage}


def build_finetune_labels(record, scores, safe, maximum, predictor, head, cfg):
    """Stage 3b labels of one question: the single action chosen for every case that has a row."""
    plan = question_plan(record, scores, safe, maximum, predictor, cfg)
    chosen, extra = finetune_choices(plan, head, predictor.device, cfg)
    return {"choice": torch.from_numpy(chosen.reshape(-1).astype(np.uint8)),
            "coverage": {**plan_coverage(plan, cfg), **extra}}


def check_route_labels(record, labels, cfg):
    """Saved Stage 3a labels must rebuild their label-time observations under the current code.

    Returns the source of every case that has a row.
    """
    width, cases = (cfg.action_count + 7) // 8, decision_case_count(cfg)
    bits = labels["options"]
    if bits.dtype != torch.uint8 or tuple(bits.shape) != (cases, width):
        raise ValueError(f"Saved Stage 3 labels for {record.uid} have inconsistent shapes.")
    support = np.unpackbits(bits.numpy(), axis=1)
    if support[:, cfg.action_count:].any():
        raise ValueError(f"Saved Stage 3 labels for {record.uid} have inconsistent shapes.")
    support = support[:, :cfg.action_count].astype(bool)
    source = np.flatnonzero(support.any(1))
    if len(source) != labels["coverage"]["action_rows"] or (support[source] & ~decision_valid(source, cfg)).any():
        raise ValueError(f"Saved Stage 3 labels for {record.uid} target invalid actions.")
    observed = decision_row_observations(record, source, cfg)
    if hashlib.sha256(observed.tobytes()).hexdigest() != labels["observation_sha256"]:
        raise ValueError(f"Saved Stage 3 labels for {record.uid} no longer reproduce their "
                         "label-time observations; observation code or data changed.")
    return source


def check_finetune_labels(record, labels, cfg, reference=None):
    """Saved Stage 3b labels choose an allowed action for exactly the cases that have a row."""
    choice, cases = labels["choice"], decision_case_count(cfg)
    if choice.dtype != torch.uint8 or tuple(choice.shape) != (cases,):
        raise ValueError(f"Saved Stage 3 labels for {record.uid} have inconsistent shapes.")
    values = choice.numpy()
    source = np.flatnonzero(values != NO_CHOICE)
    if (len(source) != labels["coverage"]["action_rows"] or (values[source] >= cfg.action_count).any() or
            not decision_valid(source, cfg)[np.arange(len(source)), values[source]].all()):
        raise ValueError(f"Saved Stage 3 labels for {record.uid} choose invalid actions.")
    if reference is not None and not np.array_equal(source, label_rows(reference.numpy(), "pretrain")):
        raise ValueError(f"Saved Stage 3b labels for {record.uid} do not cover the Stage 3a rows.")
    return source


def decision_role_labels(phase, role, records, ids, cfg, out, fingerprint, build, check, sync=None):
    """Build, validate, or reuse one phase's labels for one role; returns them in RAM.

    Each question completes as one small atomic file, and the finished role is merged into a
    single file. `build(idx)` labels a question; `check(idx, labels)` raises when saved labels
    do not belong to the current code and data.
    """
    spec, out = HEAD_PHASES[phase], Path(out)
    key = spec["key"]
    uids = [records[idx].uid for idx in ids]
    role_path = decision_role_path(out, phase, role)
    paths = question_paths(records, ids, out / "decision_labels" / f"{phase}_{role}")
    store = load_label_file(role_path) if role_path.exists() else None
    if store is not None and (
            store.get("schema_version") != DECISION_LABEL_SCHEMA or store.get("fingerprint") != fingerprint
            or store.get("phase") != phase or store.get("role") != role or store.get("uids") != uids
            or not isinstance(store.get(key), torch.Tensor) or len(store[key]) != len(ids)):
        raise ValueError(f"Incompatible decision labels: {role_path}")
    values, extras, parts = None, {name: [] for name in spec["extras"]}, []
    started = last_print = time.monotonic()
    built = reused = 0
    print(f"{spec['title']} {role} labels: building/validating {len(ids)} questions.", flush=True)
    for position, (idx, path) in enumerate(zip(ids, paths), 1):
        record = records[idx]
        labels = None
        if store is not None:
            labels = {key: store[key][position-1], "coverage": store["coverage"][position-1],
                      **{name: store[name][position-1] for name in spec["extras"]}}
        elif path.exists():
            labels = load_label_file(path)
            if (labels.get("schema_version") != DECISION_LABEL_SCHEMA or labels.get("fingerprint") != fingerprint
                    or labels.get("phase") != phase or labels.get("uid") != record.uid):
                raise ValueError(f"Incompatible decision labels: {path}")
        if labels is not None:
            check(idx, labels)
            reused += 1
        else:
            labels = {"schema_version": DECISION_LABEL_SCHEMA, "fingerprint": fingerprint, "phase": phase,
                      "uid": record.uid, **build(idx)}
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch(path, labels)
            built += 1
        if store is None:
            if values is None:
                values = torch.empty((len(ids), *labels[key].shape), dtype=labels[key].dtype)
            values[position-1] = labels[key]
            for name in extras:
                extras[name].append(labels[name])
            parts.append(labels["coverage"])
        last_print = decision_progress(f"{spec['title']} {role} labels", position, len(ids), started,
                                       last_print, cfg, f" | built {built}, reused {reused}")
        if sync is not None:
            sync.push(f"{spec['title']}: {role} labels {position}/{len(ids)} questions")
    if store is None:
        store = {"schema_version": DECISION_LABEL_SCHEMA, "fingerprint": fingerprint, "phase": phase,
                 "role": role, "uids": uids, key: values, **extras, "coverage": parts}
        atomic_torch(role_path, store)
    # The merged role file is complete; remove per-question files and leftovers.
    for path in paths:
        path.unlink(missing_ok=True)
    try:
        (out / "decision_labels" / f"{phase}_{role}").rmdir()
    except OSError:
        pass  # Missing, or holds files this workflow does not own.
    if sync is not None:
        sync.push(f"{spec['title']}: {role} labels complete", force=True)
    return store


def decision_question_audit(records, splits, teacher, predictor, cfg, out, role, position):
    """Rebuild one saved question's cases with their dispositions, checked against its Stage 3a labels.

    Stage 3 stores only the labels the head trains on. Goal ranks, distances to the next goal,
    and dispositions are deterministic, so they are recomputed here on demand (use the labeling
    device; another device may round probabilities differently).
    """
    fingerprint = decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    path = decision_role_path(out, "pretrain", role)
    store = load_label_file(path)
    if (store.get("schema_version") != DECISION_LABEL_SCHEMA or store.get("fingerprint") != fingerprint
            or store.get("role") != role):
        raise ValueError(f"Incompatible decision labels: {path}")
    idx = splits[role][position]
    plan = question_plan(records[idx], teacher["scores"][idx], teacher["safe"][idx],
                         teacher["maximum"][idx], predictor, cfg)
    packed = np.packbits(plan["targets"].reshape(plan["rows"].size, -1), axis=1)
    if not np.array_equal(packed, store["options"][position].numpy()):
        raise ValueError("Rebuilt Stage 3 labels differ from the saved labels.")
    graph, names = plan["graph"], action_names(cfg)
    cases = []
    for order in range(plan["status"].shape[0]):
        for i, state in enumerate(graph.states):
            rank = int(plan["achieved"][order, i])
            cases.append({"scenario": order, "candidate_mask": state.candidate_mask,
                          "evaluator_mask": state.evaluator_mask,
                          "status": DECISION_STATUSES[plan["status"][order, i]],
                          "evaluators": evaluator_names(state.evaluator_mask, cfg),
                          "goal_rank": None if rank == cfg.n else rank,
                          "has_row": bool(plan["rows"][order, i]),
                          "targets": [names[a] for a in np.flatnonzero(plan["targets"][order, i])]})
    return {"uid": records[idx].uid, "cases": cases, "coverage": plan_coverage(plan, cfg)}


def masked_action_loss(model, rows, order, cfg, device, optimizer=None, *, batch_size=None):
    if not len(order):
        return 0., 0
    packed = rows if isinstance(rows, dict) else pack_decision_rows(rows, cfg)
    # One question fits in memory: transfer once, then index all minibatches there.
    h_all = packed["h"].to(device=device, dtype=torch.float32)
    target_all = packed["target"].to(device=device, dtype=torch.float32)
    valid_all = packed["valid"].to(device=device, dtype=torch.bool)
    indices = torch.as_tensor(np.asarray(order, dtype=np.int64), device=device)
    total, count = torch.zeros((), dtype=torch.float64, device=device), 0
    size = cfg.decision_batch_size if batch_size is None else batch_size
    if optimizer is not None and size != cfg.decision_batch_size:
        raise ValueError("Training must use decision_batch_size to preserve optimizer steps.")
    for start in range(0, len(order), size):
        batch = indices[start:start + size]
        h, target, valid = h_all[batch], target_all[batch], valid_all[batch]
        logits = model(h).masked_fill(~valid, -1e9)
        loss = -(target * nn.functional.log_softmax(logits, dim=1)).sum(1).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        # Synchronize a GPU once per question instead of once per minibatch.
        total += loss.detach().double() * len(batch)
        count += len(batch)
    return float(total), count


def decision_progress(label, completed, total, started, last_print, cfg, detail="", every=None, seconds=30):
    now = time.monotonic()
    every = cfg.print_every if every is None else every
    if completed == 1 or completed == total or completed % every == 0 or now - last_print >= seconds:
        elapsed = now - started
        eta = elapsed * (total - completed) / completed
        print(f"{label}: {completed}/{total} questions | elapsed {elapsed/60:.1f} min | "
              f"ETA {eta/60:.1f} min{detail}", flush=True)
        return now
    return last_print


def decision_manifest(out, fingerprint):
    """Record, or verify, the dataset identity every Stage 3 label file is bound to."""
    manifest = {"schema_version": DECISION_LABEL_SCHEMA, "implementation_revision": PIPELINE_REVISION,
                "fingerprint": fingerprint,
                "roles": {phase: {role: decision_role_path(out, phase, role).relative_to(out).as_posix()
                                  for role in ("policy", "dev")} for phase in HEAD_PHASES}}
    path = Path(out) / "decision_dataset_manifest.json"
    if not path.exists():
        atomic_json(path, manifest)
    elif json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Decision dataset manifest has an incompatible teacher, predictor, split, or action contract.")


def phase_labels(phase, records, splits, cfg, out, fingerprint, builders, checkers, sync):
    """Both roles' labels for one phase, with their coverage written beside the labels."""
    labels, coverage = {}, {}
    for role in ("policy", "dev"):
        labels[role] = decision_role_labels(phase, role, records, splits[role], cfg, out, fingerprint,
                                            builders[role], checkers[role], sync)
        coverage[role] = combine_coverage(labels[role]["coverage"])
        title = HEAD_PHASES[phase]["title"]
        print(f"{title} {role}: {coverage[role]['state_order_cases']} state/order cases, "
              f"{coverage[role]['action_rows']} head rows, {coverage[role]['informative_rows']} of them "
              f"with a route to their next goal; statuses {coverage[role]['status_counts']}; "
              f"cheapest first steps per routed row {coverage[role]['option_counts']}"
              + (f"; {coverage[role].get('tied_rows', 0)} rows had tied cheapest first steps, decided by the "
                 f"pretrained head's continuations ({coverage[role].get('tied_rows_moved', 0)} of them away from "
                 f"its favorite move), {coverage[role].get('changed_rows', 0)} rows in all differ from that "
                 f"favorite, and {coverage[role].get('no_thread_reaches_rows', 0)} routed rows have no "
                 "continuation that reached the goal" if phase == "finetune" else "") + ".", flush=True)
        if not coverage[role]["action_rows"]:
            raise ValueError(f"No actionable Stage 3 rows for {role}: {coverage[role]}")
    path = Path(out) / "decision_label_coverage.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    saved[phase] = coverage
    atomic_json(path, saved)
    return labels, coverage


def fit_action_head(phase, records, splits, labels, predictor, cfg, device, out, fingerprint,
                    sync=None, initial_weights=None):
    """Train the action head on one phase's labels. A completed epoch is a resume point."""
    spec, out = HEAD_PHASES[phase], Path(out)
    key, title = spec["key"], spec["title"]
    epochs, lr = getattr(cfg, spec["epochs"]), getattr(cfg, spec["lr"])
    seed_everything(cfg.seed + spec["seed"])

    def question_rows(role, position):
        values = labels[role][key][position].numpy()
        source = label_rows(values, phase)
        # Row sources come from the question's labels; observations and features are rebuilt.
        return decision_training_rows(records[splits[role][position]], source, values[source], phase,
                                      predictor, cfg)

    model = SupervisedActionHead(cfg).to(device)
    if initial_weights is not None:
        model.load_state_dict(initial_weights)
    # One fused update kernel on CUDA: the head's small minibatch steps are otherwise launch-bound.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay,
                           fused=torch.device(device).type == "cuda")
    best, weights, best_epoch, stale, history = math.inf, None, 0, 0, []
    rng = np.random.RandomState(cfg.seed + spec["seed"] + 1)
    progress_path = out / spec["progress"]
    start_epoch = 0
    if cfg.resume and progress_path.exists():
        progress = load_checkpoint(progress_path)
        if (progress.get("schema_version") != 2 or progress.get("phase") != phase
                or progress.get("fingerprint") != fingerprint):
            raise ValueError("Decision head progress has an incompatible training contract.")
        start_epoch = progress["epoch"]
        if not (0 < start_epoch <= epochs and
                len(progress["history"]) == start_epoch and
                1 <= progress["best_epoch"] <= start_epoch and progress["stale"] >= 0):
            raise ValueError("Decision head progress has inconsistent epoch bookkeeping.")
        model.load_state_dict(progress["current_weights"])
        opt.load_state_dict(progress["optimizer"])
        rng.set_state(progress["rng_state"])
        best, weights, best_epoch = progress["dev_loss"], progress["best_weights"], progress["best_epoch"]
        stale, history = progress["stale"], progress["history"]
        del progress
        print(f"{title} head resume: {start_epoch} completed epochs; "
              "restored optimizer, shuffle RNG, and best development checkpoint.", flush=True)
    print(f"{title} head training: device={device}, up to {epochs} epochs at learning rate {lr:g}, "
          f"train batch={cfg.decision_batch_size}, dev batch={cfg.decision_eval_batch_size}. "
          "Labels are complete; epochs only re-encode stored observations with the frozen "
          "predictor.", flush=True)
    for epoch in range(start_epoch, epochs):
        if stale >= cfg.patience:
            break
        epoch_started = time.monotonic()
        model.train()
        started = last_print = time.monotonic()
        train_total, train_count = 0., 0
        for position, question in enumerate(rng.permutation(len(splits["policy"])), 1):
            rows = question_rows("policy", question)
            value, n = masked_action_loss(model, rows, rng.permutation(len(rows["target"])), cfg, device, opt)
            train_total, train_count = train_total + value, train_count + n
            del rows
            last_print = decision_progress(f"{title} epoch {epoch+1} train", position, len(splits["policy"]),
                                           started, last_print, cfg, f" | rows {train_count}")
        model.eval()
        with torch.no_grad():
            started = last_print = time.monotonic()
            total, count = 0., 0
            for position in range(len(splits["dev"])):
                rows = question_rows("dev", position)
                value, n = masked_action_loss(model, rows, range(len(rows["target"])), cfg, device,
                                              batch_size=cfg.decision_eval_batch_size)
                total, count = total + value, count + n
                del rows
                last_print = decision_progress(f"{title} epoch {epoch+1} dev", position + 1, len(splits["dev"]),
                                               started, last_print, cfg, f" | rows {count}")
            dev_loss = total / count
        history.append({"epoch": epoch+1, "dev_loss": dev_loss})
        if dev_loss < best - 1e-7:
            best, weights, stale, best_epoch = dev_loss, cpu_state(model), 0, epoch+1
        else:
            stale += 1
        # Only complete train+dev epochs commit. An interruption repeats at most
        # the current partial epoch, with the exact question/row shuffle restored.
        atomic_torch(progress_path, {
            "schema_version": 2, "phase": phase, "fingerprint": fingerprint, "epoch": epoch+1,
            "current_weights": cpu_state(model), "optimizer": opt.state_dict(),
            "rng_state": rng.get_state(), "best_weights": weights,
            "best_epoch": best_epoch, "dev_loss": best, "stale": stale, "history": history})
        print(f"{title} epoch {epoch+1}/{epochs}: train loss {train_total/train_count:.4f}, "
              f"dev loss {dev_loss:.4f}, best {best:.4f} (epoch {best_epoch}), "
              f"patience {stale}/{cfg.patience} | elapsed {(time.monotonic()-epoch_started)/60:.1f} min",
              flush=True)
        if sync is not None:
            sync.push(f"{title}: epoch {epoch+1} progress")
        if stale >= cfg.patience:
            break
    return {"weights": weights, "best_epoch": best_epoch, "dev_loss": best, "history": history,
            "input_dim": cfg.head_input_dim, "action_count": cfg.action_count,
            "action_names": action_names(cfg), "dataset_fingerprint": fingerprint, "phase": phase}


def stage3_storage_note(splits, cfg, out):
    cases = decision_case_count(cfg)
    per_question = cases * ((cfg.action_count + 7) // 8 + 1)  # Both phases' labels.
    upper = 2 * sum(len(splits[role]) for role in ("policy", "dev")) * per_question
    free = shutil.disk_usage(out).free
    print(f"Stage 3 storage: {Path(out).resolve()} | {free / 1024**3:.2f} GiB free. Labels need at "
          f"most {upper / 1024**2:.1f} MiB for both phases ({(cfg.action_count + 7) // 8} bytes per "
          f"case for the cheapest first steps and one for the fine-tuning label); frozen features, row "
          "sources, and audit fields are recomputed rather than saved.", flush=True)
    if upper > free:
        print(f"WARNING: Stage 3 labels may need up to {upper / 1024**2:.1f} MiB, but only "
              f"{free / 1024**2:.1f} MiB is free. Free space now; completed questions are reused on "
              "resume.", flush=True)


def train_pretrained_head(records, splits, teacher, predictor, cfg, device, out=None, sync=None):
    """Stage 3a: pretrain on every cheapest first step of every state under every zero-shot order."""
    if splits["policy"] != splits["supervised"]:
        raise ValueError("Stage 3 policy questions must match the Stage 1/2 supervised questions.")
    if out is None:
        raise ValueError("Stage 3 requires an output directory for its label files.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    frozen_before = predictor_fingerprint(predictor)
    decision_manifest(out, fingerprint)
    # Free space before any write: an interrupted write leaves a .tmp file.
    reclaimed = reclaim_derived_stage3_files(out)
    if reclaimed:
        print(f"Stage 3: removed {reclaimed / 1024**2:.2f} MiB of incomplete writes.", flush=True)
    stage3_storage_note(splits, cfg, out)
    builders = {role: (lambda idx: build_route_labels(records[idx], teacher["scores"][idx],
                                                      teacher["safe"][idx], teacher["maximum"][idx],
                                                      predictor, cfg)) for role in ("policy", "dev")}
    checkers = {role: (lambda idx, labels: check_route_labels(records[idx], labels, cfg))
                for role in ("policy", "dev")}
    labels, coverage = phase_labels("pretrain", records, splits, cfg, out, fingerprint, builders, checkers, sync)
    held = sum(labels[role]["options"].numel() for role in labels)
    print(f"Stage 3a labels in RAM: {held / 1024**2:.1f} MiB for "
          f"{sum(coverage[role]['action_rows'] for role in coverage)} rows.", flush=True)
    checkpoint = fit_action_head("pretrain", records, splits, labels, predictor, cfg, device, out,
                                 fingerprint, sync)
    if predictor_fingerprint(predictor) != frozen_before:
        raise AssertionError("Frozen Stage 2 predictor changed during decision training.")
    return {**checkpoint, "coverage": coverage}


def finetune_labels(records, splits, teacher, predictor, pretrained, cfg, device, out, sync=None):
    """Stage 3b labels, built or validated: one action per row, chosen by the pretrained head's continuations.

    This is the dataset half of Stage 3b: it needs the pretrained head, but no fine-tuning.
    Returns the labels in RAM, their coverage, and the fingerprint they are bound to.
    """
    out = Path(out)
    dataset = decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    if pretrained.get("dataset_fingerprint") != dataset:
        raise ValueError("The pretrained decision head belongs to a different dataset or action contract.")
    fingerprint = finetune_fingerprint(dataset, pretrained["weights"])
    decision_manifest(out, dataset)
    head = SupervisedActionHead(cfg).to(device)
    head.load_state_dict(pretrained["weights"])
    head.eval()
    positions = {role: {idx: position for position, idx in enumerate(splits[role])} for role in ("policy", "dev")}
    references = {role: (load_label_file(decision_role_path(out, "pretrain", role))["options"]
                         if decision_role_path(out, "pretrain", role).exists() else None)
                  for role in ("policy", "dev")}

    def builder(idx):
        return build_finetune_labels(records[idx], teacher["scores"][idx], teacher["safe"][idx],
                                     teacher["maximum"][idx], predictor, head, cfg)

    def checker(role):
        def check(idx, labels):
            reference = references[role]
            check_finetune_labels(records[idx], labels, cfg,
                                  None if reference is None else reference[positions[role][idx]])
        return check

    builders = {role: builder for role in ("policy", "dev")}
    checkers = {role: checker(role) for role in ("policy", "dev")}
    labels, coverage = phase_labels("finetune", records, splits, cfg, out, fingerprint, builders, checkers, sync)
    references.clear()  # Only the row check needs the Stage 3a labels, which are large.
    return labels, coverage, fingerprint


def train_finetuned_head(records, splits, teacher, predictor, pretrained, cfg, device, out, sync=None):
    """Stage 3b: fine-tune the pretrained head on one action per row, chosen by its own continuations."""
    frozen_before = predictor_fingerprint(predictor)
    labels, coverage, fingerprint = finetune_labels(records, splits, teacher, predictor, pretrained, cfg,
                                                    device, out, sync)
    checkpoint = fit_action_head("finetune", records, splits, labels, predictor, cfg, device, out,
                                 fingerprint, sync, initial_weights=pretrained["weights"])
    if predictor_fingerprint(predictor) != frozen_before:
        raise AssertionError("Frozen Stage 2 predictor changed during decision training.")
    return {**checkpoint, "coverage": coverage, "pretrained_fingerprint": head_fingerprint(pretrained["weights"])}

# %% Reports, artifacts, and stage orchestration
def wilson_interval(errors, total):
    if not total:
        return None
    p, z = errors / total, 1.959963984540054
    den = 1 + z*z/total
    centre = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0., centre-half), min(1., centre+half)]


def paired_interval(a, b, repetitions, seed):
    if not a or not b:
        return None
    if [r["uid"] for r in a] != [r["uid"] for r in b]:
        raise ValueError("Paired reports have different target universes.")
    grouped = {}
    for x, y in zip(a, b):
        grouped.setdefault(x["uid"], []).append(x["correct"]-y["correct"])
    return bootstrap_difference([np.mean(values) for values in grouped.values()], repetitions, seed)


def bootstrap_difference(question_differences, repetitions, seed):
    """Mean paired difference with a bootstrap over questions, in report order."""
    difference = np.asarray(question_differences, float)
    rng = np.random.RandomState(seed)
    sampled = [float(rng.choice(difference, len(difference), replace=True).mean())
               for _ in range(max(1, repetitions))]
    return {"delta": float(difference.mean()),
            "ci95": np.percentile(sampled, [2.5, 97.5]).tolist(),
            "benefits": int((difference > 0).sum()), "harms": int((difference < 0).sum())}


def select_heuristics(records, ids, cfg):
    selected = {}
    for mode in ("Absolute", "Marginal"):
        best = (-1., None)
        for mask in ("Self", "Others", "All"):
            for strategy in ("ScoreTake", "ScoreMake", "Holistic"):
                scores = {i: heuristic_scores(records[i], cfg, mode, mask, strategy) for i in ids}
                ap = ranking_metrics(records, ids, scores, cfg)["ap"]
                if ap > best[0]:
                    best = (ap, (mode, mask, strategy))
        selected[mode] = best[1]
    return selected


def teacher_report(records, ids, cfg, scores, heuristics, title):
    section(title)
    if not ids:
        print("No eligible records. See data_audit.json for exclusions; no metric is imputed.")
        return None
    baseline = {i: np.zeros(cfg.n) for i in ids}
    methods = {"Baseline (Retrieval Order)": baseline}
    for mode, spec in heuristics.items():
        methods[f"{mode} ({spec[1]}/{spec[2]})"] = {
            i: heuristic_scores(records[i], cfg, *spec) for i in ids}
    methods["ResNet teacher"] = scores
    metrics = {name: ranking_metrics(records, ids, s, cfg) for name, s in methods.items()}
    base_ap = metrics["Baseline (Retrieval Order)"]["ap"]
    print(f"Queries: {len(ids)} | Full-pool oracle coverage: {metrics['ResNet teacher']['coverage']:.1%}")
    print(f"{'Ranking method':<43} | {'Mean AP':>8} | {'Top-1':>8} | {'AP delta':>9}")
    print("-"*80)
    for name, m in metrics.items():
        print(f"{name:<43} | {m['ap']:8.4f} | {m['top1']:7.1%} | {(m['ap']-base_ap)*100:+8.2f}pp")
    print("All eligible questions included, including all-wrong pools. Heuristics selected on development only.")
    return metrics


def fixed_order_items(cfg, order="interleaved"):
    if not isinstance(order, str):
        if not isinstance(order, (list, tuple)):
            raise ValueError("A fixed order must be a built-in name or a list of acquisition items.")
        return list(order)
    evaluators = [f"R{i+1}" for i in range(cfg.k)]
    one_shots = [f"OS{i+1}" for i in range(cfg.k)]
    zero_shots = [f"ZS{i+1}" for i in range(cfg.zero_shots)]
    if order == "evaluators_first":
        return ["ZS1"] + evaluators + zero_shots[1:] + one_shots
    if order == "zero_shots_first":
        return zero_shots + [item for pair in zip(evaluators, one_shots) for item in pair]
    if order != "interleaved":
        raise ValueError(f"Unknown fixed acquisition order: {order}")
    items = ["ZS1"]
    for source in range(cfg.k):
        items.extend((evaluators[source], one_shots[source]))
        if source < cfg.zero_shots - 1:
            items.append(zero_shots[source+1])
    items.extend(zero_shots[cfg.k+1:])
    return items


def fixed_order_actions(cfg, order="interleaved"):
    items = fixed_order_items(cfg, order)
    if not items or items[0] != "ZS1":
        raise ValueError("Every fixed sequence must start with the initial ZS1 candidate.")
    state, actions = State(), []
    for item in items[1:]:
        match = re.fullmatch(r"(ZS|R|OS)([1-9][0-9]*)", item) if isinstance(item, str) else None
        if match is None:
            raise ValueError(f"Invalid fixed acquisition item: {item!r}; use ZS2, R1, OS1, etc.")
        kind, number = match.group(1), int(match.group(2))
        limit = cfg.zero_shots if kind == "ZS" else cfg.k
        if number > limit:
            raise ValueError(f"Fixed acquisition item {item} exceeds the configured pool.")
        action = 0 if kind == "ZS" else number if kind == "R" else cfg.k + number
        nxt = raw_next_state(state, action, cfg)
        # The zero-shot action fills the first missing slot, so ZSn must be exactly that slot;
        # Rj and OSj name their action, which the acquisition rules must allow at this point.
        if nxt is None or kind == "ZS" and (nxt.candidate_mask ^ state.candidate_mask) != 1 << (number-1):
            raise ValueError(f"Invalid fixed acquisition order at {item}: a duplicate, a source that "
                             "the similarity order has not reached yet, or an out-of-order zero-shot.")
        actions.append(action)
        state = nxt
    return actions


def fixed_acquisition_sequences(cfg):
    if not isinstance(cfg.fixed_acquisition_orders, dict):
        raise ValueError("fixed_acquisition_orders must map method names to acquisition orders.")
    sequences = {}
    for name, order in cfg.fixed_acquisition_orders.items():
        if not isinstance(name, str) or re.fullmatch(r"fixed(?:_[a-z][a-z0-9_]*)?", name) is None:
            raise ValueError("Fixed method names must be 'fixed' or start with 'fixed_' using lowercase letters/numbers.")
        sequences[name] = {"items": fixed_order_items(cfg, order),
                           "actions": fixed_order_actions(cfg, order)}
    return sequences


# The two learned policies: the Stage 3a head and the Stage 3b head fine-tuned from it.
LEARNED_METHODS = ("supervised_pretrained", "supervised")


def evaluation_methods(cfg):
    return ("full", *cfg.fixed_acquisition_orders, *LEARNED_METHODS)

def rollout(record, maximum, predictor, cfg, policy="supervised", head=None, trace=False,
            continue_after_max=None):
    if policy not in evaluation_methods(cfg):
        raise ValueError(f"Unknown policy: {policy}")
    if continue_after_max is None:
        continue_after_max = cfg.continue_after_max
    elif type(continue_after_max) is not bool:
        raise ValueError("continue_after_max must be a boolean or None")
    if policy == "full":
        full_cfg = copy.deepcopy(cfg)
        full_cfg.max_cost = None
        state = State((1 << cfg.n)-1, (1 << cfg.k)-1)
        _, probabilities = predictor.predict(record, state, full_cfg)
        reason, trajectory = "full_budget_reference", []
        first_max_recognition = (
            {"candidate": record.candidate_ids[selected_candidate(probabilities, state, cfg)],
             "cost": state_cost(state, cfg)} if max_recognized(probabilities, state, cfg) else None)
    else:
        state, reason, trajectory, first_max_recognition = State(), None, [], None
        sequence = (fixed_acquisition_sequences(cfg)[policy]["actions"].copy()
                    if policy in cfg.fixed_acquisition_orders else None)
        while reason is None:
            features, probabilities = predictor.predict(record, state)
            recognized_max = max_recognized(probabilities, state, cfg)
            if recognized_max and first_max_recognition is None:
                first_max_recognition = {"candidate": record.candidate_ids[selected_candidate(probabilities, state, cfg)],
                                         "cost": state_cost(state, cfg)}
            if policy in LEARNED_METHODS and continue_after_max:
                if not valid_actions(state, cfg).any():
                    full = (state.candidate_mask == (1 << cfg.n)-1 and
                            state.evaluator_mask == (1 << cfg.k)-1)
                    reason = "ranked_list_complete" if full else "budget"
            else:
                reason = stopping_reason(probabilities, state, cfg)
            if trace:
                step = {"candidate_mask": int(state.candidate_mask),
                        "evaluator_mask": int(state.evaluator_mask),
                        "evaluators": evaluator_names(state.evaluator_mask, cfg),
                        "candidates": [record.candidate_ids[i] for i in np.flatnonzero(present_mask(state, cfg))],
                        "selected": record.candidate_ids[selected_candidate(probabilities, state, cfg)],
                        "max_signals": max_recognition_signals(probabilities, state, cfg),
                        "max_recognized": recognized_max,
                        "recognition_threshold": cfg.recognition_threshold,
                        "cost": state_cost(state, cfg)}
            if reason is not None:
                if trace:
                    trajectory.append({**step, "action": None, "reason": reason})
                break
            valid = valid_actions(state, cfg)
            if sequence is not None:
                action = sequence[0] if sequence else None
                if action is None or not valid[action]:
                    reason = "fixed_sequence_exhaustion" if action is None else "budget"
                    if trace:
                        trajectory.append({**step, "action": None,
                                           "reason": reason, "blocked_fixed_action": action})
                    break
                sequence.pop(0)
            elif policy in LEARNED_METHODS:
                action = greedy_action(head, features, valid, predictor.device)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            nxt = advance(state, action, cfg)
            if trace:
                acquired = np.flatnonzero(present_mask(nxt, cfg) & ~present_mask(state, cfg))
                slot = int(acquired[0]) if len(acquired) else None
                acquired_evaluators = nxt.evaluator_mask & ~state.evaluator_mask
                evaluator = acquired_evaluators.bit_length() - 1 if acquired_evaluators else None
                item = (f"ZS{slot+1}" if slot is not None and slot < cfg.zero_shots else
                        f"OS{slot-cfg.zero_shots+1}" if slot is not None else f"R{evaluator+1}")
                trajectory.append({**step, "action": int(action),
                                   "action_name": action_names(cfg)[action],
                                   "acquired_item": item,
                                   "acquired_candidate": record.candidate_ids[slot] if slot is not None else None,
                                   "acquired_evaluator": record.evaluator_ids[evaluator] if evaluator is not None else None,
                                   "cost_after": state_cost(nxt, cfg),
                                   "incremental_cost": state_cost(nxt, cfg) - state_cost(state, cfg),
                                   "valid_actions": np.flatnonzero(valid).tolist()})
            state = nxt
    selected = selected_candidate(probabilities, state, cfg)
    has_max = bool(maximum.any())
    exact_max = bool(maximum[selected]) if has_max else None
    order = ranked_indices(probabilities[:cfg.n], np.flatnonzero(present_mask(state, cfg)))
    return {"uid": record.uid, "group": record.group,
            "selected": record.candidate_ids[selected],
            "ranked_candidates": [record.candidate_ids[i] for i in order],
            "first_max_recognition": first_max_recognition,
            "candidate_mask": int(state.candidate_mask), "evaluator_mask": int(state.evaluator_mask),
            "acquisition_complete": (state.candidate_mask == (1 << cfg.n)-1 and
                                     state.evaluator_mask == (1 << cfg.k)-1),
            "final_max_recognized": max_recognized(probabilities, state, cfg),
            "correct": int(record.labels[selected]), "has_max": has_max,
            "any_correct": bool(record.labels.any()),
            "exact_max": exact_max, "cost": state_cost(state, cfg),
            "full_cost": cfg.full_cost,
            "call_counts": acquisition_call_counts(bin(state.candidate_mask).count("1"),
                                                   bin(state.evaluator_mask).count("1"), cfg),
            "full_call_counts": acquisition_call_counts(cfg.n, cfg.k, cfg),
            "saved_fraction": 1 - state_cost(state, cfg)/cfg.full_cost,
            "reason": reason, "trajectory": trajectory}

def question_rollouts(record, scores, safe, maximum, predictor, cfg, head=None,
                      policy="supervised", trace=False):
    """One question's evaluation rollouts, one per zero-shot order."""
    rows = []
    for order in permutations(range(cfg.zero_shots)):
        scenario, _, _, scenario_maximum = permute_zero_shots(record, scores, safe, maximum, order, cfg)
        # Evaluation measures the first predicted MAX stop, regardless of
        # the application wrapper's configurable continuation behavior.
        row = rollout(scenario, scenario_maximum, predictor, cfg, policy, head, trace,
                      continue_after_max=False)
        row["zero_shot_order"] = list(order)
        rows.append(row)
    return rows


def evaluate_policy(records, ids, teacher, predictor, cfg, head=None,
                    policy="supervised", trace=False):
    return [row for idx in ids
            for row in question_rollouts(records[idx], teacher["scores"][idx], teacher["safe"][idx],
                                         teacher["maximum"][idx], predictor, cfg, head, policy, trace)]

# Question cohorts by the teacher's historical ranking. no_max: the teacher's top-ranked
# candidate is wrong, so no stop can return a MAX candidate and every predicted-MAX stop
# there is false. It splits into teacher_miss (a correct candidate exists below that top)
# and all_wrong (no candidate is correct, so no stop can lower top-1).
SAVINGS_COHORTS = {"all": lambda m: True,
                   "max_present": lambda m: m["has_max"],
                   "no_max": lambda m: not m["has_max"],
                   "teacher_miss": lambda m: not m["has_max"] and m["any_correct"],
                   "all_wrong": lambda m: not m["any_correct"]}


def call_savings_summary(question_means):
    groups = {name: [row for row in question_means if keep(row)] for name, keep in SAVINGS_COHORTS.items()}
    result = {}
    for name, group in groups.items():
        with_max = [row for row in group if row["has_max"]]
        metrics = {"n": len(group),
                   "top1_correct": float(np.mean([row["correct"] for row in group])) if group else None,
                   "exact_max_recovery": float(np.mean([row["exact_max"] for row in with_max])) if with_max else None,
                   "predicted_max_stop_rate": float(np.mean([row["predicted_max"] for row in group])) if group else None,
                   "wrong_predicted_max_stop_rate": float(np.mean([row["wrong_predicted_max"] for row in group])) if group else None}
        stop_weight = sum(row["predicted_max"] for row in group)
        metrics["predicted_max_stop_precision"] = (
            float(sum(row["predicted_max"] - row["wrong_predicted_max"] for row in group) / stop_weight)
            if stop_weight and with_max else None)
        for unit in ("solver_calls", "total_calls"):
            used = np.asarray([row["call_counts"][unit] for row in group])
            full = np.asarray([row["full_call_counts"][unit] for row in group])
            metrics[unit] = {"mean_full": float(full.mean()) if group else None,
                             "mean_used": float(used.mean()) if group else None,
                             "mean_saved": float((full-used).mean()) if group else None,
                             "total_saved": float((full-used).sum()) if group else None,
                             # Over every benchmark question, so max_present + no_max = all.
                             "saved_per_benchmark_question": (float((full-used).sum() / len(question_means))
                                                              if group else None),
                             "saved_fraction": float((full-used).sum()/full.sum()) if group else None}
            for stop in ("correct_max", "wrong_max", "other"):
                metrics[unit][f"mean_saved_by_{stop}_stop"] = (
                    float(np.mean([row["savings_by_stop"][unit][stop] for row in group])) if group else None)
            # Condition on predicted-MAX stops after weighting each question equally.
            # A question's zero-shot orders must not multiply its influence.
            full_on_stop = (sum(row["full_call_counts"][unit] * row["predicted_max"]
                                for row in group) / stop_weight if stop_weight else None)
            saved_on_stop = (sum(row["savings_by_stop"][unit]["correct_max"] +
                                 row["savings_by_stop"][unit]["wrong_max"] for row in group) /
                             stop_weight if stop_weight else None)
            metrics[unit].update({
                "mean_used_on_predicted_max_stop": float(full_on_stop - saved_on_stop) if stop_weight else None,
                "mean_saved_on_predicted_max_stop": float(saved_on_stop) if stop_weight else None,
                "saved_fraction_on_predicted_max_stop": float(saved_on_stop / full_on_stop) if stop_weight else None})
        result[name] = metrics
    return result


def question_summary(group):
    """One question's rows averaged over its zero-shot orders, so each question counts once."""
    return {"correct": np.mean([r["correct"] for r in group]),
            "cost": np.mean([r["cost"] for r in group]),
            "saved": np.mean([r["saved_fraction"] for r in group]),
            "call_counts": {unit: np.mean([r["call_counts"][unit] for r in group])
                            for unit in ("solver_calls", "total_calls")},
            "full_call_counts": {unit: np.mean([r["full_call_counts"][unit] for r in group])
                                 for unit in ("solver_calls", "total_calls")},
            "savings_by_stop": {unit: {stop: np.mean([
                (r["full_call_counts"][unit] - r["call_counts"][unit])
                if savings_stop_category(r) == stop else 0. for r in group])
                for stop in ("correct_max", "wrong_max", "other")}
                for unit in ("solver_calls", "total_calls")},
            "has_max": group[0]["has_max"], "any_correct": group[0]["any_correct"],
            "exact_max": np.mean([r["exact_max"] for r in group]) if group[0]["has_max"] else None,
            "predicted_max": np.mean([r["reason"] == "predicted_max" for r in group]),
            "max_recognized": np.mean([r.get("first_max_recognition") is not None or
                                       r["reason"] == "predicted_max" for r in group]),
            "wrong_predicted_max": np.mean([r["reason"] == "predicted_max" and not r["exact_max"] for r in group]),
            "false_max": np.mean([r["reason"] == "predicted_max" for r in group])
            if not group[0]["has_max"] else None}


def policy_summary(rows):
    if not rows:
        return None
    by_question = {}
    for row in rows:
        by_question.setdefault(row["uid"], []).append(row)
    return summarize_questions([question_summary(group) for group in by_question.values()],
                               Counter(r["reason"] for r in rows))


def summarize_questions(means, reasons):
    """A method's report from its per-question means (question_summary) and stop-reason counts."""
    if not means:
        return None
    with_max = [m for m in means if m["has_max"]]
    without_max = [m for m in means if not m["has_max"]]
    costs = np.asarray([m["cost"] for m in means])
    return {"n": len(means), "top1_correct": float(np.mean([m["correct"] for m in means])),
            "exact_max_recovery": (float(np.mean([m["exact_max"] for m in with_max]))
                                   if with_max else None),
            "max_present_n": len(with_max), "no_max_n": len(without_max),
            "no_max_top1_correct": (float(np.mean([m["correct"] for m in without_max]))
                                    if without_max else None),
            "no_max_false_predicted_max": (float(np.mean([m["false_max"] for m in without_max]))
                                           if without_max else None),
            "max_recognition_rate": float(np.mean([m["max_recognized"] for m in means])),
            "mean_cost": float(costs.mean()), "median_cost": float(np.median(costs)),
            "p90_cost": float(np.percentile(costs, 90)),
            "p95_cost": float(np.percentile(costs, 95)),
            "mean_saved_fraction": float(np.mean([m["saved"] for m in means])),
            "call_savings": call_savings_summary(means),
            "stop_reasons": dict(reasons)}


def savings_stop_category(row):
    if row["reason"] != "predicted_max":
        return "other"
    return "correct_max" if row["exact_max"] else "wrong_max"

def question_continuation(record, scores, safe, maximum, predictor, cfg, head):
    """One question's greedy continuation toward every reference goal, over all zero-shot orders."""
    full_cfg = copy.deepcopy(cfg)
    full_cfg.max_cost = None
    question_prefix, question_lost = [], []
    question_final_prefix, question_complete = [], []
    for zero_order in permutations(range(cfg.zero_shots)):
        scenario, _, scenario_safe, scenario_maximum = permute_zero_shots(
            record, scores, safe, maximum, zero_order, cfg)
        order = permuted_reference_order(scores, zero_order, cfg)
        _, full = predictor.predict(scenario, State((1 << cfg.n)-1, (1 << cfg.k)-1), full_cfg)
        state, best_prefix, lost = State(), 0, False
        while True:
            features, probabilities = predictor.predict(scenario, state)
            flags = [goal_condition(rank, order, scenario_safe, scenario_maximum, probabilities,
                                    full, state, cfg) for rank in range(cfg.n)]
            prefix = next((rank for rank, good in enumerate(flags) if not good), cfg.n)
            if prefix < best_prefix:
                lost = True
            best_prefix = max(best_prefix, prefix)
            valid = valid_actions(state, cfg)
            if not valid.any() or prefix == cfg.n:
                break
            state = advance(state, greedy_action(head, features, valid, predictor.device), cfg)
        question_prefix.append(best_prefix)
        question_lost.append(lost)
        question_final_prefix.append(prefix)
        question_complete.append(state.candidate_mask == (1 << cfg.n)-1 and
                                 state.evaluator_mask == (1 << cfg.k)-1)
    return {"reached": float(np.mean(question_prefix)), "lost": float(np.mean(question_lost)),
            "final_prefix": float(np.mean(question_final_prefix)),
            "complete": float(np.mean(question_complete)),
            "per_rank": [float(np.mean(np.asarray(question_prefix) >= rank))
                         for rank in range(1, cfg.n+1)],
            "per_rank_final": [float(np.mean(np.asarray(question_final_prefix) >= rank))
                               for rank in range(1, cfg.n+1)]}


def continuation_diagnostic(records, ids, teacher, predictor, cfg, head):
    return continuation_summary([question_continuation(records[idx], teacher["scores"][idx],
                                                       teacher["safe"][idx], teacher["maximum"][idx],
                                                       predictor, cfg, head) for idx in ids], cfg)


def continuation_summary(questions, cfg):
    """Continuation diagnostics over per-question results, each question weighted equally."""
    reached, broken = [q["reached"] for q in questions], [q["lost"] for q in questions]
    final_prefixes, completed = [q["final_prefix"] for q in questions], [q["complete"] for q in questions]
    per_rank, per_rank_final = [q["per_rank"] for q in questions], [q["per_rank_final"] for q in questions]
    return {"n": len(questions), "orders_per_question": math.factorial(cfg.zero_shots),
            "mean_rank_prefix_reached": float(np.mean(reached)) if reached else None,
            "per_rank_reach": np.mean(per_rank, axis=0).tolist() if per_rank else None,
            "mean_final_rank_prefix": float(np.mean(final_prefixes)) if final_prefixes else None,
            "per_rank_final_recognition": (np.mean(per_rank_final, axis=0).tolist()
                                           if per_rank_final else None),
            "all_goals_recognized_rate": (float(np.mean([r[-1] for r in per_rank_final]))
                                          if per_rank_final else None),
            "acquisition_complete_rate": float(np.mean(completed)) if completed else None,
            "prior_goal_loss_rate": float(np.mean(broken)) if broken else None,
            "questions_with_prior_goal_lost": int(np.count_nonzero(broken))}

def stage_snapshot_audit(records, ids, teacher, predictor, cfg, cache=None):
    """Calibrated predictor metrics over every reachable and every presence state.

    Each question's metric sums are computed once and reused by every report that
    contains it (`cache`), as its rollouts are; no observation or logit is kept.
    """
    if not ids:
        return None
    cache = {} if cache is None else cache
    missing = [idx for idx in ids if idx not in cache]
    for name, states in (("reachable", reachable_states(cfg)), ("every", None)):
        if missing:
            views = SnapshotViews(records, missing, cfg, predictor.device, teacher, states)
            sums = snapshot_sums(predictor.model, views, predictor.temperatures, cfg)
            for idx, row in zip(missing, sums):
                cache.setdefault(idx, {})[name] = row
    metrics = summarize_snapshot_sums([cache[idx]["reachable"] for idx in ids])
    metrics["every_presence_state"] = summarize_snapshot_sums([cache[idx]["every"] for idx in ids])
    return metrics


# One line per evaluation rollout; earlier code wrote every traced row to
# final_trajectories.jsonl instead, pooled-audit questions twice.
ROLLOUT_FILE = "final_rollouts.jsonl"
LEGACY_TRAJECTORY_FILE = "final_trajectories.jsonl"


def rollout_outcome(method, row):
    """What only the models decide in an evaluation rollout; Stage 4 saves nothing else.

    Costs, call counts, correctness, and MAX recovery follow from these fields with the
    records, teacher labels, and configuration; ranked lists and traces are recomputed
    by repeating the rollout (question_rollout_audit).
    """
    return {"uid": row["uid"], "method": method, "zero_shot_order": row["zero_shot_order"],
            "candidate_mask": row["candidate_mask"], "evaluator_mask": row["evaluator_mask"],
            "selected": row["selected"], "reason": row["reason"],
            "final_max_recognized": row["final_max_recognized"]}


def question_report(record, scores, safe, maximum, predictor, cfg, heads, sink=None):
    """Roll out one question once per method and zero-shot order; keep only report inputs.

    `heads` maps each learned method (LEARNED_METHODS) to its decision head. Each rollout's
    outcome streams to `sink` as it is produced. The question keeps its per-method means
    (question_summary), and per zero-shot order its stop reason, correctness, MAX recovery,
    and saved calls for paired comparisons, plus each learned head's continuation
    diagnostic; untraced rows are discarded.
    """
    methods = {}
    for method in evaluation_methods(cfg):
        rows = question_rollouts(record, scores, safe, maximum, predictor, cfg, heads.get(method), method)
        if sink is not None:
            sink.writelines(json.dumps(rollout_outcome(method, row), allow_nan=False) + "\n"
                            for row in rows)
        methods[method] = {"means": question_summary(rows),
                           "reasons": tuple(row["reason"] for row in rows),
                           "correct": tuple(row["correct"] for row in rows),
                           "exact_max": tuple(row["exact_max"] for row in rows),
                           "saved_calls": tuple(row["full_call_counts"]["total_calls"] -
                                                row["call_counts"]["total_calls"] for row in rows)}
    return {"methods": methods,
            "continuation": {method: question_continuation(record, scores, safe, maximum, predictor, cfg,
                                                           heads[method]) for method in LEARNED_METHODS}}


# Each evaluation rollout's MAX-recognition outcome, in report order with its label.
RECOGNITION_OUTCOMES = {"correct_max_stop": "MAX stop: correct MAX",
                        "kept_stop": "MAX stop: other, top-1 kept",
                        "lost_stop": "MAX stop: other, top-1 lost",
                        "missed_max": "No MAX stop: MAX missed",
                        "no_max_no_stop": "No MAX stop: no MAX"}


def recognition_outcome(reason, exact_max, has_max, correct, full_correct):
    """A predicted-MAX stop returns the teacher's MAX or another candidate, whose top-1 is
    kept or lost against the full pool in the same zero-shot order."""
    if reason == "predicted_max":
        if exact_max:
            return "correct_max_stop"
        return "lost_stop" if correct < full_correct else "kept_stop"
    return "missed_max" if has_max else "no_max_no_stop"


def recognition_outcomes(questions, method):
    """Per outcome: share of rollouts, calls saved, and top-1 lost/gained versus the full pool.

    Each is a mean over zero-shot orders within a question, then over questions, so the
    outcomes add up to all rollouts, the mean saved calls, and the paired top-1 change.
    """
    result = {key: {"rollouts": 0., "saved_calls": 0., "top1_lost": 0., "top1_gained": 0.}
              for key in RECOGNITION_OUTCOMES}
    for question in questions:
        own, full = question["methods"][method], question["methods"]["full"]
        weight = 1 / (len(questions) * len(own["reasons"]))
        for reason, exact, correct, saved, reference in zip(
                own["reasons"], own["exact_max"], own["correct"], own["saved_calls"], full["correct"]):
            cell = result[recognition_outcome(reason, exact, own["means"]["has_max"], correct, reference)]
            cell["rollouts"] += weight
            cell["saved_calls"] += saved * weight
            cell["top1_lost"] += (reference > correct) * weight
            cell["top1_gained"] += (correct > reference) * weight
    return result


def method_comparisons(questions, cfg):
    """Paired differences, first method minus second, with bootstrap intervals over questions.

    Each method is compared with the full pool, then each learned head with each fixed order,
    then the fine-tuned head with the pretrained one, on saved calls and top-1 over all
    questions and on exact MAX over MAX-present questions.
    """
    pairs = [(name, "full") for name in evaluation_methods(cfg) if name != "full"]
    pairs += [(learned, name) for learned in LEARNED_METHODS for name in cfg.fixed_acquisition_orders]
    pairs += [("supervised", "supervised_pretrained")]
    result = {}
    for first, second in pairs:
        def difference(key, keep=lambda question: True):
            values = [np.mean(question["methods"][first][key]) - np.mean(question["methods"][second][key])
                      for question in questions if keep(question)]
            return bootstrap_difference(values, cfg.bootstrap_samples, cfg.seed) if values else None
        result[f"{first} vs {second}"] = {
            "saved_calls": difference("saved_calls"), "top1": difference("correct"),
            "exact_max": difference("exact_max", lambda question: question["methods"][first]["means"]["has_max"])}
    return result


def interval_text(item, scale=1., unit=""):
    if item is None:
        return "--"
    low, high = (scale * value for value in item["ci95"])
    return f"{scale * item['delta']:+.1f}{unit} [{low:+.1f}, {high:+.1f}]"


def paired_top1(questions, cfg):
    """Fine-tuned head minus each other method's top-1, paired within questions as in paired_interval."""
    return {baseline: bootstrap_difference(
                [np.mean([x - y for x, y in zip(question["methods"]["supervised"]["correct"],
                                                question["methods"][baseline]["correct"])])
                 for question in questions], cfg.bootstrap_samples, cfg.seed)
            for baseline in evaluation_methods(cfg) if baseline != "supervised"}


def policy_report(records, ids, teacher, predictor, cfg, heads, title, evaluated=None, sink=None,
                  snapshot_cache=None):
    """One report's acquisition comparison and its questions' report inputs, in `ids` order.

    `heads` maps each learned method to its decision head. `evaluated` maps question indices to question_report results, so reports that share
    questions (the pooled audit and each benchmark) roll every question out once per
    method and zero-shot order. New rollouts stream their outcomes to `sink`;
    `snapshot_cache` shares the snapshot diagnostic's per-question sums the same way.
    """
    section(title)
    if not ids:
        print("No eligible questions; metrics are unavailable.")
        return None, []
    evaluated = {} if evaluated is None else evaluated
    missing = [idx for idx in ids if idx not in evaluated]
    shared = f"; {len(ids) - len(missing)} reuse earlier rollouts" if len(missing) < len(ids) else ""
    print(f"Stage 4 rollouts: {len(missing)} questions x {math.factorial(cfg.zero_shots)} zero-shot "
          f"orders x {len(evaluation_methods(cfg))} methods{shared}.", flush=True)
    started = last_print = time.monotonic()
    for position, idx in enumerate(missing, 1):
        evaluated[idx] = question_report(records[idx], teacher["scores"][idx], teacher["safe"][idx],
                                         teacher["maximum"][idx], predictor, cfg, heads, sink)
        last_print = decision_progress("Stage 4 rollouts", position, len(missing), started, last_print, cfg,
                                       every=max(cfg.print_every, -(-len(missing) // 10)), seconds=300)
    questions = [evaluated[idx] for idx in ids]
    summary = {}
    for name in evaluation_methods(cfg):
        reasons = Counter()
        for question in questions:  # In row order, as policy_summary counts them.
            reasons.update(question["methods"][name]["reasons"])
        summary[name] = summarize_questions([question["methods"][name]["means"] for question in questions],
                                            reasons)
    comparisons = method_comparisons(questions, cfg)
    outcomes = {name: recognition_outcomes(questions, name) for name in summary if name != "full"}
    sizes = {group: item["n"] for group, item in summary["full"]["call_savings"].items()}
    full_calls = acquisition_call_counts(cfg.n, cfg.k, cfg)["total_calls"]
    print(f"Questions: {sizes['all']} | MAX present {sizes['max_present']} | no MAX {sizes['no_max']} "
          f"(teacher miss {sizes['teacher_miss']}, all wrong {sizes['all_wrong']})")

    print(f"\n1) API calls versus accuracy (per question; full pool = {full_calls} calls)")
    print(f"{'Method':<26} | {'API calls':>9} | {'Saved':>6} | {'Top-1':>6} | {'Exact MAX':>9} | "
          f"{'MAX stops':>9} | {'Stop precision':>14}")
    for name, item in summary.items():
        calls, stops = item["call_savings"]["all"]["total_calls"], item["call_savings"]["all"]
        exact = "--" if item["exact_max_recovery"] is None else f"{item['exact_max_recovery']:.1%}"
        precision = ("--" if stops["predicted_max_stop_precision"] is None else
                     f"{stops['predicted_max_stop_precision']:.1%}")
        print(f"{name:<26} | {calls['mean_used']:>9.1f} | {calls['saved_fraction']:>6.1%} | "
              f"{item['top1_correct']:>6.1%} | {exact:>9} | {stops['predicted_max_stop_rate']:>9.1%} | "
              f"{precision:>14}")

    print("\n2) Paired differences, first minus second (95% CI over questions)")
    print(f"{'Comparison':<48} | {'Saved calls/q':>22} | {'Top-1':>22} | {'Exact MAX':>22}")
    for pair, item in comparisons.items():
        print(f"{pair:<48} | {interval_text(item['saved_calls']):>22} | "
              f"{interval_text(item['top1'], 100, 'pp'):>22} | {interval_text(item['exact_max'], 100, 'pp'):>22}")

    groups = {"all": "All", "max_present": "MAX present", "no_max": "No MAX",
              "teacher_miss": "Teacher miss", "all_wrong": "All wrong"}
    print("\n3) API calls saved by question group (% of the group's full-pool calls)")
    print(f"{'Method':<26} | " + " | ".join(f"{label:>12}" for label in groups.values()) +
          f" | {'From no MAX':>11}")
    for name, item in summary.items():
        if name == "full":
            continue
        savings = item["call_savings"]
        cells = ["--" if savings[group]["total_calls"]["saved_fraction"] is None else
                 f"{savings[group]['total_calls']['saved_fraction']:.1%}" for group in groups]
        total = savings["all"]["total_calls"]["total_saved"]
        share = f"{(savings['no_max']['total_calls']['total_saved'] or 0.) / total:.1%}" if total else "--"
        print(f"{name:<26} | " + " | ".join(f"{cell:>12}" for cell in cells) + f" | {share:>11}")

    print("\n4) MAX recognition outcomes (share of rollouts; per question: calls saved, top-1 change vs full)")
    print(f"{'Method':<26} | {'Outcome':<27} | {'Rollouts':>8} | {'Saved/q':>7} | {'Top-1 lost':>10} | "
          f"{'Top-1 gained':>12}")
    columns = ("rollouts", "saved_calls", "top1_lost", "top1_gained")
    for name, table in outcomes.items():
        rows = [(RECOGNITION_OUTCOMES[key], table[key]) for key in RECOGNITION_OUTCOMES]
        rows.append(("Total", {column: sum(cell[column] for cell in table.values()) for column in columns}))
        for position, (label, cell) in enumerate(rows):
            print(f"{name if position == 0 else '':<26} | {label:<27} | {cell['rollouts']:>8.1%} | "
                  f"{cell['saved_calls']:>7.1f} | {cell['top1_lost']:>10.1%} | {cell['top1_gained']:>12.1%}")

    first_state = full_calls - acquisition_call_counts(1, 0, cfg)["total_calls"]
    print(f"\nNotes: API calls = candidate generations + measurement solves + their graders; retrieval is "
          f"excluded. Each question counts once: its {math.factorial(cfg.zero_shots)} zero-shot orders are averaged.")
    print("Rollouts stop at the first predicted MAX (the deployed wrapper may continue). MAX = the teacher's "
          "top-ranked candidate when it is correct; every MAX stop on a no-MAX question is false.")
    print("Teacher miss: a correct candidate ranks below a wrong top. All wrong: no candidate is correct, so a "
          f"stop costs no accuracy (stopping at the first state would save {first_state} calls).")
    print("Exact MAX = share of MAX-present questions whose returned candidate is the MAX. MAX stops = share "
          "of rollouts stopped at a predicted MAX; stop precision = share of those stops returning the MAX.")
    print("Top-1 lost/gained pair each rollout with the full pool in the same zero-shot order; "
          f"95% CI from a {cfg.bootstrap_samples}-sample bootstrap over questions.")
    return {"policies": summary, "comparisons": comparisons, "recognition_outcomes": outcomes,
            "cost_unit": cfg.cost_unit,
            "fixed_acquisition_sequences": fixed_acquisition_sequences(cfg),
            "learned_continues_after_max": False,
            "inference_continues_after_max_default": cfg.continue_after_max,
            "early_stop_rule": {"reason": "predicted_max", "threshold": cfg.recognition_threshold,
                                "signals": ["global_MAX", "global_SAFE", "selected_MAX", "selected_SAFE"]},
            "call_accounting": "Recorded generation, baseline/cross measurement solves, and their grading calls; excludes retrieval and target-answer grading.",
            "snapshot_diagnostics": stage_snapshot_audit(records, ids, teacher, predictor, cfg,
                                                         snapshot_cache),
            "continuation": {method: continuation_summary([question["continuation"][method]
                                                           for question in questions], cfg)
                             for method in LEARNED_METHODS}}, questions


def question_rollout_audit(records, idx, teacher, predictor, cfg, heads, out, benchmark):
    """Rebuild one question's traced Stage 4 rollouts, checked against its saved outcomes.

    Stage 4 saves only rollout_outcome fields. Traces (every prediction check with its
    MAX/SAFE signals, every acquisition and its cost), ranked lists, costs, and
    correctness are deterministic, so they are recomputed here (use the reporting
    device; another device may round probabilities differently and fail the check).
    """
    record, saved = records[idx], {}
    needle = json.dumps(record.uid)
    with (Path(out) / ROLLOUT_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            if needle in line:
                outcome = json.loads(line)
                if outcome["uid"] == record.uid:
                    saved[outcome["method"], tuple(outcome["zero_shot_order"])] = outcome
    rows = []
    for method in evaluation_methods(cfg):
        for row in question_rollouts(record, teacher["scores"][idx], teacher["safe"][idx],
                                     teacher["maximum"][idx], predictor, cfg, heads.get(method), method, trace=True):
            if saved.pop((method, tuple(row["zero_shot_order"])), None) != rollout_outcome(method, row):
                raise ValueError(f"Recomputed Stage 4 rollout of {record.uid} ({method}, zero-shot order "
                                 f"{row['zero_shot_order']}) differs from {ROLLOUT_FILE}. Rerun report() "
                                 "after changing evaluation settings, and audit on the reporting device.")
            rows.append({"benchmark": benchmark, "method": method, **row})
    if saved:
        raise ValueError(f"{ROLLOUT_FILE} has rollouts of {record.uid} that this configuration does not "
                         f"evaluate: {sorted(saved)}. Rerun report() after changing evaluation settings.")
    return {"uid": record.uid, "rows": rows}


def hub_access_token():
    """A Hub token from the environment, Kaggle Secrets, or an existing login; never printed."""
    for name in ("HF_SYNC_TOKEN", "HF_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
    except Exception:
        client = None
    for name in ("HF_SYNC_TOKEN", "HF_TOKEN") if client is not None else ():
        try:
            token = client.get_secret(name)
        except Exception:
            continue
        if token:
            return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def hub_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("hf_repo needs the huggingface_hub package: %pip install -q huggingface_hub") from exc
    return HfApi(token=hub_access_token())


def folder_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


# What the three training phases build, relative to output_dir: the compact records, the teacher
# (folds and out-of-fold labels), the frozen snapshot predictor, and Stage 3's labels together with the
# pretrained head that the fine-tuning labels were rolled out from. The rest (the splits and data audit
# that Stage 0 rewrites each session, head training progress, the fine-tuned head, the inference
# bundle, reports) is run output for the run mirror.
DATASET_SCHEMA = 1
DATASET_MANIFEST = "dataset_manifest.json"
DATASET_STAGING = ".dataset_download"    # Downloads land here first; never uploaded.
DATASET_VERSION = r"[0-9a-f]{12}"
DATASET_PATTERNS = ("compact_records.pt", "teacher_fold_*.pt", "teacher_final.pt", "teacher_completed.pt",
                    "snapshot_completed.pt", "decision_head_pretrained.pt",
                    "decision_dataset_manifest.json", "decision_label_coverage.json",
                    "decision_labels/*", DATASET_MANIFEST)
# Reports that Stage 3 writes beside the labels; present in a complete dataset, not hashed.
DATASET_SUPPORT_FILES = ("decision_dataset_manifest.json", "decision_label_coverage.json")


def in_dataset(relative):
    """Whether a path inside output_dir (posix, relative) belongs to the dataset store."""
    return any(fnmatch.fnmatch(relative, pattern) for pattern in DATASET_PATTERNS)


def dataset_artifacts():
    """The files a complete dataset must hold. Each is written once, so the manifest hashes it."""
    return ["compact_records.pt", "teacher_completed.pt", "snapshot_completed.pt", "decision_head_pretrained.pt",
            *[f"decision_labels/{phase}_{role}.pt" for phase in HEAD_PHASES for role in ("policy", "dev")]]


def input_sources(paths):
    """Each input log by name, size, and content: its directory and mtime change between sessions."""
    return [{"path": str(Path(p).resolve()), "name": Path(p).name, "bytes": Path(p).stat().st_size,
             "sha256": file_digest(p)} for p in paths]


def dataset_construction(contract_cfg, sources):
    """What a dataset version is hashed from: the contract's settings, the revision, and the logs' content.

    `sources` lists name, size, and SHA-256 of each log in order, or is None while the logs are unknown.
    """
    def canonical(value):  # 0 and 0. are one setting; they must be one version.
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    return json.loads(json.dumps(canonical({"schema_version": DATASET_SCHEMA,
                                            "implementation_revision": PIPELINE_REVISION,
                                            "config": contract_cfg, "sources": sources})))


def dataset_version(construction):
    payload = json.dumps(construction, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def dataset_differences(old, new):
    """Names of the settings that make two dataset constructions different versions."""
    names = [key for key in ("schema_version", "implementation_revision") if old.get(key) != new.get(key)]
    old_cfg, new_cfg = old.get("config", {}), new.get("config", {})
    names += [f"config.{key}" for key in sorted(set(old_cfg) | set(new_cfg)) if old_cfg.get(key) != new_cfg.get(key)]
    if old.get("sources") != new.get("sources"):
        names.append("the input logs")
    return names


class DatasetStore:
    """The datasets that the three training phases build, versioned on the Hub under a dataset id.

    A dataset is the output of the phases that precede head training: compact records and splits,
    the teacher's out-of-fold labels, the frozen snapshot predictor, and Stage 3's labels
    (`DATASET_PATTERNS`). It is uploaded to `<repo>/<dataset_id>/<version>/`, where the version is a
    hash of everything that decides its content (the contract's settings, the implementation
    revision, and the input logs' content). The same id with another construction is therefore
    another version beside the first, and a notebook that selects the id finds the version its
    configuration needs: a complete one is downloaded and verified, never rebuilt; a partial one
    resumes; only an absent one is built. Uploads go out as the build progresses, so a session
    killed mid-build loses little. Run one session per dataset id and version.
    """
    def __init__(self, out, repo, dataset_id, interval_minutes, resume=True, api=None):
        self.out, self.repo, self.id = Path(out), repo, dataset_id
        self.interval, self.resume = interval_minutes, resume
        self.api = hub_api() if api is None else api
        self.version = self.construction = self.remote = self.channel = None
        self.others, self.state, self.pulled = {}, None, False

    @property
    def prefix(self):
        return f"{self.id}/{self.version}"

    def hub(self, action, call):
        """A Hub read with retries. An outage is never mistaken for an absent dataset."""
        error = None
        for attempt in range(3):
            if attempt:
                time.sleep(5 * 2 ** (attempt - 1))
            try:
                return call()
            except Exception as exc:  # Network, rate limit, or credentials.
                error = exc
        raise RuntimeError(f"Could not {action} in {self.repo}: {error}. Nothing is rebuilt while it is unknown "
                           "whether the Hub already holds the dataset; check the token (HF_SYNC_TOKEN or "
                           "HF_TOKEN), the repository id, and Kaggle Internet, then run again.") from error

    def catalog(self):
        """Every uploaded version of this dataset id, as {version: manifest}."""
        if not self.hub("check whether the repository exists",
                        lambda: self.api.repo_exists(self.repo, repo_type="dataset")):
            return {}
        staging = self.out / DATASET_STAGING
        listed = [name for name in self.hub("list the dataset versions", lambda: self.api.list_repo_files(
            self.repo, repo_type="dataset")) if re.fullmatch(rf"{re.escape(self.id)}/{DATASET_VERSION}/{DATASET_MANIFEST}", name)]
        if not listed:
            return {}

        def fetch():
            # A fresh folder per attempt: a half-finished attempt must not pass for a download.
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            return self.api.snapshot_download(repo_id=self.repo, repo_type="dataset", local_dir=str(staging),
                                              allow_patterns=listed)
        try:
            self.hub("download the dataset manifests", fetch)
            found = {}
            for name in listed:
                path = staging / name
                if not path.is_file():
                    raise RuntimeError(f"{name} is listed in {self.repo} but did not download; nothing is "
                                       "rebuilt while the Hub's state is unknown. Run again.")
                found[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
            return found
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def resolve(self, construction, names):
        """Pick this configuration's version and return its construction, input logs included.

        With the logs at hand the version is a hash of their content. Without them (a new session
        that did not attach the logs), the one complete version of this id built with this
        configuration and these log names is used, and its recorded logs stand in for them.
        """
        catalog = self.catalog()
        if construction["sources"] is not None:
            self.version = dataset_version(construction)
        else:
            def same(manifest):
                recorded = manifest.get("construction", {})
                return bool(manifest.get("complete")) and [s["name"] for s in recorded.get("sources") or []] == names \
                    and not dataset_differences(recorded, {**construction, "sources": recorded.get("sources")})
            matches = [version for version, manifest in catalog.items() if same(manifest)]
            if len(matches) != 1:
                raise ValueError(
                    f"The input logs {names} are not available, and {self.repo} holds "
                    f"{len(matches)} complete versions of dataset {self.id!r} for this configuration "
                    f"({sorted(matches) or 'none'}), not exactly one. Attach the logs so the version is identified "
                    "by their content, or keep the settings the dataset was built with.")
            self.version = matches[0]
            construction = {**construction, "sources": catalog[self.version]["construction"]["sources"]}
        if catalog.get(self.version) and dataset_differences(catalog[self.version].get("construction", {}), construction):
            raise ValueError(f"The manifest of dataset {self.id!r} version {self.version} does not describe the "
                             "construction it is stored under; the upload is corrupt.")
        self.construction, self.remote = construction, catalog.get(self.version)
        self.others = {version: dataset_differences(manifest.get("construction", {}), construction)
                       for version, manifest in catalog.items() if version != self.version}
        self.channel = OutputSync(self.out, self.repo, self.interval, self.api, path_in_repo=self.prefix,
                                  allow_patterns=list(DATASET_PATTERNS),
                                  name=f"Dataset {self.id} version {self.version} in {self.repo}")
        return construction

    def local_manifest(self):
        path = self.out / DATASET_MANIFEST
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def local_files(self):
        found = (path.relative_to(self.out).as_posix() for path in self.out.rglob("*") if path.is_file())
        return sorted(rel for rel in found if in_dataset(rel) and not rel.endswith(".tmp"))

    def foreign_files(self):
        """Files in the output folder that are neither dataset files nor a download in progress."""
        found = (path.relative_to(self.out).as_posix() for path in self.out.rglob("*") if path.is_file())
        return sorted(rel for rel in found if not in_dataset(rel) and not rel.startswith(f"{DATASET_STAGING}/"))

    def manifest(self, complete=False, artifacts=None, info=None):
        return {"schema_version": DATASET_SCHEMA, "dataset_id": self.id, "version": self.version,
                "implementation_revision": PIPELINE_REVISION, "construction": self.construction,
                "complete": complete, "artifacts": artifacts or {}, "info": info or {}}

    def verify(self, manifest):
        """Every hashed artifact of a complete dataset must be here and byte-identical."""
        for relative, meta in manifest["artifacts"].items():
            path = self.out / relative
            if not path.is_file():
                raise ValueError(f"Dataset {self.id!r} version {self.version} is missing {relative} in {self.out}.")
            if path.stat().st_size != meta["bytes"] or file_digest(path) != meta["sha256"]:
                raise ValueError(
                    f"{path} differs from dataset {self.id!r} version {self.version} on {self.repo}: it was "
                    "rebuilt or changed here, or the download is damaged. Use an empty output_dir to download "
                    "the dataset, or another dataset_id to keep this build as a new dataset.")

    def download(self):
        """Fetch the version's files that the output folder lacks; existing files are kept."""
        prefix = self.prefix + "/"
        remote = [name[len(prefix):] for name in self.hub("list the dataset files", lambda: self.api.list_repo_files(
            self.repo, repo_type="dataset")) if name.startswith(prefix)]
        wanted = [rel for rel in remote if in_dataset(rel) and not rel.endswith(".tmp")]
        missing = [rel for rel in wanted if not (self.out / rel).exists()]
        if not missing:
            return 0
        staging = self.out / DATASET_STAGING

        def fetch():
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)
            return self.api.snapshot_download(repo_id=self.repo, repo_type="dataset", local_dir=str(staging),
                                              allow_patterns=[prefix + "*"])
        try:
            self.hub("download the dataset", fetch)
            moved = 0
            for relative in missing:
                source, target = staging / self.id / self.version / relative, self.out / relative
                if not source.is_file():
                    raise RuntimeError(f"{relative} is listed in dataset {self.id!r} version {self.version} on "
                                       f"{self.repo} but did not download.")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)  # The staging folder is on the same filesystem.
                moved += 1
            return moved
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def pull(self):
        """Bring the output folder in line with the Hub: download what exists, verify what is complete."""
        local = self.local_manifest()
        if local is not None and (local.get("dataset_id"), local.get("version")) != (self.id, self.version):
            raise ValueError(f"The output folder holds dataset {local.get('dataset_id')!r} version "
                             f"{local.get('version')}, not {self.id!r} version {self.version}. Choose a new "
                             "output_dir or the matching dataset_id.")
        held = [rel for rel in self.local_files() if rel != DATASET_MANIFEST]
        name = f"Dataset {self.id!r} version {self.version}"
        if self.remote is None:
            self.state = "adopted" if held else "new"
            print(f"{name}: not in {self.repo} yet. "
                  + (f"The {len(held)} dataset files already in {self.out} are uploaded under it, not rebuilt; "
                     "missing parts are built and uploaded as they finish." if held else
                     "It is built and uploaded as it progresses."), flush=True)
            if local is None:
                atomic_json(self.out / DATASET_MANIFEST, self.manifest())
        else:
            if not self.resume:
                raise ValueError(f"{name} is already in {self.repo}. Set resume=True to reuse it, or choose "
                                 "another dataset_id.")
            moved = self.download()
            if self.remote.get("complete"):
                self.verify(self.remote)
                atomic_json(self.out / DATASET_MANIFEST, self.remote)
                self.state = "complete"
                self.pulled = True
                size = sum((self.out / rel).stat().st_size for rel in self.remote["artifacts"])
                print(f"{name}: complete in {self.repo}; {moved} files downloaded, {len(self.remote['artifacts'])} "
                      f"artifacts ({size / 1024**2:.1f} MiB) verified. Nothing is rebuilt.", flush=True)
            else:
                self.state = "partial"
                if self.local_manifest() is None:
                    atomic_json(self.out / DATASET_MANIFEST, self.manifest())
                print(f"{name}: partly built in {self.repo}; {moved} files downloaded. The build resumes from "
                      "them.", flush=True)
        if self.others:
            print("Other versions under this id stay untouched: " + "; ".join(
                f"{version} differs in {', '.join(changed) or 'nothing recorded'}"
                for version, changed in sorted(self.others.items())) + ".", flush=True)
        return self.state

    def finish(self, info=None):
        """Seal the dataset: hash its artifacts, mark the manifest complete, and upload it."""
        if self.state == "complete":
            return  # Pulled complete and verified: the stored version is read-only here.
        local = self.local_manifest()
        if not (local or {}).get("complete"):
            lacking = [rel for rel in (*dataset_artifacts(), *DATASET_SUPPORT_FILES) if not (self.out / rel).is_file()]
            if lacking:
                raise RuntimeError(f"Dataset {self.id!r} version {self.version} is not complete: {lacking} "
                                   f"missing from {self.out}.")
            artifacts = {rel: {"bytes": (self.out / rel).stat().st_size, "sha256": file_digest(self.out / rel)}
                         for rel in dataset_artifacts()}
            atomic_json(self.out / DATASET_MANIFEST,
                        {**self.manifest(True, artifacts, info),
                         "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        # One commit carries the sealed manifest with any file the Hub still lacks.
        self.channel.push(f"Dataset {self.id} version {self.version}: complete", force=True, required=True)
        self.state = "complete"

    def describe(self):
        return "Dataset store: read-only (complete version downloaded)." if self.state == "complete" and self.pulled             else self.channel.describe()


class OutputSync:
    """Mirror output_dir to a private Hugging Face dataset repository.

    Kaggle keeps /kaggle/working only while a session lives. Every stage
    already completes through atomic files in output_dir, so uploading that
    folder makes a run resumable across sessions: a new session pulls the
    folder, Stages 0-2 load their completed checkpoints, and Stage 3 continues
    from its completed questions and epochs. Uploads skip unchanged files by
    content hash, never include `.tmp` files, and remove per-question label
    files from the mirror once a role is merged. The traced Stage 4 rows of
    earlier code (final_trajectories.jsonl, derived and superseded) are never
    downloaded and leave the mirror at the next upload. Run one session per repo.

    The dataset store reuses this class for one folder of a shared repository: `path_in_repo`
    puts its files under `<dataset_id>/<version>/`, and `allow_patterns` limits it to the
    files that make up the dataset. `ignore_patterns` keeps those files out of the run mirror.
    """
    def __init__(self, out, repo, interval_minutes, api=None, *, path_in_repo=None, allow_patterns=None,
                 ignore_patterns=(), name=None):
        self.out, self.repo, self.interval = Path(out), repo, interval_minutes * 60
        self.api = hub_api() if api is None else api
        self.path_in_repo, self.allow_patterns = path_in_repo, allow_patterns
        self.ignore_patterns = ["*.tmp", f"{DATASET_STAGING}/*", *ignore_patterns]
        self.name = name or f"Mirror {repo}"
        self.last = time.monotonic()
        self.uploads, self.last_message, self.last_error = 0, None, None

    def pull(self):
        """Download the mirrored run into an empty output folder; a filled folder is current."""
        if self.out.exists() and any(self.out.iterdir()):
            return "local"
        if not self.api.repo_exists(self.repo, repo_type="dataset"):
            return "new"
        self.api.snapshot_download(repo_id=self.repo, repo_type="dataset", local_dir=str(self.out),
                                   ignore_patterns=[".gitattributes", LEGACY_TRAJECTORY_FILE])
        shutil.rmtree(self.out / ".cache", ignore_errors=True)  # Download bookkeeping only.
        return "pulled"

    def push(self, message, *, force=False, required=False):
        """Upload changed files; inside loops only once per interval, at stage ends always.

        A failed upload never discards local work: periodic failures warn and
        retry at the next sync point; a required upload raises so a bad token
        or repository is visible at Stage 0 rather than after hours of training.
        """
        if not force and time.monotonic() - self.last < self.interval:
            return False
        error = None
        # Only a shared-repository channel needs these arguments; the run mirror's call is unchanged.
        scope = {key: value for key, value in (("path_in_repo", self.path_in_repo),
                                               ("allow_patterns", self.allow_patterns)) if value}
        for attempt in range(3):
            if attempt:
                time.sleep(5 * 2 ** (attempt - 1))
            try:
                self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
                self.api.upload_folder(folder_path=str(self.out), repo_id=self.repo, repo_type="dataset",
                                       commit_message=message, ignore_patterns=list(self.ignore_patterns),
                                       delete_patterns=["decision_labels/*/*.pt", LEGACY_TRAJECTORY_FILE],
                                       **scope)
            except Exception as exc:  # Network, rate limit, or credentials; local files stay complete.
                error = exc
                continue
            self.last, self.last_message, self.last_error = time.monotonic(), message, None
            self.uploads += 1
            return True
        self.last, self.last_error = time.monotonic(), error
        if required:
            raise RuntimeError(f"Upload to {self.repo} failed: {error}. Check the token (HF_SYNC_TOKEN or "
                               "HF_TOKEN), the repository id, and Kaggle Internet; completed files remain "
                               f"in {self.out}.") from error
        print(f"WARNING: upload to {self.repo} failed ({error}); retrying at the next sync point. "
              f"Completed files remain in {self.out}.", flush=True)
        return False

    def describe(self):
        return (f"{self.name}: {self.uploads} uploads; last upload: {self.last_message or 'none'}; "
                f"last error: {self.last_error or 'none'}.")


class SyncGroup:
    """Several mirrors of one output folder, each with its own files, repository, and timing."""
    def __init__(self, *syncs):
        self.syncs = [sync for sync in syncs if sync is not None]

    def push(self, message, *, force=False, required=False):
        # Every mirror is attempted in turn; a required upload raises at the first one that fails.
        return any([sync.push(message, force=force, required=required) for sync in self.syncs])

    def describe(self):
        return " ".join(sync.describe() for sync in self.syncs)


def contract_identity(contract):
    """Input logs are identified by filename, size, and content; directory and mtime vary."""
    identity = json.loads(json.dumps(contract))
    identity["sources"] = [{key: source.get(key) for key in ("name", "bytes", "sha256")}
                           for source in identity.get("sources", [])]
    return identity


class Workflow:
    """Notebook stages; completed checkpoints resume, interrupted stages restart safely."""
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.validate()
        torch.set_num_threads(cfg.cpu_threads)
        seed_everything(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        # With a dataset store its files live there; the run mirror carries the training output.
        run_sync = (OutputSync(self.out, cfg.hf_repo, cfg.hf_sync_minutes,
                               ignore_patterns=DATASET_PATTERNS if cfg.dataset_id else ())
                    if cfg.hf_repo else None)
        if run_sync is not None:
            state = run_sync.pull()
            print({"local": f"Mirror {cfg.hf_repo}: {self.out} already holds files; they are used as is.",
                   "new": f"Mirror {cfg.hf_repo}: no uploaded run yet; this run is uploaded as it progresses.",
                   "pulled": f"Mirror {cfg.hf_repo}: downloaded {folder_size(self.out) / 1024**2:.1f} MiB "
                             f"into {self.out}; completed stages resume."}[state], flush=True)
        # Logs are identified by the sources below, not by the configured paths.
        contract_cfg = {key: value for key, value in asdict(cfg).items() if key not in NON_CONTRACT_SETTINGS}
        paths = [cfg.train_file] + cfg.test_files
        self.datasets = None
        if cfg.dataset_id:
            self.datasets = DatasetStore(self.out, cfg.dataset_repo, cfg.dataset_id, cfg.hf_sync_minutes, cfg.resume)
            # A log's filename is its benchmark identity; its directory and mtime are not.
            sources = input_sources(paths) if all(Path(p).is_file() for p in paths) else None
            construction = self.datasets.resolve(
                dataset_construction(contract_cfg, None if sources is None else
                                     [{key: source[key] for key in ("name", "bytes", "sha256")} for source in sources]),
                [Path(p).name for p in paths])
            if sources is None:
                print("Input logs are not attached; the dataset's recorded logs identify this run.", flush=True)
                sources = [{"path": str(Path(p)), **source} for p, source in zip(paths, construction["sources"])]
        else:
            sources = input_sources(paths)
        contract = {"schema_version": 4, "implementation_revision": PIPELINE_REVISION,
                    "config": contract_cfg, "sources": sources}
        path = self.out / "contract.json"
        existing = path.exists()
        if existing:
            old = json.loads(path.read_text(encoding="utf-8"))
            # Round-trip converts tuples to JSON lists consistently.
            if contract_identity(old) != contract_identity(contract):
                raise ValueError("Output folder belongs to a different config/data contract. Choose a new output_dir.")
            if not cfg.resume:
                raise ValueError("Output folder already contains this run. Set resume=True or choose a new folder.")
            # Before Stage 0 writes its reports: a resumed run that filled its
            # disk regains the space held by derived caches and partial writes.
            reclaimed = reclaim_derived_stage3_files(self.out)
            if reclaimed:
                print(f"Reclaimed {reclaimed / 1024**2:.2f} MiB of incomplete writes in {self.out} "
                      "before resuming.", flush=True)
        # Files that no contract explains are never adopted, dataset files included.
        if not existing and any(self.out.iterdir()):
            raise ValueError("Use an empty output folder; existing unrecognized files will not be overwritten.")
        if self.datasets is not None:
            self.datasets.pull()
        if not existing:
            atomic_json(path, contract)
        # The dataset's mirror comes first: a legacy run mirror's per-question label files leave it
        # only after the dataset store has uploaded them.
        # A complete version pulled from the Hub is read-only: no channel may rewrite it.
        stored = None if self.datasets is None or self.datasets.pulled else self.datasets.channel
        mirrors = [mirror for mirror in (stored, run_sync)
                   if mirror is not None]
        self.sync = None if not mirrors else mirrors[0] if len(mirrors) == 1 else SyncGroup(*mirrors)
        self.records, self.splits = None, None

    def prepare(self):
        section(f"DATA AUDIT | device={self.device} | {self.cfg.n} candidates, {self.cfg.k} evaluators")
        # Names the embedded code, so a stale notebook is visible before any stage trains.
        print(f"Implementation revision {PIPELINE_REVISION}; Stage 3 labels every presence state with its "
              "cheapest routes to the next goal and pretrains on them (Stage 3a), then fine-tunes on the "
              "route its own continuations make cheapest (Stage 3b). Compact labels "
              f"(schema {DECISION_LABEL_SCHEMA}: a bit mask per state/order case for 3a, one byte for 3b) "
              "are bound to the input logs' content rather than their paths. Stage 4 rolls out each "
              f"question once and saves compact outcomes to {ROLLOUT_FILE}, not trajectories, for both "
              "heads. Stages 1-2 train on every "
              f"presence state under every zero-shot order ({len(presence_states(self.cfg.zero_shots, self.cfg.k)):,} "
              f"x {math.factorial(self.cfg.zero_shots)} views per question), assembled on the "
              "device and never stored; each keeps the checkpoint with the lowest development "
              "loss (AP is only a reported metric).", flush=True)
        cache_path = self.out / "compact_records.pt"
        if self.cfg.resume and cache_path.exists():
            cache = load_checkpoint(cache_path)
            self.records = [Record(**r) for r in cache["records"]]
            self.audit = cache["audit"]
            print("Using compact cached records (source size/sha256 contract verified).")
        else:
            self.records, self.audit = load_records(self.cfg)
            atomic_torch(cache_path, {"records": [asdict(r) for r in self.records], "audit": self.audit})
        self.splits = split_records(self.records, self.cfg)
        print(f"{'Dataset':<32} | {'Read':>6} | {'Eligible':>8} | {'Excluded':>8} | {'Oracle':>7}")
        print("-"*76)
        for name, counts in self.audit.items():
            total, eligible = counts.get("total", 0), counts.get("eligible", 0)
            oracle = counts.get("positive_pool", 0)/eligible if eligible else 0
            print(f"{name:<32} | {total:6d} | {eligible:8d} | {total-eligible:8d} | {oracle:6.1%}")
            excluded = {k:v for k,v in counts.items() if k not in {"total", "eligible", "positive_pool"} and v}
            if excluded:
                print("  Exclusions: " + ", ".join(f"{k}={v}" for k,v in sorted(excluded.items())))
        print("Roles: " + ", ".join(f"{k}={len(self.splits[k])}" for k in ("supervised", "policy", "dev", "audit")))
        print("All three stages train on the same supervised questions; policy is an alias.")
        print("Dev/audit source: " + ("pooled test_files, shared for model selection and reporting."
              if self.cfg.use_test_files_for_dev_and_audit else "separate held-out training-file partitions."))
        print("Strict complete-pool analysis; exclusions are reported, never silently counted as wrong.")
        print("CCS uses recorded rates/configured attempts; per-trial API validity is not inferred from rates.")
        print("Exact normalized-text duplicates removed; near-duplicate/corpus contamination requires a separate audit.")
        atomic_json(self.out / "data_audit.json", self.audit)
        atomic_json(self.out / "split_manifest.json", {"data_sha256": data_digest(self.records),
                    "evaluation_protocol": split_protocol(self.cfg, self.splits),
                    "roles": {k: [self.records[i].uid for i in ids] for k,ids in self.splits.items()},
                    "groups": {r.uid: r.group for r in self.records}})
        # The first upload is required, so a bad token or repository fails here, not hours later.
        self.push("Stage 0: data audit, compact records, and split", required=True)
        return self

    def push(self, message="manual upload", *, force=True, required=False):
        """Upload the output folder to hf_repo now; a no-op without a mirror."""
        return False if self.sync is None else self.sync.push(message, force=force, required=required)

    def build_datasets(self):
        """Make every dataset the training phases consume, once: records, teacher labels, predictor, Stage 3 labels.

        Each dataset comes from the phase before it: the teacher's out-of-fold labels define the
        SAFE/MAX targets, the frozen snapshot predictor defines every Stage 3 goal, and Stage 3b's
        labels are rolled out with the pretrained head. So this builds Stages 1, 2 and 3a and then
        labels 3b; it stops short of fine-tuning, which only reads the labels. With `dataset_id`, a
        complete version on the Hub was downloaded and verified when the workflow was created and
        the calls below only load it; a partial version resumes; the finished dataset is sealed
        and uploaded.
        """
        if self.records is None:
            self.prepare()
        store = self.datasets
        section("DATASETS | " + (f"{self.cfg.dataset_id!r} version {store.version} ({store.state})" if store is not None
                                 else "no dataset_id: the datasets stay in this run's output folder"))
        self.train_teacher()
        self.train_snapshot()
        self.train_pretrained_head()
        section("DATASETS | Stage 3b labels, rolled out with the pretrained head")
        labels, coverage, _ = finetune_labels(self.records, self.splits, self.teacher, self.predictor,
                                              self.pretrained_checkpoint, self.cfg, self.device, self.out,
                                              self.sync)
        del labels
        if store is not None:
            store.finish({"records": len(self.records),
                          "questions": {role: len(self.splits[role]) for role in ("supervised", "dev", "audit")},
                          "label_coverage": coverage})
            print(f"Dataset {self.cfg.dataset_id!r} version {store.version} is complete in {store.repo}. A later "
                  "session with the same dataset_id and configuration downloads it instead of building it.",
                  flush=True)
        return self

    def train_teacher(self):
        if self.records is None:
            self.prepare()
        section("STAGE 1 | Masked joint correctness model and out-of-fold ranking")
        path = self.out / "teacher_completed.pt"
        if self.cfg.resume and path.exists():
            self.teacher = load_checkpoint(path)
            print("Loaded completed teacher stage.")
        else:
            self.teacher = fit_teachers(self.records, self.splits, self.cfg, self.device, self.out, self.sync)
            atomic_torch(path, self.teacher)
        self.heuristics = select_heuristics(self.records, self.splits["dev"], self.cfg)
        atomic_json(self.out / "heuristics_selected_on_dev.json", self.heuristics)
        ids = self.splits["supervised"]
        empty = int((self.teacher["safe"][ids].sum(1) == 0).sum())
        print(f"OOF labels: {len(ids)} supervised questions; empty SAFE/MAX: {empty}/{len(ids)}.")
        print("Reference rankings and SAFE/MAX labels come from complete-state predictions; "
              "partial states only train the models.")
        teacher_report(self.records, self.splits["dev"], self.cfg, self.teacher["scores"],
                       self.heuristics, "TEACHER DEVELOPMENT REPORT (metrics only; checkpoints are selected "
                       "on development loss)")
        self.push("Stage 1 complete: teacher folds and out-of-fold labels")
        return self

    def train_snapshot(self):
        if not hasattr(self, "teacher"):
            self.train_teacher()
        section(f"STAGE 2 | Snapshot ResNet ({self.cfg.input_dim} inputs, {3*self.cfg.n+2} prediction outputs)")
        path = self.out / "snapshot_completed.pt"
        if self.cfg.resume and path.exists():
            checkpoint = load_checkpoint(path)
            print("Loaded completed snapshot stage.")
        else:
            checkpoint = fit_snapshot(self.records, self.splits, self.teacher, self.cfg, self.device)
            atomic_torch(path, checkpoint)
        self.predictor = FrozenPredictor(checkpoint, self.cfg, self.device)
        print("Encoder and prediction heads frozen. Development temperature calibration: "
              + ("enabled." if self.cfg.calibrate else "disabled."))
        print(f"Recognition threshold for supervised goals: {self.cfg.recognition_threshold:.2f}.")
        self.push("Stage 2 complete: frozen snapshot predictor")
        return self

    def dataset_fingerprint(self):
        return decision_dataset_fingerprint(self.records, self.splits, self.teacher, self.predictor, self.cfg)

    def load_head(self, checkpoint):
        head = SupervisedActionHead(self.cfg).to(self.device)
        head.load_state_dict(checkpoint["weights"])
        return head.eval()

    def train_pretrained_head(self):
        """Stage 3a: pretrain on every cheapest first step of every state."""
        if not hasattr(self, "predictor"):
            self.train_snapshot()
        section("STAGE 3a | Pretraining the frozen-encoder head on every cheapest first step")
        path = self.out / "decision_head_pretrained.pt"
        fingerprint = self.dataset_fingerprint()
        if self.cfg.resume and path.exists():
            checkpoint = load_checkpoint(path)
            if (checkpoint.get("action_count") != self.cfg.action_count or checkpoint.get("phase") != "pretrain"
                    or checkpoint.get("dataset_fingerprint") != fingerprint):
                raise ValueError("Completed pretrained decision head has an incompatible dataset or action contract.")
            print("Loaded completed pretrained decision head.")
        else:
            checkpoint = train_pretrained_head(self.records, self.splits, self.teacher, self.predictor,
                                               self.cfg, self.device, self.out, self.sync)
            atomic_torch(path, checkpoint)
            self.push("Stage 3a complete: pretrained decision head")
        self.pretrained_checkpoint = checkpoint
        self.pretrained_head = self.load_head(checkpoint)
        print(f"Pretrained decision head: epoch {checkpoint['best_epoch']}, "
              f"development loss {checkpoint['dev_loss']:.4f}.")
        return self

    def train_decision_head(self):
        """Stage 3: the pretrained head (3a), then its fine-tuned successor (3b) and the inference bundle."""
        if not hasattr(self, "pretrained_checkpoint"):
            self.train_pretrained_head()
        section("STAGE 3b | Fine-tuning on the cheapest continuation of the pretrained head")
        path = self.out / "decision_head_completed.pt"
        fingerprint = finetune_fingerprint(self.dataset_fingerprint(), self.pretrained_checkpoint["weights"])
        if self.cfg.resume and path.exists():
            checkpoint = load_checkpoint(path)
            if (checkpoint.get("action_count") != self.cfg.action_count or checkpoint.get("phase") != "finetune"
                    or checkpoint.get("dataset_fingerprint") != fingerprint):
                raise ValueError("Completed decision head has an incompatible dataset or action contract.")
            print("Loaded completed fine-tuned decision head.")
        else:
            checkpoint = train_finetuned_head(self.records, self.splits, self.teacher, self.predictor,
                                              self.pretrained_checkpoint, self.cfg, self.device, self.out,
                                              self.sync)
            atomic_torch(path, checkpoint)
        self.head = self.load_head(checkpoint)
        self.heads = {"supervised_pretrained": self.pretrained_head, "supervised": self.head}
        self.decision_checkpoint = checkpoint
        if not all(torch.equal(value.cpu(), self.predictor.model.state_dict()[key].cpu())
                   for key, value in load_checkpoint(self.out / "snapshot_completed.pt")["weights"].items()):
            raise AssertionError("Frozen snapshot predictor changed during head training.")
        atomic_torch(self.out / "inference_bundle.pt", {
            "schema_version": 5, "implementation_revision": PIPELINE_REVISION,
            "config": asdict(self.cfg),
            "snapshot": load_checkpoint(self.out / "snapshot_completed.pt"),
            "decision_head": checkpoint, "pretrained_head": self.pretrained_checkpoint,
            "action_names": action_names(self.cfg),
            "cost_note": "Fixed-m cached bundles; no target-grading cost at deployment."})
        print(f"Fine-tuned decision head: epoch {checkpoint['best_epoch']}, "
              f"development loss {checkpoint['dev_loss']:.4f}.")
        self.push("Stage 3 complete: fine-tuned decision head and inference bundle")
        return self

    def decision_label_audit(self, role, position):
        """Every case of one saved Stage 3a question with its disposition and cheapest first steps, recomputed."""
        if not hasattr(self, "predictor"):
            self.train_snapshot()
        return decision_question_audit(self.records, self.splits, self.teacher, self.predictor,
                                       self.cfg, self.out, role, position)

    def rollout_audit(self, name, position):
        """Traced Stage 4 rollouts of one report question (`audit` or a benchmark), recomputed."""
        if not hasattr(self, "heads"):
            self.train_decision_head()
        return question_rollout_audit(self.records, self.splits[name][position], self.teacher,
                                      self.predictor, self.cfg, self.heads, self.out, name)

    def report(self):
        if not hasattr(self, "heads"):
            self.train_decision_head()
        results = {"evaluation_protocol": split_protocol(self.cfg, self.splits),
                   "decision_training": {
                       "supervised_pretrained": {"best_epoch": self.pretrained_checkpoint["best_epoch"],
                                                 "dev_loss": self.pretrained_checkpoint["dev_loss"],
                                                 "coverage": self.pretrained_checkpoint["coverage"]},
                       "supervised": {"best_epoch": self.decision_checkpoint["best_epoch"],
                                      "dev_loss": self.decision_checkpoint["dev_loss"],
                                      "coverage": self.decision_checkpoint["coverage"]},
                       "action_names": action_names(self.cfg)}}
        external = [Path(path).stem for path in self.cfg.test_files]
        names = ["audit"] + external
        rollout_path = self.out / ROLLOUT_FILE
        tmp = rollout_path.with_suffix(".jsonl.tmp")
        # Pooled audit questions are also benchmark questions: each question is rolled
        # out once, and only its summary inputs stay in RAM until the last report.
        evaluated, snapshots = {}, {}
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                for name in names:
                    ids = self.splits[name]
                    ranking = teacher_report(self.records, ids, self.cfg,
                                             self.teacher["scores"], self.heuristics,
                                             f"RANKING | {name} ({len(ids)} eligible questions)")
                    policy, questions = policy_report(self.records, ids, self.teacher, self.predictor,
                                                      self.cfg, self.heads, f"ACQUISITION | {name}",
                                                      evaluated, handle, snapshots)
                    results[name] = {"ranking": ranking, "adaptive": policy}
                    if questions:
                        results[name]["paired_top1"] = paired_top1(questions, self.cfg)
        except BaseException:
            tmp.unlink(missing_ok=True)  # Stage 4 restarts; partial outcomes are never reused.
            raise
        os.replace(tmp, rollout_path)
        # Superseded: earlier code wrote every traced row there, pooled-audit rows twice.
        (self.out / LEGACY_TRAJECTORY_FILE).unlink(missing_ok=True)
        available = [name for name in external if results[name]["adaptive"] is not None]
        if available:
            results["external_macro"] = {
                "included": available,
                "methods": {method: {
                    "top1_correct": float(np.mean([
                        results[name]["adaptive"]["policies"][method]["top1_correct"]
                        for name in available])),
                    "exact_max_recovery": float(np.mean([
                        results[name]["adaptive"]["policies"][method]["exact_max_recovery"]
                        for name in available
                        if results[name]["adaptive"]["policies"][method]["exact_max_recovery"] is not None]))
                    if any(results[name]["adaptive"]["policies"][method]["exact_max_recovery"] is not None
                           for name in available) else None,
                    "mean_cost": float(np.mean([
                        results[name]["adaptive"]["policies"][method]["mean_cost"]
                        for name in available]))}
                    for method in evaluation_methods(self.cfg)}}
        atomic_json(self.out / "results.json", results)
        print(f"Saved reports, rollout outcomes, checkpoints, and inference bundle to {self.out}")
        print(f"{ROLLOUT_FILE}: one line per question, zero-shot order, and method; "
              "WORK.rollout_audit(name, position) recomputes a question's full trajectories.")
        print("These are cached historical outcomes, not live API or Kaggle evidence.")
        self.results = results
        self.push("Stage 4 complete: reports and rollout outcomes")
        return results


def load_inference_bundle(path, device="cpu", variant="supervised"):
    """The config, frozen predictor, and one decision head of a bundle.

    `variant` picks "supervised" (the fine-tuned head, Stage 3b) or "supervised_pretrained"
    (the Stage 3a head).
    """
    bundle = load_checkpoint(path)
    if bundle.get("schema_version") != 5 or bundle.get("implementation_revision") != PIPELINE_REVISION:
        raise ValueError(f"Expected a version 5 inference bundle with revision {PIPELINE_REVISION} training contract; old bundles are incompatible.")
    heads = {"supervised": "decision_head", "supervised_pretrained": "pretrained_head"}
    if variant not in heads:
        raise ValueError(f"Unknown decision head variant {variant!r}; use one of {sorted(heads)}.")
    cfg = Config(**bundle["config"])
    cfg.validate()
    checkpoint = bundle[heads[variant]]
    if (bundle.get("action_names") != action_names(cfg) or
            checkpoint.get("action_count") != cfg.action_count or
            checkpoint.get("input_dim") != cfg.head_input_dim):
        raise ValueError("Inference bundle has an incompatible decision action/input contract.")
    predictor = FrozenPredictor(bundle["snapshot"], cfg, torch.device(device))
    head = SupervisedActionHead(cfg).to(device)
    head.load_state_dict(checkpoint["weights"])
    head.eval()
    return cfg, predictor, head


def run_pipeline(cfg=None):
    work = Workflow(cfg or Config())
    work.prepare().build_datasets().train_decision_head()
    work.report()
    return work


# %% CLI (the standalone notebook runs each stage in its own cell)
if __name__ == "__main__":
    run_pipeline()
