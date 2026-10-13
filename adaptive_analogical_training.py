"""Kaggle-compatible ranking -> snapshot -> supervised acquisition training.

The companion notebook embeds this module: uploading the notebook alone is enough.
No provider calls are made. Acquisition is simulated from recorded Layer-1 data.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from array import array
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

PIPELINE_REVISION = 6  # Stages 1-2 train on every presence state and zero-shot order.


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
    output_dir: str = "/kaggle/working/adaptive_analogical_shared_training_v6_run"
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
    patience: int = 15              # Checks without improvement (Stages 1-2) or epochs (Stage 3).
    auxiliary_weight: float = .25
    calibrate: bool = True          # Development-only temperature selection.
    recognition_threshold: float = .50
    later_rank_filter: bool = True
    continue_after_max: bool = True  # Complete the ranked pool after recognition.
    cost_unit: str = "total_calls"  # Includes evaluator graders; solver_calls remains optional.
    max_cost: Optional[float] = None # None = cost of the complete pool.
    decision_hidden: int = 64
    decision_epochs: int = 100
    decision_lr: float = 1e-4
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

    @property
    def n(self):
        return self.zero_shots + self.k

    @property
    def action_count(self):
        return self.k + 2

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
        assert self.decision_eval_batch_size >= 2
        assert self.dev_checks_per_epoch >= 1
        assert 0 <= self.teacher_complete_share < 1 and 0 <= self.snapshot_complete_share < 1
        assert self.print_every > 0
        assert self.hf_sync_minutes > 0
        if self.hf_repo is not None and not re.fullmatch(r"[\w.-]+/[\w.-]+", str(self.hf_repo)):
            raise ValueError("hf_repo must be a Hugging Face dataset repository id such as 'user/name'.")
        fixed_acquisition_sequences(self)


# Outside the run contract and the Stage 3 fingerprint: runtime controls, the Hub
# mirror, and the input log paths. A log is identified by its content, and its
# filename stem is the benchmark named in each record uid.
NON_CONTRACT_SETTINGS = ("resume", "output_dir", "device", "cpu_threads", "print_every",
                         "fixed_acquisition_orders", "continue_after_max", "decision_eval_batch_size",
                         "hf_repo", "hf_sync_minutes", "train_file", "test_files")


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


def compressed_checkpoint(path):
    with Path(path).open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def load_checkpoint(path):
    # Only load checkpoints created by this workflow, never untrusted .pt files.
    # Gzip covers schema-3 Stage 3 audit shards read once for migration.
    if compressed_checkpoint(path):
        with gzip.open(path, "rb") as handle:
            return torch.load(handle, map_location="cpu", weights_only=False)
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


def train_teacher(records, train_ids, val_ids, cfg, device, seed, name):
    """Train on every presence state and zero-shot order; select on complete-state AP."""
    seed_everything(seed)
    full = State((1 << cfg.n)-1, (1 << cfg.k)-1)
    # The ranking passed to later stages is read from complete states, so selection is too.
    vx = np.stack([observation(records[i], full, cfg) for i in val_ids])
    # The complete view is the one later stages read, so it gets its own share of training.
    views = SnapshotViews(records, train_ids, cfg, device, complete_share=cfg.teacher_complete_share)
    model = ResNet(cfg.input_dim, cfg.n, cfg).to(device)

    def evaluate(model):
        pred = predict_array(model, vx, device)
        score = float(np.mean([ap_score(ranked_indices(p, retrieval_order(cfg)), records[i].labels)
                               for p, i in zip(pred, val_ids)]))
        return score, f"development AP {score:.4f}"

    fit = train_on_views(model, views, cfg.teacher_epochs, seed, lambda logits, y: teacher_loss(logits, y, cfg),
                         evaluate, lambda new, old: new > old + 1e-8, cfg, name)
    print(f"{name:<24} | epochs {fit['epochs']:6.2f} | best {fit['best_epoch']:6.2f} | "
          f"development AP {fit['best']:.4f}")
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
    """Construct the fully measured version of any coherent structural state."""
    cells = sum(evaluators << (i * cfg.k) for i in range(cfg.n) if candidates & (1 << i))
    state = SnapshotState(candidates, retrieved, evaluators, retrieved,
                          evaluators, evaluators, cells, cells)
    validate_snapshot_state(state, cfg)
    return state


def validate_snapshot_state(state, cfg, allow_hidden_source=False):
    """Enforce causal availability, including source without evaluator."""
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
    if not allow_hidden_source and state.candidates >> cfg.zero_shots & ~state.evaluators:
        raise ValueError("one-shot candidate without retrieved source")
    if state.similarity_observed & ~state.retrieved:
        raise ValueError("similarity without retrieved source")
    if state.baseline_attempted & ~state.evaluators or state.baseline_observed & ~state.baseline_attempted:
        raise ValueError("invalid baseline masks")
    legal_cells = sum(state.evaluators << (i * cfg.k)
                      for i in range(cfg.n) if state.candidates & (1 << i))
    if state.ccs_attempted & ~legal_cells or state.ccs_observed & ~state.ccs_attempted:
        raise ValueError("invalid CCS masks")


def as_snapshot_state(state, cfg, allow_hidden_source=False):
    if isinstance(state, SnapshotState):
        return state
    if not isinstance(state, State):
        raise TypeError("Expected State or SnapshotState")
    active = state.evaluator_mask
    if allow_hidden_source:
        cells = sum(active << (i * cfg.k) for i in range(cfg.n)
                    if state.candidate_mask & (1 << i))
        result = SnapshotState(state.candidate_mask, active, active, active,
                               active, active, cells, cells)
        validate_snapshot_state(result, cfg, allow_hidden_source=True)
        return result
    return complete_snapshot_state(state.candidate_mask, active, active, cfg)


@lru_cache(maxsize=4)
def structural_catalog(zero_shots, k):
    """Every nonempty structure with an active evaluator for each one-shot source."""
    rows = []
    for zero_mask in range(1 << zero_shots):
        for code in range(3 ** k):
            candidates, retrieved, evaluators, rest = zero_mask, 0, 0, code
            for i in range(k):
                choice, rest = rest % 3, rest // 3
                if choice:
                    retrieved |= 1 << i
                    evaluators |= 1 << i
                if choice == 2:
                    candidates |= 1 << (zero_shots + i)
            if candidates:
                rows.append((candidates, retrieved, evaluators))
    return tuple(rows)


@lru_cache(maxsize=8)
def structural_states(zero_shots, k):
    """All valid presence states, including states without ZS1."""
    return tuple(State(candidates, evaluators)
                 for candidates, _, evaluators in structural_catalog(zero_shots, k))


@lru_cache(maxsize=8)
def structural_graph(zero_shots, k):
    """Budget-independent, monotone transitions shared by every question."""
    cfg = Config(zero_shots=zero_shots, k=k)
    states = structural_states(zero_shots, k)
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
    states, graph = structural_graph(zero_shots, k)
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


def raw_next_state(state, action, cfg):
    mask = present_mask(state, cfg)
    if action == 0:
        missing = np.flatnonzero(~bit_array(state.evaluator_mask, cfg.k))
        if not len(missing):
            return None
        return State(state.candidate_mask, state.evaluator_mask | (1 << int(missing[0])))
    if action == 1:
        missing = np.flatnonzero(~mask[:cfg.zero_shots])
        if not len(missing):
            return None
        return State(state.candidate_mask | (1 << int(missing[0])), state.evaluator_mask)
    source = action - 2
    if not 0 <= source < cfg.k or not state.evaluator_mask & (1 << source):
        return None
    slot = cfg.zero_shots + source
    if mask[slot]:
        return None
    return State(state.candidate_mask | (1 << slot), state.evaluator_mask)


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


def observation(record, state, cfg, allow_hidden_source=False):
    """Only observed entries are indexed; no labels/teacher scores enter x."""
    state = as_snapshot_state(state, cfg, allow_hidden_source)
    validate_snapshot_state(state, cfg, allow_hidden_source=allow_hidden_source)
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
def structural_observation_template(states, zero_shots, k, repeats, cost_unit,
                                    max_cost, allow_hidden_source):
    """Cache state-only bytes/masks; records supply only observed measurements.

    Build through the scalar contract once, retaining its validation, float32
    rounding and budget features. A default catalog needs about 1.8 MiB, rather
    than repeating this Python work for every question and historical order.
    """
    cfg = Config(zero_shots=zero_shots, k=k, repeats=repeats,
                 cost_unit=cost_unit, max_cost=max_cost)
    empty = Record("", "", "", [], [], np.zeros(k, np.float32),
                   np.zeros(k, np.float32), np.zeros((cfg.n, k), np.float32),
                   np.zeros(cfg.n, np.float32))
    template = np.stack([observation(empty, state, cfg, allow_hidden_source)
                         for state in states])
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


def observation_many(record, states, cfg, allow_hidden_source=False):
    """Batch complete structural states; augmented snapshots keep scalar logic."""
    states = tuple(states)
    if not states or not all(isinstance(state, State) for state in states):
        return np.stack([observation(record, state, cfg, allow_hidden_source)
                         for state in states])
    template, masks, offsets = structural_observation_template(
        states, cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit,
        cfg.max_cost, allow_hidden_source)
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


@lru_cache(maxsize=4)
def presence_states(zero_shots, k):
    """Every nonempty candidate subset with every evaluator subset.

    These are all states that hiding candidates or evaluators can produce,
    including one-shots whose evaluator is hidden: (2**8 - 1) * 2**5 = 8,160 by
    default, a superset of the 1,912 structural_states.
    """
    return tuple(State(candidates, evaluators)
                 for candidates in range(1, 1 << (zero_shots + k))
                 for evaluators in range(1 << k))


SNAPSHOT_EVAL_BATCH = 16384


class SnapshotViews:
    """Every (question, zero-shot order, presence state) view, assembled per batch.

    Nothing is materialized or written. A shared state table (about 7 MiB for the
    default 8,160 states) holds the state-only observation bytes and masks; each
    question adds its measurements, labels, and SAFE/MAX under each zero-shot order
    (under 1 KiB). Rows equal observation(..., allow_hidden_source=True) and
    snapshot_target of the permuted record bit for bit. Views are indexed
    question-major, then zero-shot order, then state.
    """
    def __init__(self, records, ids, cfg, device, teacher=None, states=None, complete_share=0.):
        self.cfg, self.device, self.ids = cfg, torch.device(device), list(ids)
        states = presence_states(cfg.zero_shots, cfg.k) if states is None else tuple(states)
        full = State((1 << cfg.n) - 1, (1 << cfg.k) - 1)
        template, masks, (sim, base, ccs) = structural_observation_template(
            states, cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost, True)
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

    `evaluate(model)` returns a comparable score and its description. Patience counts
    checks without improvement, so training can stop inside a long epoch.
    """
    if views.rows < 2:
        raise ValueError(f"{name}: too few training views.")
    # One fused update kernel on CUDA: small-model steps are otherwise launch-bound.
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                           fused=views.device.type == "cuda")
    best, best_text, weights, best_epoch, stale, history = None, "", None, 0., 0, []
    checks, position, started = cfg.dev_checks_per_epoch, 0., time.monotonic()
    complete = len(views.ids) * views.order_count * views.complete_copies
    print(f"{name}: {len(views.ids)} questions x {views.order_count} zero-shot orders x "
          f"{views.state_count} presence states = {views.rows:,} views; each complete view "
          f"{views.complete_copies:,}x, so {views.epoch_rows:,} rows per epoch, "
          f"{complete / views.epoch_rows:.1%} complete; {checks} development checks per epoch; "
          f"patience counts checks.", flush=True)
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
            history.append({"epoch": position, "train_loss": float(total) / seen, "score": score})
            epoch_total += total
            total.zero_()
            seen = 0
            if best is None or better(score, best):
                best, best_text, weights, best_epoch, stale = score, text, cpu_state(model), position, 0
            else:
                stale += 1
            if stale >= cfg.patience:
                break
        # Training loss averages this epoch's rows up to its last check.
        print(f"{name} | epoch {position:6.3f} | train loss {float(epoch_total) / max(1, done - seen):.4f} | "
              f"latest {text} | best {best_text} at epoch {best_epoch:.3f} | "
              f"checks without improvement {stale}/{cfg.patience} | "
              f"elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True)
        if stale >= cfg.patience:
            break
    return {"weights": weights, "best": best, "best_epoch": best_epoch,
            "epochs": position, "history": history}


def snapshot_loss(logits, targets, cfg):
    n = cfg.n
    mask = targets[:, 3*n + 2:]
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets[:, :3*n+2], reduction="none")
    losses = [((raw[:, b*n:(b+1)*n] * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
              for b in range(3)]
    return losses[0] + cfg.auxiliary_weight * (losses[1] + losses[2] + raw[:, 3*n:].mean(0).sum())


SNAPSHOT_SUM_FIELDS = ("views", "top1", "brier", "brier_cells",
                       "safe_predicted", "safe_true_positive", "safe_true",
                       "max_predicted", "max_true_positive", "max_true")


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
            sums.index_add_(0, q, torch.stack([c.double() for c in columns], 1))
    return sums.cpu().numpy()


def summarize_snapshot_sums(sums):
    total = np.asarray(sums, np.float64).reshape(-1, len(SNAPSHOT_SUM_FIELDS)).sum(0)
    value = dict(zip(SNAPSHOT_SUM_FIELDS, total))
    ratio = lambda a, b: float(a / b) if b else None
    return {"snapshots": int(value["views"]), "top1": ratio(value["top1"], value["views"]),
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
        metrics = summarize_snapshot_sums(snapshot_sums(model, dev, uncalibrated, cfg))
        return ((metrics["top1"], -metrics["brier"]),
                f"development Top-1 {metrics['top1']:.1%} | Brier {metrics['brier']:.4f}")

    fit = train_on_views(model, views, cfg.snapshot_epochs, cfg.seed + 201,
                         lambda logits, y: snapshot_loss(logits, y, cfg), evaluate,
                         lambda new, old: new > old, cfg, "Snapshot")
    model.load_state_dict(fit["weights"])
    temps = fit_temperatures(model, dev, cfg)
    print(f"Snapshot selected at epoch {fit['best_epoch']:.2f}; development Top-1 {fit['best'][0]:.1%}.")
    return {"weights": fit["weights"], "temperatures": temps, "history": fit["history"],
            "best_epoch": fit["best_epoch"], "input_dim": cfg.input_dim,
            "presence_states": views.state_count, "zero_shot_orders": views.order_count,
            "views_per_epoch": views.rows, "rows_per_epoch": views.epoch_rows,
            "complete_copies": views.complete_copies,
            "dev_summary": summarize_snapshot_sums(
                snapshot_sums(model, dev, temperature_vector(temps, cfg), cfg))}


class FrozenPredictor:
    def __init__(self, checkpoint, cfg, device):
        self.cfg, self.device = cfg, device
        self.model = ResNet(cfg.input_dim, 3*cfg.n+2, cfg).to(device)
        self.model.load_state_dict(checkpoint["weights"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.temperatures = temperature_vector(checkpoint["temperatures"], cfg)

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

    def predict_many(self, record, states, cfg_override=None, allow_hidden_source=False,
                     observations=None):
        """Batch frozen inference; only visible measurements enter the network."""
        cfg = cfg_override or self.cfg
        states = tuple(states)
        xs = observations if observations is not None else observation_many(
            record, states, cfg, allow_hidden_source)
        hs, ps = [], []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(xs), 1024):
                h = self.model.encode(torch.as_tensor(xs[start:start+1024], device=self.device))
                logits = self.model.output_layer(h).cpu().numpy()
                hs.append(h.cpu().numpy())
                ps.append(1 / (1 + np.exp(-np.clip(logits / self.temperatures, -40, 40))))
        hidden, probabilities = np.concatenate(hs), np.concatenate(ps)
        present = np.stack([present_mask(state, self.cfg) for state in states])
        candidate_probabilities = probabilities[:, :3*self.cfg.n].reshape(-1, 3, self.cfg.n)
        candidate_probabilities[~np.broadcast_to(present[:, None, :], candidate_probabilities.shape)] = 0
        return hidden, probabilities


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

def greedy_action(head, hidden, valid, device):
    if not valid.any():
        raise ValueError("No valid acquisition action.")
    with torch.no_grad():
        logits = head(torch.as_tensor(hidden[None], device=device))[0].cpu().numpy()
    return int(np.where(valid, logits, -np.inf).argmax())


def action_names(cfg):
    return (["add_next_evaluator", "add_zero_shot"] +
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

def states_from(start, cfg):
    seen, pending = {start}, [start]
    for state in pending:
        for action in np.flatnonzero(valid_actions(state, cfg)):
            nxt = advance(state, int(action), cfg)
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return pending

def decision_scenario(record, scores, safe, maximum, predictor, cfg, states,
                      allow_hidden_source=False, reference_order=None):
    """Cache frozen evidence and goals without optimizing a clairvoyant path."""
    # Reuse exactly the same observation bytes for inference and visible grouping.
    observations = observation_many(record, states, cfg, allow_hidden_source)
    if isinstance(predictor, FrozenPredictor):
        hidden, probabilities = predictor.predict_many(
            record, states, allow_hidden_source=allow_hidden_source, observations=observations)
    else:
        hidden, probabilities = predictor.predict_many(
            record, states, allow_hidden_source=allow_hidden_source)
    full_cfg = copy.deepcopy(cfg)
    full_cfg.max_cost = None
    _, full_probability = predictor.predict(record, State((1 << cfg.n)-1, (1 << cfg.k)-1), full_cfg)
    order = reference_order if reference_order is not None else ranked_indices(scores, retrieval_order(cfg))
    goal = decision_goals(order, safe, maximum, probabilities, full_probability, states, cfg)
    keys = [(state.candidate_mask, state.evaluator_mask, hashlib.sha256(x.tobytes()).hexdigest())
            for state, x in zip(states, observations)]
    return {"record": record, "states": states, "hidden": hidden, "goal": goal,
            "keys": keys, "index": {state: j for j, state in enumerate(states)}}


def decision_goals(order, safe, maximum, probabilities, full_probabilities, states, cfg):
    """Vectorized form of goal_condition, retaining its presence and tie rules."""
    masks = np.asarray([state.candidate_mask for state in states], dtype=np.int64)
    goal = np.empty((len(states), cfg.n), bool)
    n, threshold = cfg.n, cfg.recognition_threshold
    for rank, i in enumerate(order):
        met = (masks & (1 << i)) != 0
        if rank == 0 and maximum.any():
            met &= (probabilities[:, [3*n, 3*n+1, n+i, 2*n+i]] >= threshold).all(axis=1)
            present = np.asarray([present_mask(state, cfg) for state in states])
            selected = np.where(present, probabilities[:, :n], -np.inf).argmax(axis=1)
            met &= selected == i
        else:
            if cfg.later_rank_filter and rank < n - 1:
                met &= probabilities[:, i] > max(full_probabilities[j] for j in order[rank+1:])
            if rank > 0 and safe[i]:
                met &= probabilities[:, n+i] >= threshold
        goal[:, rank] = met
    return goal


def visible_state_key(record, state, cfg, allow_hidden_source=False):
    # Candidate IDs and hidden order are audit information, not model features.
    observed = observation(record, state, cfg, allow_hidden_source)
    return (state.candidate_mask, state.evaluator_mask, hashlib.sha256(observed.tobytes()).hexdigest())


# Case dispositions in precedence order; a plan stores each as an int8 code.
DECISION_STATUSES = ("OUT_OF_BUDGET", "COMPLETE", "EXHAUSTED", "BUDGET", "ACTION", "UNREACHABLE")


def plan_decision_rows(scenarios, cfg, transition_table=None):
    """Choose common actions until acquired evidence distinguishes the futures.

    Each visible group contains equiprobable historical orders with the same
    visible observation. Goals never split a group or enter the head's input.
    Search cost ends at goal attainment or action exhaustion. Each successor
    prioritizes its first unfinished rank, including an earlier lost goal.
    Values follow that recovery policy rather than an unrestricted fixed-goal
    shortest path.

    The plan is held in arrays rather than per-case dictionaries: for every
    case (scenario-major), a status code, goal rank, and training row; for
    every row, in planning order, its first case, tied optimal actions,
    concrete continuation, and per-action values. Solved continuations take
    24 bytes per case and target rank. Labels are compacted from these arrays;
    aggregate_decision_rows expands them into audit dictionaries.
    """
    groups = {}
    # Actions and costs depend on the structure/config, never on historical order.
    unique_states = {state for scenario in scenarios for state in scenario["states"]}
    _, _, cached_valid, cached_costs = decision_structure(
        cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    valid_by_state = {state: cached_valid[state] if state in cached_valid else valid_actions(state, cfg)
                      for state in unique_states}
    actions_by_state = {state: np.flatnonzero(valid) for state, valid in valid_by_state.items()}
    costs = {state: cached_costs[state] if state in cached_costs else state_cost(state, cfg)
             for state in unique_states}
    def fill_pool_action(state):
        """Prefer the next valid retrieval, otherwise the cheapest candidate."""
        return min(actions_by_state[state], key=lambda action: (
            0 if action == 0 else 1,
            costs[advance(state, int(action), cfg)] - costs[state], int(action)))
    prefixes = [np.logical_and.accumulate(scenario["goal"], axis=1) for scenario in scenarios]
    goal_ranks = [prefix.sum(axis=1) for prefix in prefixes]
    full = State((1 << cfg.n)-1, (1 << cfg.k)-1)
    pending, complete, exhausted, budget, action_status, unreachable = (
        -1, *(DECISION_STATUSES.index(name) for name in
              ("COMPLETE", "EXHAUSTED", "BUDGET", "ACTION", "UNREACHABLE")))
    starts, status = [0], []
    for sid, scenario in enumerate(scenarios):
        starts.append(starts[-1] + len(scenario["states"]))
        for j, state in enumerate(scenario["states"]):
            rank = int(goal_ranks[sid][j])
            status.append(complete if rank == cfg.n else
                          exhausted if state == full else
                          budget if not valid_by_state[state].any() else pending)
            groups.setdefault(scenario["keys"][j], []).append((sid, j, rank))

    # Compare the same feature pairs with the same NumPy tolerances, in bounded
    # batches, instead of invoking allclose thousands of times per question.
    comparisons = []
    for group in groups.values():
        sid, j, _ = group[0]
        hidden = scenarios[sid]["hidden"][j]
        comparisons.extend((hidden, scenarios[s]["hidden"][index])
                           for s, index, _ in group if (s, index) != (sid, j))
    for start in range(0, len(comparisons), 1024):
        batch = comparisons[start:start+1024]
        if not np.allclose(np.stack([pair[0] for pair in batch]),
                           np.stack([pair[1] for pair in batch]), atol=1e-6):
            raise ValueError("One visible state has inconsistent frozen features.")
    del comparisons

    transitions, shared_transitions = [], {}
    for scenario in scenarios:
        structure = tuple(scenario["states"])
        if structure not in shared_transitions:
            indexed = []
            for state in structure:
                possible = (transition_table[state] if transition_table is not None else
                            {int(action): advance(state, int(action), cfg)
                             for action in actions_by_state[state]})
                indexed.append({action: scenario["index"][nxt]
                                for action, nxt in possible.items()
                                if (costs[nxt] if nxt in costs else state_cost(nxt, cfg)) <= cfg.budget + 1e-6})
            shared_transitions[structure] = indexed
        transitions.append(shared_transitions[structure])

    n = cfg.n
    # (reach, cost, steps) for each case and target rank, filled as groups are solved.
    continuation = array("d", [math.nan]) * (starts[-1] * n * 3)
    case_row = [-1] * starts[-1]
    capacity = len(groups)
    row_case = np.empty(capacity, np.int64)
    row_chosen = np.empty(capacity, np.int64)
    row_support = np.zeros((capacity, cfg.action_count), bool)
    row_values = np.empty((capacity, cfg.action_count, 3), np.float64)
    row_key = []
    # The graph only adds items. Solve all visible-state groups backwards, so
    # every future decision is the same policy used to label that future row.
    ordered = sorted(groups.items(), key=lambda item: -(
        bin(item[0][0]).count("1") + bin(item[0][1]).count("1")))
    for key, group in ordered:
        sid, j, _ = group[0]
        state = scenarios[sid]["states"][j]
        active = [(s, index, rank) for s, index, rank in group
                  if status[starts[s] + index] == pending]
        if not active:
            for s, index, _ in group:
                achieved = int(goal_ranks[s][index])
                at = (starts[s] + index) * n * 3
                for target_rank in range(n):
                    continuation[at:at+3] = array("d", (float(achieved > target_rank), 0., 0.))
                    at += 3
            continue
        values = np.zeros((cfg.action_count, 3), np.float64)
        values[:, 1:] = np.inf
        for action in actions_by_state[state]:
            outcomes = []
            for s, index, rank in active:
                current = scenarios[s]
                successor = transitions[s][index][int(action)]
                nxt = current["states"][successor]
                delta = costs[nxt] - costs[state]
                if prefixes[s][successor, rank]:
                    outcome = (1., delta, 1.)
                else:
                    child = ((starts[s] + successor) * n + rank) * 3
                    if continuation[child] != continuation[child]:
                        raise AssertionError("A successor group was used before it was solved.")
                    outcome = (continuation[child], delta + continuation[child+1], 1. + continuation[child+2])
                outcomes.append(outcome)
            values[action] = outcomes[0] if len(outcomes) == 1 else np.mean(outcomes, axis=0)
        preferred = actions_by_state[state]
        for column, maximize in ((0, True), (1, False), (2, False)):
            scores = values[preferred, column]
            best = scores.max() if maximize else scores.min()
            preferred = preferred[(scores == best) | (np.abs(scores - best) <= 1e-9)]
        # Equal-cost recognition evidence takes precedence over generating a
        # candidate. Reachability and cost remain the primary objectives.
        if 0 in preferred:
            preferred = preferred[preferred == 0]
        chosen = int(preferred[0])  # Concrete tied continuation is deterministic.
        if values[chosen, 0] <= 0:
            # The constructed continuations have zero reach for the current
            # goal; another legal route may exist. Keep teaching an acquisition
            # so the head remains usable through the full pool.
            chosen = int(fill_pool_action(state))
            preferred = np.asarray([chosen])
        for s, index, _ in group:
            achieved = int(goal_ranks[s][index])
            at = (starts[s] + index) * n * 3
            if status[starts[s] + index] != pending:
                for target_rank in range(n):
                    continuation[at:at+3] = array("d", (float(achieved > target_rank), 0., 0.))
                    at += 3
                continue
            current = scenarios[s]
            successor = transitions[s][index][chosen]
            nxt = current["states"][successor]
            delta = costs[nxt] - costs[state]
            child = (starts[s] + successor) * n * 3
            for target_rank in range(n):
                if achieved > target_rank or prefixes[s][successor, target_rank]:
                    outcome = (1., 0., 0.) if achieved > target_rank else (1., delta, 1.)
                else:
                    if continuation[child] != continuation[child]:
                        raise AssertionError("A successor group was used before it was solved.")
                    outcome = (continuation[child], delta + continuation[child+1], 1. + continuation[child+2])
                continuation[at:at+3] = array("d", outcome)
                at += 3
                child += 3
        row = len(row_key)
        row_case[row], row_chosen[row], row_values[row] = starts[sid] + j, chosen, values
        row_support[row, preferred] = True
        row_key.append(key)
        reached = values[chosen, 0] > 0
        for s, index, _ in group:
            if status[starts[s] + index] == pending:
                status[starts[s] + index] = action_status if reached else unreachable
            case_row[starts[s] + index] = row
    if pending in status:
        raise AssertionError("A decision case was left without a disposition.")
    rows = len(row_key)
    return {"starts": np.asarray(starts, np.int64), "status": np.asarray(status, np.int8),
            "goal_rank": (np.concatenate(goal_ranks) if goal_ranks else np.zeros(0, np.int64)),
            "case_row": np.asarray(case_row, np.int64), "row_case": row_case[:rows],
            "row_key": row_key, "row_support": row_support[:rows],
            "row_chosen": row_chosen[:rows], "row_values": row_values[:rows],
            "valid_by_state": valid_by_state}


def aggregate_decision_rows(scenarios, cfg, transition_table=None):
    """Plan, then expand into one audit dictionary per row and per case."""
    plan = plan_decision_rows(scenarios, cfg, transition_table)
    starts, status, goal_rank, case_row = (plan[key] for key in ("starts", "status", "goal_rank", "case_row"))
    members = [[] for _ in range(len(plan["row_key"]))]
    for case in np.flatnonzero(case_row >= 0).tolist():
        members[case_row[case]].append(case)
    trained = (DECISION_STATUSES.index("ACTION"), DECISION_STATUSES.index("UNREACHABLE"))
    rows = []
    for row, (first, key, support, chosen, values) in enumerate(zip(
            plan["row_case"].tolist(), plan["row_key"], plan["row_support"],
            plan["row_chosen"].tolist(), plan["row_values"])):
        sid = int(np.searchsorted(starts, first, side="right")) - 1
        scenario, j = scenarios[sid], first - int(starts[sid])
        preferred = np.flatnonzero(support)
        target = np.zeros(cfg.action_count, np.float32)
        target[list(preferred)] = 1 / len(preferred)
        reach, cost, steps = values[chosen]
        ranks = Counter(int(goal_rank[case]) for case in members[row] if int(status[case]) in trained)
        rows.append({"uid": scenario["record"].uid, "state_key": key,
                     "h": scenario["hidden"][j], "target": target,
                     "valid": plan["valid_by_state"][scenario["states"][j]],
                     "goal_rank": next(iter(ranks)) if len(ranks) == 1 else -1,
                     "goal_rank_distribution": dict(ranks), "reach": float(reach),
                     "expected_cost": float(cost), "expected_steps": float(steps),
                     "action_reach": values[:, 0], "action_expected_cost": values[:, 1],
                     "action_expected_steps": values[:, 2],
                     "objective": "rank_goal" if reach > 0 else "fill_pool_unreachable_goal"})
    cases = []
    for sid, scenario in enumerate(scenarios):
        for j, state in enumerate(scenario["states"]):
            case = int(starts[sid]) + j
            rank, row = int(goal_rank[case]), int(case_row[case])
            cases.append({"scenario": sid, "candidate_mask": state.candidate_mask,
                          "evaluator_mask": state.evaluator_mask,
                          "status": DECISION_STATUSES[status[case]],
                          "evaluators": evaluator_names(state.evaluator_mask, cfg),
                          "goal_rank": None if rank == cfg.n else rank,
                          "training_row": None if row < 0 else row})
    return rows, cases


def decision_question_scenarios(record, scores, safe, maximum, predictor, cfg):
    """Every zero-shot order of one question over the in-budget states."""
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    scenarios = []
    for zero_order in permutations(range(cfg.zero_shots)):
        r, s, y_safe, y_max = permute_zero_shots(record, scores, safe, maximum, zero_order, cfg)
        scenarios.append(decision_scenario(
            r, s, y_safe, y_max, predictor, cfg, states,
            reference_order=permuted_reference_order(scores, zero_order, cfg)))
    return scenarios


def build_decision_rows(record, scores, safe, maximum, predictor, cfg):
    """Audit rows and every state/order case of one question, over-budget cases included."""
    all_states, transitions, _, costs = decision_structure(
        cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    scenarios = decision_question_scenarios(record, scores, safe, maximum, predictor, cfg)
    rows, cases = aggregate_decision_rows(scenarios, cfg, transitions)
    excluded = [state for state in all_states if costs[state] > cfg.budget + 1e-6]
    for sid in range(len(scenarios)):
        cases.extend({"scenario": sid, "candidate_mask": state.candidate_mask,
                      "evaluator_mask": state.evaluator_mask, "status": "OUT_OF_BUDGET",
                      "evaluators": evaluator_names(state.evaluator_mask, cfg),
                      "goal_rank": None, "training_row": None} for state in excluded)
    if len(cases) != len(all_states) * len(scenarios):
        raise AssertionError("Exhaustive decision coverage mismatch")
    return rows, cases


def build_decision_labels(record, scores, safe, maximum, predictor, cfg):
    """One question's stored labels and coverage, planned without audit dictionaries.

    The same plan as build_decision_rows; only its arrays are kept, so a
    default-geometry question peaks near 20 MiB instead of 60 MiB.
    """
    all_states, transitions, _, _ = decision_structure(
        cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    # The scenarios (frozen features and keys) are released once the plan returns.
    plan = plan_decision_rows(decision_question_scenarios(record, scores, safe, maximum, predictor, cfg),
                              cfg, transitions)
    coverage = plan_coverage(plan, cfg, (len(all_states) - len(states)) * math.factorial(cfg.zero_shots))
    return {"coverage": coverage, **compact_label_arrays(record, plan["row_case"], plan["row_support"],
                                                         plan["row_key"], cfg)}

class SupervisedActionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.decision_hidden), nn.ReLU(),
                                 nn.Linear(cfg.decision_hidden, cfg.action_count))

    def forward(self, hidden):
        return self.net(hidden)

def predictor_fingerprint(predictor):
    digest = hashlib.sha256()
    for name, tensor in sorted(predictor.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(np.asarray(predictor.temperatures).tobytes())
    return digest.hexdigest()


def decision_dataset_fingerprint(records, splits, teacher, predictor, cfg, legacy_log_paths=None):
    """Stage 3 identity: config, records, roles, teacher labels, and frozen predictor.

    Schema 4 leaves out the input log paths, so logs downloaded to a new directory
    keep their labels; the records' content and uids already identify them. Earlier
    code hashed the configured paths too (schema 3): `legacy_log_paths`, the train
    file followed by the test files, reproduces that fingerprint for upgrades.
    """
    config = {key: value for key, value in asdict(cfg).items() if key not in NON_CONTRACT_SETTINGS}
    schema = 4
    if legacy_log_paths is not None:
        schema = 3
        config.update(train_file=legacy_log_paths[0], test_files=list(legacy_log_paths[1:]))
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema": schema, "revision": PIPELINE_REVISION, "config": config,
                              "predictor": predictor_fingerprint(predictor),
                              "data_sha256": data_digest(records),
                              "roles": {role: [records[i].uid for i in splits[role]]
                                        for role in ("policy", "dev")}}, sort_keys=True).encode())
    for role in ("policy", "dev"):
        for idx in splits[role]:
            for key in ("scores", "safe", "maximum"):
                digest.update(np.asarray(teacher[key][idx], dtype=np.float32).tobytes())
    return digest.hexdigest()


# Schema 5 stores one action byte per row plus one first-case bitmap per question;
# schema 4 (an int32 source per row) converts in place without relabeling.
DECISION_LABEL_SCHEMA = 5


def question_paths(records, ids, directory):
    return [directory / f"{position:06d}_{hashlib.sha256(records[idx].uid.encode()).hexdigest()[:12]}.pt"
            for position, idx in enumerate(ids)]


def decision_shard_paths(records, splits, out):
    """Schema-3 audit shards from earlier revision-5 runs, read once to migrate."""
    return {role: question_paths(records, splits[role], Path(out) / "decision_shards" / role)
            for role in ("policy", "dev")}


def decision_role_path(out, role):
    return Path(out) / "decision_labels" / f"{role}.pt"


def reclaim_derived_stage3_files(out):
    """Delete derived caches and incomplete writes; completed work is never touched.

    Earlier revision-5 runs kept a `.training.pt` companion beside each
    schema-3 audit shard (derived from the shard, up to 4 MiB each), and an
    interrupted atomic write leaves a `.tmp` file. Neither holds completed
    work. Removing them before a resumed run writes anything lets a run that
    filled its disk continue without manual cleanup.
    """
    out, reclaimed = Path(out), 0
    for pattern in ("*.tmp", "decision_labels/*.tmp", "decision_labels/*/*.tmp",
                    "decision_shards/*/*.tmp", "decision_shards/*/*.training.pt"):
        for path in out.glob(pattern):
            if path.is_file():
                reclaimed += path.stat().st_size
                path.unlink()
    return reclaimed


def legacy_log_paths(cfg, out):
    """Log paths a schema-3 fingerprint may have hashed, train file first.

    As configured now, resolved, as contract.json records them (resolved, by the
    session that created the run), and as the inference bundle's config names
    them (the configured strings when Stage 3 last completed).
    """
    paths = [cfg.train_file] + list(cfg.test_files)
    candidates = [paths, [str(Path(path).resolve()) for path in paths]]
    contract = Path(out) / "contract.json"
    if contract.exists():
        candidates.append([source.get("path") for source in
                           json.loads(contract.read_text(encoding="utf-8")).get("sources", [])])
    bundle = Path(out) / "inference_bundle.pt"
    if bundle.exists():
        config = load_checkpoint(bundle).get("config", {})
        candidates.append([config.get("train_file")] + list(config.get("test_files") or []))
    unique = []
    for candidate in candidates:
        if (len(candidate) == len(paths) and all(isinstance(path, str) for path in candidate)
                and candidate not in unique):
            unique.append(candidate)
    return unique


def rewrite_checkpoint(path, value):
    """Atomic in-place rewrite that keeps a schema-3 shard's gzip framing."""
    path = Path(path)
    if not compressed_checkpoint(path):
        atomic_torch(path, value)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=1, mtime=0) as zipped:
                torch.save(value, zipped)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def upgrade_decision_fingerprints(records, splits, teacher, predictor, cfg, out):
    """Rebind Stage 3 files from the path-bound schema-3 fingerprint; returns the current one.

    Earlier code hashed the input log paths into the fingerprint, so the same logs
    at a new path rejected completed labels, progress, and the head. Files whose
    fingerprint is that schema-3 value for a known path set (`legacy_log_paths`)
    get the current fingerprint in place, without relabeling or retraining. The
    manifest is rewritten last, so an interrupted upgrade repeats; once it is done,
    only the manifest is read.
    """
    out = Path(out)
    fingerprint = decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    manifest_path = out / "decision_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest is None or manifest.get("fingerprint") == fingerprint:
        return fingerprint
    legacy = {decision_dataset_fingerprint(records, splits, teacher, predictor, cfg, paths)
              for paths in legacy_log_paths(cfg, out)}
    if manifest.get("fingerprint") not in legacy:
        # Another config, split, teacher, or predictor: the callers reject these files.
        print("NOTE: Stage 3 files match neither this run nor its path-bound fingerprint at the "
              "configured, resolved, contract.json, or inference-bundle log paths. If only the log "
              "paths changed, place the logs at the paths that built them for one session; the "
              "files are then upgraded in place.", flush=True)
        return fingerprint
    reclaim_derived_stage3_files(out)  # Free space, and drop shard companions, before rewriting.
    paths = [out / "decision_head_completed.pt", out / "decision_head_progress.pt",
             *sorted(out.glob("decision_labels/*.pt")), *sorted(out.glob("decision_labels/*/*.pt")),
             *sorted(out.glob("decision_shards/*/*.pt"))]
    upgraded = 0
    for path in paths:
        if not path.is_file():
            continue
        value = load_checkpoint(path)
        key = "dataset_fingerprint" if path.name == "decision_head_completed.pt" else "fingerprint"
        if isinstance(value, dict) and value.get(key) in legacy:
            value[key] = fingerprint
            rewrite_checkpoint(path, value)
            upgraded += 1
        del value
    atomic_json(manifest_path, dict(manifest, fingerprint=fingerprint))
    print(f"Stage 3 fingerprint: the manifest and {upgraded} label, progress, or head files hashed "
          "the input log paths; rebound them to the logs' content without relabeling.", flush=True)
    return fingerprint


def label_coverage(rows, cases, cfg):
    """One question's state/order dispositions; the rows and cases are not stored."""
    statuses = Counter(case["status"] for case in cases)
    structures = structural_states(cfg.zero_shots, cfg.k)
    expected = {(sid, state.candidate_mask, state.evaluator_mask)
                for sid in range(math.factorial(cfg.zero_shots)) for state in structures}
    actual = {(case["scenario"], case["candidate_mask"], case["evaluator_mask"])
              for case in cases}
    if len(actual) != len(cases) or actual != expected:
        raise ValueError("Decision labels have incomplete state/order coverage.")
    if set(statuses) - set(DECISION_STATUSES):
        raise ValueError("Decision labels contain an unknown status.")
    _, _, cached_valid, _ = decision_structure(
        cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    trained_statuses = Counter()
    for case in cases:
        index = case["training_row"]
        state = State(case["candidate_mask"], case["evaluator_mask"])
        valid = cached_valid[state] if state in cached_valid else valid_actions(state, cfg)
        if bool(valid.any()) != (index is not None):
            raise ValueError("Decision label action mapping is incomplete.")
        if index is not None and not 0 <= index < len(rows):
            raise ValueError("Decision labels have an invalid training-row reference.")
        if index is not None:
            trained_statuses[case["status"]] += 1
    for row in rows:
        if (len(row["target"]) != cfg.action_count or
                len(row["valid"]) != cfg.action_count or
                not np.isclose(row["target"].sum(), 1) or
                np.any(row["target"][~row["valid"]])):
            raise ValueError("Decision labels have an invalid masked action target.")
    return {"questions": 1, "structural_states": len(structures),
            "state_order_cases": len(cases), "action_rows": len(rows),
            "status_counts": dict(statuses), "trained_status_counts": dict(trained_statuses),
            "actionable_by_target_rank": dict(Counter(str(case["goal_rank"] + 1)
                for case in cases if case["status"] == "ACTION"))}


def plan_coverage(plan, cfg, out_of_budget):
    """label_coverage of a question's plan, without expanding its cases.

    The plan covers every in-budget state under every zero-shot order; the
    over-budget cases, which never have a row, are only counted.
    """
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    structures = structural_states(cfg.zero_shots, cfg.k)
    orders = math.factorial(cfg.zero_shots)
    status, case_row = plan["status"], plan["case_row"]
    if (len(status) != orders * len(states) or
            out_of_budget != orders * (len(structures) - len(states)) or
            not np.array_equal(plan["starts"], np.arange(orders + 1) * len(states))):
        raise ValueError("Decision labels have incomplete state/order coverage.")
    valid = decision_valid_table(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    if not np.array_equal(case_row >= 0, np.tile(valid.any(1), orders)):
        raise ValueError("Decision label action mapping is incomplete.")
    if case_row.max(initial=-1) >= len(plan["row_case"]):
        raise ValueError("Decision labels have an invalid training-row reference.")
    names = [DECISION_STATUSES[code] for code in status.tolist()]
    statuses = Counter(names)
    if out_of_budget:
        statuses["OUT_OF_BUDGET"] += out_of_budget
    return {"questions": 1, "structural_states": len(structures),
            "state_order_cases": len(status) + out_of_budget, "action_rows": len(plan["row_case"]),
            "status_counts": dict(statuses),
            "trained_status_counts": dict(Counter(name for name, row in zip(names, case_row.tolist())
                                                  if row >= 0)),
            "actionable_by_target_rank": dict(Counter(str(rank + 1) for name, rank in
                                                      zip(names, plan["goal_rank"].tolist())
                                                      if name == "ACTION"))}


def combine_coverage(parts):
    statuses, trained_statuses, ranks = Counter(), Counter(), Counter()
    for part in parts:
        statuses.update(part["status_counts"])
        trained_statuses.update(part["trained_status_counts"])
        ranks.update(part["actionable_by_target_rank"])
    return {"questions": len(parts),
            "structural_states_per_question": parts[0]["structural_states"] if parts else 0,
            "state_order_cases": sum(p["state_order_cases"] for p in parts),
            "action_rows": sum(p["action_rows"] for p in parts),
            "status_counts": dict(statuses), "trained_status_counts": dict(trained_statuses),
            "actionable_by_target_rank": dict(ranks)}


def pack_decision_rows(rows, cfg, keys=("h", "target", "valid")):
    """Stack the head's row arrays from full audit rows."""
    shapes = {"h": (cfg.hidden_dim, torch.float32), "target": (cfg.action_count, torch.float32),
              "valid": (cfg.action_count, torch.bool)}
    return {key: torch.from_numpy(np.stack([row[key] for row in rows])) if rows else
            torch.empty((0, shapes[key][0]), dtype=shapes[key][1])
            for key in keys}


@lru_cache(maxsize=8)
def decision_valid_table(zero_shots, k, repeats, cost_unit, max_cost):
    """Valid-action masks for the in-budget states, in label-source order."""
    states, _ = budget_state_index(zero_shots, k, repeats, cost_unit, max_cost)
    _, _, valid, _ = decision_structure(zero_shots, k, repeats, cost_unit, max_cost)
    table = np.stack([valid[state] for state in states])
    table.setflags(write=False)
    return table


def decision_valid(source, cfg):
    table = decision_valid_table(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    return table[np.asarray(source, dtype=np.int64) % len(table)]


def decision_targets(preferred, cfg):
    """Each row's soft target: an even split over its tied optimal actions."""
    support = np.unpackbits(np.asarray(preferred, np.uint8), axis=1,
                            count=cfg.action_count).astype(bool)
    # Same float64 -> float32 rounding as the planner's 1 / len(preferred).
    return np.where(support, 1 / np.maximum(support.sum(1, keepdims=True), 1), 0).astype(np.float32)


def decision_row_observations(record, source, cfg):
    """Rebuild each row's label-time observation from its scenario * states + state source."""
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    template, masks, offsets = structural_observation_template(
        states, cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost, False)
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


@lru_cache(maxsize=8)
def decision_state_sizes(zero_shots, k, repeats, cost_unit, max_cost):
    """Acquired candidates plus evaluators of each in-budget state: the planning-order key."""
    states, _ = budget_state_index(zero_shots, k, repeats, cost_unit, max_cost)
    sizes = np.asarray([bin(state.candidate_mask).count("1") + bin(state.evaluator_mask).count("1")
                        for state in states], np.int64)
    sizes.setflags(write=False)
    return sizes


def decision_case_count(cfg):
    """In-budget state/order cases of one question: zero-shot orders times in-budget states."""
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    return math.factorial(cfg.zero_shots) * len(states)


def decision_first_cases(source, cfg):
    """Bitmap over a question's in-budget cases marking each row's first visible case."""
    source = np.asarray(source, np.int64)
    cases = decision_case_count(cfg)
    if source.ndim != 1 or len(np.unique(source)) != len(source) or (
            len(source) and (source.min() < 0 or source.max() >= cases)):
        raise ValueError("Decision rows need distinct in-budget first cases.")
    bits = np.zeros(cases, bool)
    bits[source] = True
    return np.packbits(bits)


def decision_row_sources(first_cases, cfg):
    """Each row's first case (zero-shot scenario * in-budget states + state), in row order.

    The planner solves visible groups from the largest structures down, in
    order of first appearance among equal sizes, so the bitmap of first cases
    fixes every row's source and position; no per-row index is stored.
    """
    first_cases = np.asarray(first_cases, np.uint8)
    cases = decision_case_count(cfg)
    if first_cases.shape != ((cases + 7) // 8,) or np.unpackbits(first_cases)[cases:].any():
        raise ValueError("Decision first-case bitmap has the wrong size or padding.")
    source = np.flatnonzero(np.unpackbits(first_cases, count=cases))
    sizes = decision_state_sizes(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    return source[np.argsort(-sizes[source % len(sizes)], kind="stable")]


def compact_label_arrays(record, source, support, keys, cfg):
    """Keep only what the head trains on: about one byte per row.

    A row's first case (zero-shot scenario and in-budget state) fixes its
    observation, frozen features, and valid mask; a bitmask of its tied optimal
    actions fixes its soft target. Rows keep planning order, which a bitmap of
    first cases reproduces, so a question stores one byte per row (up to eight
    actions) plus one bit per in-budget case. Planner values and case
    dispositions are deterministic audit data, recomputed by
    decision_question_audit.
    """
    states, _ = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    source, support = np.asarray(source, np.int64), np.asarray(support, bool)
    first_cases = decision_first_cases(source, cfg)
    if not np.array_equal(decision_row_sources(first_cases, cfg), source):
        raise ValueError("Decision rows are not in planning order; their sources cannot be rebuilt.")
    if (support.shape != (len(source), cfg.action_count) or not support.any(1).all() or
            (support & ~decision_valid(source, cfg)).any()):
        raise ValueError("Decision targets must use valid actions.")
    observed = decision_row_observations(record, source, cfg)
    for key, x, position in zip(keys, observed, source % len(states)):
        state = states[position]
        if tuple(key) != (state.candidate_mask, state.evaluator_mask, hashlib.sha256(x.tobytes()).hexdigest()):
            raise ValueError("Decision rows do not reproduce their visible observations.")
    return {"preferred": torch.from_numpy(np.packbits(support, axis=1)),
            "first_cases": torch.from_numpy(first_cases),
            "observation_sha256": hashlib.sha256(observed.tobytes()).hexdigest()}


def compact_decision_labels(record, rows, cases, cfg):
    """Stored labels from audit rows and cases, such as a migrated schema-3 shard."""
    states, index = budget_state_index(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    source = np.full(len(rows), -1, np.int64)
    for case in cases:
        row = case["training_row"]
        if row is not None and source[row] < 0:
            position = index.get(State(case["candidate_mask"], case["evaluator_mask"]), -1)
            if position < 0:
                source[row] = -2  # First visible case is over budget; rejected below.
            else:
                source[row] = case["scenario"] * len(states) + position
    if (source < 0).any():
        raise ValueError("Decision labels have a training row without an in-budget visible case.")
    target = (np.stack([row["target"] for row in rows]) if rows else
              np.zeros((0, cfg.action_count), np.float32))
    valid = (np.stack([row["valid"] for row in rows]) if rows else
             np.zeros((0, cfg.action_count), bool))
    if target.dtype != np.float32 or not np.array_equal(
            decision_targets(np.packbits(target > 0, axis=1), cfg), target):
        raise ValueError("Decision targets must split evenly over tied optimal actions.")
    if not np.array_equal(decision_valid(source, cfg), valid):
        raise ValueError("Decision valid masks do not match their in-budget states.")
    return compact_label_arrays(record, source, target > 0, [row["state_key"] for row in rows], cfg)


def schema4_first_cases(source, cfg, where):
    """Schema 4 stored an int32 source per row; schema 5 keeps the equivalent bitmap."""
    source = np.asarray(source, np.int64)
    try:
        first_cases = decision_first_cases(source, cfg)
    except ValueError:
        first_cases = None
    if first_cases is None or not np.array_equal(decision_row_sources(first_cases, cfg), source):
        raise ValueError(f"Incompatible decision labels: {where} rows are not in planning order.")
    return torch.from_numpy(first_cases)


def upgrade_schema4_labels(labels, cfg, where):
    """One question's schema-4 labels in schema 5, without relabeling."""
    upgraded = {key: value for key, value in labels.items() if key != "source"}
    upgraded.update(schema_version=DECISION_LABEL_SCHEMA,
                    first_cases=schema4_first_cases(labels["source"].numpy(), cfg, where))
    return upgraded


def upgrade_schema4_store(store, cfg, where):
    """A merged schema-4 role file in schema 5, without relabeling."""
    offsets, source = store["offsets"].tolist(), store["source"].numpy()
    first_cases = [schema4_first_cases(source[start:stop], cfg, f"{where} question {position}")
                   for position, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:]))]
    upgraded = {key: value for key, value in store.items() if key != "source"}
    upgraded.update(schema_version=DECISION_LABEL_SCHEMA,
                    first_cases=(torch.stack(first_cases) if first_cases else
                                 torch.empty((0, (decision_case_count(cfg) + 7) // 8), dtype=torch.uint8)))
    return upgraded


def check_saved_labels(record, labels, cfg):
    """Saved labels must rebuild their label-time observations under the current code.

    Returns each row's source, rebuilt from the first-case bitmap.
    """
    preferred = labels["preferred"].numpy()
    try:
        source = decision_row_sources(labels["first_cases"].numpy(), cfg)
    except ValueError as exc:
        raise ValueError(f"Saved Stage 3 labels for {record.uid} have inconsistent shapes.") from exc
    if (preferred.shape != (len(source), (cfg.action_count + 7) // 8) or
            len(source) != labels["coverage"]["action_rows"]):
        raise ValueError(f"Saved Stage 3 labels for {record.uid} have inconsistent shapes.")
    support = np.unpackbits(preferred, axis=1, count=cfg.action_count).astype(bool)
    if not support.any(1).all() or (support & ~decision_valid(source, cfg)).any():
        raise ValueError(f"Saved Stage 3 labels for {record.uid} target invalid actions.")
    observed = decision_row_observations(record, source, cfg)
    if hashlib.sha256(observed.tobytes()).hexdigest() != labels["observation_sha256"]:
        raise ValueError(f"Saved Stage 3 labels for {record.uid} no longer reproduce their "
                         "label-time observations; observation code or data changed.")
    return source


def decision_training_rows(record, source, preferred, predictor, cfg):
    """One question's head inputs: decoded targets plus re-encoded frozen features."""
    source = np.asarray(source)
    return {"h": predictor.encode(decision_row_observations(record, source, cfg)),
            "target": torch.from_numpy(decision_targets(preferred, cfg)),
            "valid": torch.from_numpy(decision_valid(source, cfg))}


def load_label_file(path):
    """Compact label files hold only tensors, strings, and counts."""
    return torch.load(path, map_location="cpu", weights_only=True)


def decision_role_labels(role, records, ids, teacher, predictor, cfg, out, fingerprint, sync=None):
    """Build, migrate, or reuse one role's labels; returns them in RAM.

    Each question completes as one small atomic file, and the finished role is
    merged into a single file. Schema-3 audit shards and schema-4 labels are
    converted without relabeling; shards are deleted once converted, so a full
    disk regains space.
    """
    out = Path(out)
    uids = [records[idx].uid for idx in ids]
    role_path = decision_role_path(out, role)
    paths = question_paths(records, ids, out / "decision_labels" / role)
    legacy_paths = question_paths(records, ids, out / "decision_shards" / role)
    expected_cases = len(structural_states(cfg.zero_shots, cfg.k)) * math.factorial(cfg.zero_shots)
    store = load_label_file(role_path) if role_path.exists() else None
    upgraded = False
    if store is not None:
        offsets = store["offsets"].tolist() if isinstance(store.get("offsets"), torch.Tensor) else []
        if (store.get("schema_version") not in (4, DECISION_LABEL_SCHEMA) or store.get("fingerprint") != fingerprint
                or store.get("role") != role or store.get("uids") != uids
                or len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(store["preferred"])):
            raise ValueError(f"Incompatible decision labels: {role_path}")
        if store["schema_version"] == 4:
            store, upgraded = upgrade_schema4_store(store, cfg, role_path), True
            print(f"Stage 3 {role} labels: converting the schema-4 role file to schema "
                  f"{DECISION_LABEL_SCHEMA} without relabeling.", flush=True)
    preferred, first_cases, digests, parts = [], [], [], []
    started = last_print = time.monotonic()
    built = reused = migrated = 0
    print(f"Stage 3 {role} labels: building/validating {len(ids)} questions.", flush=True)
    for position, (idx, path, legacy) in enumerate(zip(ids, paths, legacy_paths), 1):
        record = records[idx]
        if store is not None:
            start, stop = offsets[position-1:position+1]
            labels = {"preferred": store["preferred"][start:stop],
                      "first_cases": store["first_cases"][position-1],
                      "observation_sha256": store["observation_sha256"][position-1],
                      "coverage": store["coverage"][position-1]}
        elif path.exists():
            labels = load_label_file(path)
            if (labels.get("schema_version") not in (4, DECISION_LABEL_SCHEMA) or
                    labels.get("fingerprint") != fingerprint or labels.get("uid") != record.uid):
                raise ValueError(f"Incompatible decision labels: {path}")
        else:
            labels = None
        if labels is not None and labels.get("schema_version") == 4:
            # Rewritten in place, so a later session and the mirror hold schema 5.
            labels = upgrade_schema4_labels(labels, cfg, path)
            check_saved_labels(record, labels, cfg)
            atomic_torch(path, labels)
            migrated += 1
        elif labels is not None:
            check_saved_labels(record, labels, cfg)
            reused += 1
        else:
            if legacy.exists():
                shard = load_checkpoint(legacy)
                if (shard.get("schema_version") != 3 or shard.get("fingerprint") != fingerprint
                        or shard.get("uid") != record.uid or shard.get("expected_cases") != expected_cases):
                    raise ValueError(f"Incompatible decision shard: {legacy}")
                rows, cases = shard["rows"], shard["cases"]
                del shard
                labels = {"coverage": label_coverage(rows, cases, cfg),
                          **compact_decision_labels(record, rows, cases, cfg)}
                del rows, cases
                migrated += 1
            else:
                labels = build_decision_labels(record, teacher["scores"][idx], teacher["safe"][idx],
                                               teacher["maximum"][idx], predictor, cfg)
                built += 1
            labels = {"schema_version": DECISION_LABEL_SCHEMA, "fingerprint": fingerprint,
                      "uid": record.uid, **labels}
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch(path, labels)
        # The compact file is saved, so the large audit shard is no longer needed.
        legacy.unlink(missing_ok=True)
        preferred.append(labels["preferred"])
        first_cases.append(labels["first_cases"])
        digests.append(labels["observation_sha256"])
        parts.append(labels["coverage"])
        last_print = decision_progress(f"Stage 3 {role} labels", position, len(ids), started, last_print, cfg,
                                       f" | built {built}, reused {reused}" +
                                       (f", migrated {migrated}" if migrated else ""))
        if sync is not None:
            sync.push(f"Stage 3: {role} labels {position}/{len(ids)} questions")
    if store is None:
        store = {"schema_version": DECISION_LABEL_SCHEMA, "fingerprint": fingerprint, "role": role,
                 "uids": uids,
                 "offsets": torch.from_numpy(np.cumsum([0] + [len(rows) for rows in preferred], dtype=np.int64)),
                 "preferred": (torch.cat(preferred) if preferred else
                               torch.empty((0, (cfg.action_count + 7) // 8), dtype=torch.uint8)),
                 "first_cases": (torch.stack(first_cases) if first_cases else
                                 torch.empty((0, (decision_case_count(cfg) + 7) // 8), dtype=torch.uint8)),
                 "observation_sha256": digests, "coverage": parts}
        atomic_torch(role_path, store)
    elif upgraded:
        atomic_torch(role_path, store)
    del preferred, first_cases
    # The merged role file is complete; remove per-question files and leftovers.
    for path in paths + legacy_paths:
        path.unlink(missing_ok=True)
    for directory in (out / "decision_labels" / role, out / "decision_shards" / role, out / "decision_shards"):
        try:
            directory.rmdir()
        except OSError:
            pass  # Missing, or holds files this workflow does not own.
    if sync is not None:
        sync.push(f"Stage 3: {role} labels complete", force=True)
    return store


def decision_question_audit(records, splits, teacher, predictor, cfg, out, role, position):
    """Rebuild one saved question's full rows and cases, checked against its labels.

    Stage 3 stores only head inputs. Goal ranks, per-action reach/cost/steps, and
    case dispositions are deterministic, so they are recomputed here on demand
    (use the labeling device; another device may round probabilities differently).
    """
    fingerprint = upgrade_decision_fingerprints(records, splits, teacher, predictor, cfg, out)
    path = decision_role_path(out, role)
    store = load_label_file(path)
    if (store.get("schema_version") not in (4, DECISION_LABEL_SCHEMA) or store.get("fingerprint") != fingerprint
            or store.get("role") != role):
        raise ValueError(f"Incompatible decision labels: {path}")
    if store["schema_version"] == 4:
        # A head completed under schema 4 never rewrote its labels; read them as schema 5.
        store = upgrade_schema4_store(store, cfg, path)
    idx = splits[role][position]
    rows, cases = build_decision_rows(records[idx], teacher["scores"][idx], teacher["safe"][idx],
                                      teacher["maximum"][idx], predictor, cfg)
    compact = compact_decision_labels(records[idx], rows, cases, cfg)
    start, stop = store["offsets"][position:position+2].tolist()
    if not (torch.equal(compact["preferred"], store["preferred"][start:stop]) and
            torch.equal(compact["first_cases"], store["first_cases"][position]) and
            compact["observation_sha256"] == store["observation_sha256"][position]):
        raise ValueError("Rebuilt Stage 3 labels differ from the saved labels.")
    return {"uid": records[idx].uid, "rows": rows, "cases": cases,
            "coverage": label_coverage(rows, cases, cfg)}


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


def train_decision_head(records, splits, teacher, predictor, cfg, device, out=None, sync=None):
    if splits["policy"] != splits["supervised"]:
        raise ValueError("Stage 3 policy questions must match the Stage 1/2 supervised questions.")
    seed_everything(cfg.seed + 300)
    if out is None:
        raise ValueError("Stage 3 requires an output directory for its label files.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roles = ("policy", "dev")
    fingerprint = upgrade_decision_fingerprints(records, splits, teacher, predictor, cfg, out)
    frozen_before = predictor_fingerprint(predictor)
    manifest_path = out / "decision_dataset_manifest.json"
    manifest = {"schema_version": DECISION_LABEL_SCHEMA, "implementation_revision": PIPELINE_REVISION,
                "fingerprint": fingerprint,
                "roles": {role: decision_role_path(out, role).relative_to(out).as_posix() for role in roles}}
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous is not None:
        # Schema-3 and schema-4 runs with the same label contract are converted, not relabeled.
        if previous != manifest and not (
                previous.get("schema_version") in (3, 4) and
                previous.get("implementation_revision") == PIPELINE_REVISION and
                previous.get("fingerprint") == fingerprint):
            raise ValueError("Decision dataset manifest has an incompatible teacher, predictor, split, or action contract.")
    # Free space before any write: interrupted writes, and schema-3 training
    # companions (derived caches, some holding 4 MiB of frozen features each).
    # Workflow construction already did this; a direct call gets the same guarantee.
    reclaimed = reclaim_derived_stage3_files(out)
    legacy = decision_shard_paths(records, splits, out)
    convertible = sum(path.stat().st_size for role in roles for path in legacy[role] if path.exists())
    # Per-question files plus the merged role file, at the most rows a question can have:
    # one per in-budget case, each with one action byte, plus one first-case bitmap.
    cases = decision_case_count(cfg)
    bitmap = (cases + 7) // 8
    upper = 2 * sum(len(splits[role]) for role in roles) * (cases * ((cfg.action_count + 7) // 8) + bitmap)
    free = shutil.disk_usage(out).free
    print(f"Stage 3 storage: {out.resolve()} | {free / 1024**3:.2f} GiB free; "
          f"removed {reclaimed / 1024**2:.2f} MiB of derived companions and incomplete writes; "
          f"{convertible / 1024**2:.2f} MiB of schema-3 audit shards will be converted and deleted. "
          f"Labels need at most {upper / 1024**2:.1f} MiB (one action byte per row and a "
          f"{bitmap}-byte first-case bitmap per question); frozen features, row sources, and "
          "audit fields are recomputed rather than saved.", flush=True)
    if upper > free + convertible:
        print(f"WARNING: Stage 3 labels may need up to {upper / 1024**2:.1f} MiB, but only "
              f"{(free + convertible) / 1024**2:.1f} MiB is free or reclaimable. Free space now; "
              "completed questions are reused on resume.", flush=True)
    if previous != manifest:
        atomic_json(manifest_path, manifest)
    labels, coverage = {}, {}
    for role in roles:
        labels[role] = decision_role_labels(role, records, splits[role], teacher, predictor,
                                            cfg, out, fingerprint, sync)
        coverage[role] = combine_coverage(labels[role].pop("coverage"))
        # Only head inputs stay in RAM: action bytes, first-case bitmaps, and offsets.
        del labels[role]["observation_sha256"]
        atomic_json(out / "decision_label_coverage.json", coverage)
        print(f"Stage 3 {role}: {coverage[role]['state_order_cases']} state/order cases, "
              f"{coverage[role]['action_rows']} shared action rows; "
              f"statuses {coverage[role]['status_counts']}; "
              f"actionable ranks {coverage[role]['actionable_by_target_rank']}.", flush=True)
        if not coverage[role]["action_rows"]:
            raise ValueError(f"No actionable Stage 3 rows for {role}: {coverage[role]}")
    held = sum(labels[role][key].numel() * labels[role][key].element_size()
               for role in roles for key in ("offsets", "preferred", "first_cases"))
    print(f"Stage 3 labels in RAM: {held / 1024**2:.1f} MiB for "
          f"{sum(coverage[role]['action_rows'] for role in roles)} rows.", flush=True)
    offsets = {role: labels[role]["offsets"].tolist() for role in roles}

    def question_rows(role, position):
        start, stop = offsets[role][position:position+2]
        # Row sources are rebuilt from the question's bitmap, never held for every row.
        source = decision_row_sources(labels[role]["first_cases"][position].numpy(), cfg)
        return decision_training_rows(records[splits[role][position]], source,
                                      labels[role]["preferred"][start:stop].numpy(), predictor, cfg)

    model = SupervisedActionHead(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.decision_lr, weight_decay=cfg.weight_decay)
    best, weights, stale, history = math.inf, None, 0, []
    rng = np.random.RandomState(cfg.seed + 301)
    progress_path = out / "decision_head_progress.pt"
    start_epoch = 0
    if cfg.resume and progress_path.exists():
        progress = load_checkpoint(progress_path)
        if progress.get("schema_version") != 1 or progress.get("fingerprint") != fingerprint:
            raise ValueError("Decision head progress has an incompatible training contract.")
        start_epoch = progress["epoch"]
        if not (0 < start_epoch <= cfg.decision_epochs and
                len(progress["history"]) == start_epoch and
                1 <= progress["best_epoch"] <= start_epoch and progress["stale"] >= 0):
            raise ValueError("Decision head progress has inconsistent epoch bookkeeping.")
        model.load_state_dict(progress["current_weights"])
        opt.load_state_dict(progress["optimizer"])
        rng.set_state(progress["rng_state"])
        best, weights, best_epoch = progress["dev_loss"], progress["best_weights"], progress["best_epoch"]
        stale, history = progress["stale"], progress["history"]
        del progress
        print(f"Stage 3 head resume: {start_epoch} completed epochs; "
              "restored optimizer, shuffle RNG, and best development checkpoint.", flush=True)
    print(f"Stage 3 head training: device={device}, up to {cfg.decision_epochs} epochs, "
          f"train batch={cfg.decision_batch_size}, dev batch={cfg.decision_eval_batch_size}. "
          "Labels are complete; epochs only re-encode stored observations with the frozen "
          "predictor.", flush=True)
    for epoch in range(start_epoch, cfg.decision_epochs):
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
            last_print = decision_progress(f"Stage 3 epoch {epoch+1} train", position, len(splits["policy"]),
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
                last_print = decision_progress(f"Stage 3 epoch {epoch+1} dev", position + 1, len(splits["dev"]),
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
            "schema_version": 1, "fingerprint": fingerprint, "epoch": epoch+1,
            "current_weights": cpu_state(model), "optimizer": opt.state_dict(),
            "rng_state": rng.get_state(), "best_weights": weights,
            "best_epoch": best_epoch, "dev_loss": best, "stale": stale, "history": history})
        print(f"Stage 3 epoch {epoch+1}/{cfg.decision_epochs}: train loss {train_total/train_count:.4f}, "
              f"dev loss {dev_loss:.4f}, best {best:.4f} (epoch {best_epoch}), "
              f"patience {stale}/{cfg.patience} | elapsed {(time.monotonic()-epoch_started)/60:.1f} min",
              flush=True)
        if sync is not None:
            sync.push(f"Stage 3: epoch {epoch+1} progress")
        if stale >= cfg.patience:
            break
    if predictor_fingerprint(predictor) != frozen_before:
        raise AssertionError("Frozen Stage 2 predictor changed during decision training.")
    return {"weights": weights, "best_epoch": best_epoch, "dev_loss": best,
            "coverage": coverage, "history": history,
            "input_dim": cfg.hidden_dim, "action_count": cfg.action_count,
            "action_names": action_names(cfg),
            "dataset_fingerprint": fingerprint}

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
        action = 1 if kind == "ZS" else 0 if kind == "R" else number+1
        nxt = raw_next_state(state, action, cfg)
        if (nxt is None or
                kind == "ZS" and (nxt.candidate_mask ^ state.candidate_mask) != 1 << (number-1) or
                kind == "R" and (nxt.evaluator_mask ^ state.evaluator_mask) != 1 << (number-1)):
            raise ValueError(f"Invalid fixed acquisition order at {item}: duplicates, missing evaluator, or out-of-order zero-shot.")
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


def evaluation_methods(cfg):
    return ("full", *cfg.fixed_acquisition_orders, "supervised")

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
            hidden, probabilities = predictor.predict(record, state)
            recognized_max = max_recognized(probabilities, state, cfg)
            if recognized_max and first_max_recognition is None:
                first_max_recognition = {"candidate": record.candidate_ids[selected_candidate(probabilities, state, cfg)],
                                         "cost": state_cost(state, cfg)}
            if policy == "supervised" and continue_after_max:
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
            elif policy == "supervised":
                action = greedy_action(head, hidden, valid, predictor.device)
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
            hidden, probabilities = predictor.predict(scenario, state)
            flags = [goal_condition(rank, order, scenario_safe, scenario_maximum, probabilities,
                                    full, state, cfg) for rank in range(cfg.n)]
            prefix = next((rank for rank, good in enumerate(flags) if not good), cfg.n)
            if prefix < best_prefix:
                lost = True
            best_prefix = max(best_prefix, prefix)
            valid = valid_actions(state, cfg)
            if not valid.any() or prefix == cfg.n:
                break
            state = advance(state, greedy_action(head, hidden, valid, predictor.device), cfg)
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


def question_report(record, scores, safe, maximum, predictor, cfg, head, sink=None):
    """Roll out one question once per method and zero-shot order; keep only report inputs.

    Each rollout's outcome streams to `sink` as it is produced. The question keeps its
    per-method means (question_summary), and per zero-shot order its stop reason,
    correctness, MAX recovery, and saved calls for paired comparisons, plus its
    continuation diagnostic; untraced rows are discarded.
    """
    methods = {}
    for method in evaluation_methods(cfg):
        rows = question_rollouts(record, scores, safe, maximum, predictor, cfg, head, method)
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
            "continuation": question_continuation(record, scores, safe, maximum, predictor, cfg, head)}


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

    Each method is compared with the full pool, then the learned head with each fixed order,
    on saved calls and top-1 over all questions and on exact MAX over MAX-present questions.
    """
    pairs = [(name, "full") for name in evaluation_methods(cfg) if name != "full"]
    pairs += [("supervised", name) for name in cfg.fixed_acquisition_orders]
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
    """Learned minus each baseline's top-1, paired within questions as in paired_interval."""
    return {baseline: bootstrap_difference(
                [np.mean([x - y for x, y in zip(question["methods"]["supervised"]["correct"],
                                                question["methods"][baseline]["correct"])])
                 for question in questions], cfg.bootstrap_samples, cfg.seed)
            for baseline in evaluation_methods(cfg) if baseline != "supervised"}


def policy_report(records, ids, teacher, predictor, cfg, head, title, evaluated=None, sink=None,
                  snapshot_cache=None):
    """One report's acquisition comparison and its questions' report inputs, in `ids` order.

    `evaluated` maps question indices to question_report results, so reports that share
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
                                         teacher["maximum"][idx], predictor, cfg, head, sink)
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
    print(f"{'Comparison':<40} | {'Saved calls/q':>22} | {'Top-1':>22} | {'Exact MAX':>22}")
    for pair, item in comparisons.items():
        print(f"{pair:<40} | {interval_text(item['saved_calls']):>22} | "
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
            "continuation": continuation_summary([question["continuation"] for question in questions],
                                                 cfg)}, questions


def question_rollout_audit(records, idx, teacher, predictor, cfg, head, out, benchmark):
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
                                     teacher["maximum"][idx], predictor, cfg, head, method, trace=True):
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
    """
    def __init__(self, out, repo, interval_minutes, api=None):
        self.out, self.repo, self.interval = Path(out), repo, interval_minutes * 60
        self.api = hub_api() if api is None else api
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
        for attempt in range(3):
            if attempt:
                time.sleep(5 * 2 ** (attempt - 1))
            try:
                self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
                self.api.upload_folder(folder_path=str(self.out), repo_id=self.repo, repo_type="dataset",
                                       commit_message=message, ignore_patterns=["*.tmp"],
                                       delete_patterns=["decision_labels/*/*.pt", "decision_shards/*/*",
                                                        LEGACY_TRAJECTORY_FILE])
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
        return (f"Mirror {self.repo}: {self.uploads} uploads; last upload: {self.last_message or 'none'}; "
                f"last error: {self.last_error or 'none'}.")


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
        self.sync = OutputSync(self.out, cfg.hf_repo, cfg.hf_sync_minutes) if cfg.hf_repo else None
        if self.sync is not None:
            state = self.sync.pull()
            print({"local": f"Mirror {cfg.hf_repo}: {self.out} already holds files; they are used as is.",
                   "new": f"Mirror {cfg.hf_repo}: no uploaded run yet; this run is uploaded as it progresses.",
                   "pulled": f"Mirror {cfg.hf_repo}: downloaded {folder_size(self.out) / 1024**2:.1f} MiB "
                             f"into {self.out}; completed stages resume."}[state], flush=True)
        # Logs are identified by the sources below, not by the configured paths.
        contract_cfg = {key: value for key, value in asdict(cfg).items() if key not in NON_CONTRACT_SETTINGS}
        # A log's filename is its benchmark identity; its directory and mtime are not.
        sources = [{"path": str(Path(p).resolve()), "name": Path(p).name, "bytes": Path(p).stat().st_size,
                    "sha256": file_digest(p)} for p in [cfg.train_file] + cfg.test_files]
        contract = {"schema_version": 4, "implementation_revision": PIPELINE_REVISION,
                    "config": contract_cfg, "sources": sources}
        path = self.out / "contract.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            # Round-trip converts tuples to JSON lists consistently.
            if any("sha256" not in source for source in old.get("sources", [])):
                # Before content hashing, a contract bound each log to its path, size, and
                # mtime, which hold only within one session; upgrade a matching one in place.
                legacy = {**contract,
                          "config": {**contract_cfg, "train_file": cfg.train_file,
                                     "test_files": list(cfg.test_files)},
                          "sources": [{"path": source["path"], "bytes": source["bytes"],
                                       "mtime_ns": Path(source["path"]).stat().st_mtime_ns}
                                      for source in sources]}
                if old != json.loads(json.dumps(legacy)):
                    raise ValueError("Output folder belongs to a different config/data contract. Choose a new output_dir.")
                atomic_json(path, contract)
            elif contract_identity(old) != contract_identity(contract):
                raise ValueError("Output folder belongs to a different config/data contract. Choose a new output_dir.")
            if not cfg.resume:
                raise ValueError("Output folder already contains this run. Set resume=True or choose a new folder.")
            # Before Stage 0 writes its reports: a resumed run that filled its
            # disk regains the space held by derived caches and partial writes.
            reclaimed = reclaim_derived_stage3_files(self.out)
            if reclaimed:
                print(f"Reclaimed {reclaimed / 1024**2:.2f} MiB of derived Stage 3 companions and "
                      f"incomplete writes in {self.out} before resuming.", flush=True)
        else:
            if any(self.out.iterdir()):
                raise ValueError("Use an empty output folder; existing unrecognized files will not be overwritten.")
            atomic_json(path, contract)
        self.records, self.splits = None, None

    def prepare(self):
        section(f"DATA AUDIT | device={self.device} | {self.cfg.n} candidates, {self.cfg.k} evaluators")
        # Names the embedded code: a notebook from before compact labels wrote audit shards.
        print(f"Implementation revision {PIPELINE_REVISION}; Stage 3 stores compact labels "
              f"(schema {DECISION_LABEL_SCHEMA}, one byte per row plus one bitmap per question), "
              "binds them to the input logs' content rather than their paths, and never writes "
              "audit shards; Stage 4 rolls out each question once and saves "
              f"compact outcomes to {ROLLOUT_FILE}, not trajectories. Stages 1-2 train on every "
              f"presence state under every zero-shot order ({len(presence_states(self.cfg.zero_shots, self.cfg.k)):,} "
              f"x {math.factorial(self.cfg.zero_shots)} views per question), assembled on the "
              "device and never stored.", flush=True)
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
                       self.heuristics, "TEACHER DEVELOPMENT REPORT (used for model selection)")
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

    def train_decision_head(self):
        if not hasattr(self, "predictor"):
            self.train_snapshot()
        section("STAGE 3 | Supervised acquisition labels and frozen-encoder head")
        path = self.out / "decision_head_completed.pt"
        # Files from earlier code, whose fingerprint hashed the log paths, are rebound first.
        fingerprint = upgrade_decision_fingerprints(self.records, self.splits, self.teacher,
                                                    self.predictor, self.cfg, self.out)
        if self.cfg.resume and path.exists():
            checkpoint = load_checkpoint(path)
            if (checkpoint.get("action_count") != self.cfg.action_count or
                    checkpoint.get("dataset_fingerprint") != fingerprint):
                raise ValueError("Completed decision head has an incompatible dataset or action contract.")
            print("Loaded completed supervised decision head.")
        else:
            checkpoint = train_decision_head(self.records, self.splits, self.teacher,
                                             self.predictor, self.cfg, self.device, self.out, self.sync)
            atomic_torch(path, checkpoint)
        self.head = SupervisedActionHead(self.cfg).to(self.device)
        self.head.load_state_dict(checkpoint["weights"])
        self.head.eval()
        self.decision_checkpoint = checkpoint
        if not all(torch.equal(value.cpu(), self.predictor.model.state_dict()[key].cpu())
                   for key, value in load_checkpoint(self.out / "snapshot_completed.pt")["weights"].items()):
            raise AssertionError("Frozen snapshot predictor changed during head training.")
        atomic_torch(self.out / "inference_bundle.pt", {
            "schema_version": 4, "implementation_revision": PIPELINE_REVISION,
            "config": asdict(self.cfg),
            "snapshot": load_checkpoint(self.out / "snapshot_completed.pt"),
            "decision_head": checkpoint,
            "action_names": action_names(self.cfg),
            "cost_note": "Fixed-m cached bundles; no target-grading cost at deployment."})
        print(f"Decision head: epoch {checkpoint['best_epoch']}, "
              f"development loss {checkpoint['dev_loss']:.4f}.")
        self.push("Stage 3 complete: decision head and inference bundle")
        return self

    def decision_label_audit(self, role, position):
        """Full planner rows and cases for one saved Stage 3 question, recomputed."""
        if not hasattr(self, "predictor"):
            self.train_snapshot()
        return decision_question_audit(self.records, self.splits, self.teacher, self.predictor,
                                       self.cfg, self.out, role, position)

    def rollout_audit(self, name, position):
        """Traced Stage 4 rollouts of one report question (`audit` or a benchmark), recomputed."""
        if not hasattr(self, "head"):
            self.train_decision_head()
        return question_rollout_audit(self.records, self.splits[name][position], self.teacher,
                                      self.predictor, self.cfg, self.head, self.out, name)

    def report(self):
        if not hasattr(self, "head"):
            self.train_decision_head()
        results = {"evaluation_protocol": split_protocol(self.cfg, self.splits),
                   "decision_training": {"best_epoch": self.decision_checkpoint["best_epoch"],
                                         "dev_loss": self.decision_checkpoint["dev_loss"],
                                         "coverage": self.decision_checkpoint["coverage"],
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
                                                      self.cfg, self.head, f"ACQUISITION | {name}",
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


def load_inference_bundle(path, device="cpu"):
    bundle = load_checkpoint(path)
    if bundle.get("schema_version") != 4 or bundle.get("implementation_revision") != PIPELINE_REVISION:
        raise ValueError(f"Expected a version 4 inference bundle with revision {PIPELINE_REVISION} training contract; old bundles are incompatible.")
    # decision_cache_mb was a runtime-only Stage 3 setting in earlier revision-5 bundles.
    cfg = Config(**{key: value for key, value in bundle["config"].items() if key != "decision_cache_mb"})
    cfg.validate()
    if (bundle.get("action_names") != action_names(cfg) or
            bundle["decision_head"].get("action_count") != cfg.action_count or
            bundle["decision_head"].get("input_dim") != cfg.hidden_dim):
        raise ValueError("Inference bundle has an incompatible decision action/input contract.")
    predictor = FrozenPredictor(bundle["snapshot"], cfg, torch.device(device))
    head = SupervisedActionHead(cfg).to(device)
    head.load_state_dict(bundle["decision_head"]["weights"])
    head.eval()
    return cfg, predictor, head


def run_pipeline(cfg=None):
    work = Workflow(cfg or Config())
    work.prepare().train_teacher().train_snapshot().train_decision_head()
    work.report()
    return work


# %% CLI (the standalone notebook runs each stage in its own cell)
if __name__ == "__main__":
    run_pipeline()
