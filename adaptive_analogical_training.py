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
from torch.utils.data import DataLoader, TensorDataset

PIPELINE_REVISION = 5  # Ordered evaluator acquisition and revised supervised goals.


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
    output_dir: str = "/kaggle/working/adaptive_analogical_shared_training_v5_run"
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
    batch_size: int = 128
    learning_rate: float = .0003
    weight_decay: float = 1e-4
    teacher_epochs: int = 100
    snapshot_epochs: int = 100
    patience: int = 15
    snapshots_per_query: int = 24
    dev_snapshots_per_query: int = 16
    snapshot_reachable_fraction: float = .50
    snapshot_missing_fraction: float = .25
    snapshot_permutation_fraction: float = .10
    candidate_mask_fraction: float = .25
    evaluator_mask_fraction: float = .25
    hidden_source_fraction: float = .10
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
        assert self.snapshots_per_query >= 2 and self.dev_snapshots_per_query >= 2
        assert 0 <= self.snapshot_reachable_fraction <= 1
        assert 0 <= self.snapshot_missing_fraction <= 1
        assert 0 <= self.snapshot_permutation_fraction <= 1
        assert all(0 <= p <= 1 for p in (self.candidate_mask_fraction,
                                         self.evaluator_mask_fraction,
                                         self.hidden_source_fraction))
        assert self.print_every > 0
        assert self.hf_sync_minutes > 0
        if self.hf_repo is not None and not re.fullmatch(r"[\w.-]+/[\w.-]+", str(self.hf_repo)):
            raise ValueError("hf_repo must be a Hugging Face dataset repository id such as 'user/name'.")
        fixed_acquisition_sequences(self)


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


def make_loader(x, y, batch_size, shuffle=True):
    # Keep every complete-state example while avoiding singleton BatchNorm batches.
    size = min(batch_size, len(x))
    if size < 2 and shuffle:
        raise ValueError("BatchNorm training requires at least two samples.")
    order = torch.randperm(len(x)).tolist() if shuffle else list(range(len(x)))
    batches = [order[start:start + max(1, size)] for start in range(0, len(x), max(1, size))]
    if shuffle and len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_sampler=batches)


def predict_array(model, x, device, batch=1024, hidden=False):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            t = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device)
            values.append((model.encode(t) if hidden else model(t)).cpu().numpy())
    return np.concatenate(values)


def train_teacher(records, train_ids, val_ids, cfg, device, seed, name):
    seed_everything(seed)
    full = State((1 << cfg.n)-1, (1 << cfg.k)-1)
    vx = np.stack([observation(records[i], full, cfg) for i in val_ids])
    model = ResNet(cfg.input_dim, cfg.n, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best, best_state, stale, best_epoch = -1.0, None, 0, 0
    for epoch in range(cfg.teacher_epochs):
        x, targets = build_snapshots(records, train_ids, None, cfg,
                                     cfg.snapshots_per_query, seed + epoch,
                                     epoch=epoch, augmented=True)
        loader = make_loader(x, targets, cfg.batch_size)
        model.train()
        for bx, by in loader:
            opt.zero_grad(set_to_none=True)
            present = by[:, cfg.n:].to(device)
            raw = nn.functional.binary_cross_entropy_with_logits(
                model(bx.to(device)), by[:, :cfg.n].to(device), reduction="none")
            loss = ((raw * present).sum(1) / present.sum(1).clamp_min(1)).mean()
            loss.backward()
            opt.step()
        pred = predict_array(model, vx, device)
        score = np.mean([ap_score(ranked_indices(p, retrieval_order(cfg)), records[i].labels)
                         for p, i in zip(pred, val_ids)])
        if score > best + 1e-8:
            best, best_state, stale, best_epoch = score, cpu_state(model), 0, epoch + 1
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    print(f"{name:<24} | epochs {epoch+1:3d} | best {best_epoch:3d} | development AP {best:.4f}")
    return {"weights": best_state, "best_epoch": best_epoch,
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
    cm = present_mask(state, cfg).astype(np.float32)
    return np.r_[record.labels, safe, maximum,
                 float(np.any(safe * cm)), float(np.any(maximum * cm)), cm].astype(np.float32)


def augment_structure(state, cfg, rng):
    """Independently hide candidates/evaluators, including occasional OS sources."""
    state = as_snapshot_state(state, cfg)
    candidates, evaluators = state.candidates, state.evaluators
    for i in range(cfg.n):
        if candidates & (1 << i) and rng.rand() < cfg.candidate_mask_fraction:
            candidates &= ~(1 << i)
    for j in range(cfg.k):
        if evaluators & (1 << j) and rng.rand() < cfg.evaluator_mask_fraction:
            evaluators &= ~(1 << j)
            if candidates & (1 << (cfg.zero_shots + j)) and rng.rand() >= cfg.hidden_source_fraction:
                candidates &= ~(1 << (cfg.zero_shots + j))
    if not candidates:
        original = np.flatnonzero(bit_array(state.candidates, cfg.n))
        i = int(rng.choice(original))
        candidates = 1 << i
        if i >= cfg.zero_shots:
            evaluators |= 1 << (i - cfg.zero_shots)
    cells = sum(evaluators << (i * cfg.k) for i in range(cfg.n) if candidates & (1 << i))
    result = SnapshotState(candidates, evaluators, evaluators, evaluators,
                           evaluators, evaluators, cells, cells)
    validate_snapshot_state(result, cfg, allow_hidden_source=True)
    return result

def augment_measurements(state, cfg, rng):
    """Hide real measurements or mark recorded calls as failed; never invent values."""
    state = as_snapshot_state(state, cfg)
    attempted, observed = state.ccs_attempted, state.ccs_observed
    eligible = np.flatnonzero(bit_array(attempted, cfg.n * cfg.k))
    if len(eligible):
        pattern = int(rng.randint(5))
        if pattern == 0:                   # Independent sparse cells.
            chosen = rng.choice(eligible, max(1, len(eligible)//2), replace=False)
        elif pattern == 1:                 # Whole candidate row.
            row = int(rng.choice(eligible)) // cfg.k
            chosen = eligible[eligible // cfg.k == row]
        elif pattern == 2:                 # Whole evaluator column.
            col = int(rng.choice(eligible)) % cfg.k
            chosen = eligible[eligible % cfg.k == col]
        elif pattern == 3:                 # No successful CCS.
            chosen = eligible
        else:                              # Exactly one missing cell.
            chosen = rng.choice(eligible, 1)
        remove = bits_from_array(np.isin(np.arange(cfg.n * cfg.k), chosen))
        observed &= ~remove
        if rng.rand() < .5:                # Unattempted rather than failed.
            attempted &= ~remove
    base_observed, base_attempted = state.baseline_observed, state.baseline_attempted
    base_cells = np.flatnonzero(bit_array(base_observed, cfg.k))
    if len(base_cells) and rng.rand() < .35:
        bit = 1 << int(rng.choice(base_cells))
        base_observed &= ~bit
        if rng.rand() < .5:
            base_attempted &= ~bit
    similarity = state.similarity_observed
    sim_cells = np.flatnonzero(bit_array(similarity, cfg.k))
    if len(sim_cells) and rng.rand() < .15:
        similarity &= ~(1 << int(rng.choice(sim_cells)))
    result = SnapshotState(state.candidates, state.retrieved, state.evaluators,
                           similarity, base_attempted, base_observed, attempted, observed)
    validate_snapshot_state(result, cfg, allow_hidden_source=True)
    return result


def permute_snapshot(record, state, safe, maximum, cfg, rng):
    """Apply the same slot permutation to evidence, provenance and labels."""
    state = as_snapshot_state(state, cfg)
    eorder = rng.permutation(cfg.k)
    corder = np.r_[rng.permutation(cfg.zero_shots), cfg.zero_shots + eorder]
    def remap(bits, size, order):
        return bits_from_array(bit_array(bits, size)[order])
    cell_order = (corder[:, None] * cfg.k + eorder[None, :]).ravel()
    new_state = SnapshotState(remap(state.candidates, cfg.n, corder),
                              remap(state.retrieved, cfg.k, eorder),
                              remap(state.evaluators, cfg.k, eorder),
                              remap(state.similarity_observed, cfg.k, eorder),
                              remap(state.baseline_attempted, cfg.k, eorder),
                              remap(state.baseline_observed, cfg.k, eorder),
                              remap(state.ccs_attempted, cfg.n*cfg.k, cell_order),
                              remap(state.ccs_observed, cfg.n*cfg.k, cell_order))
    new_record = Record(record.uid, record.group, record.benchmark,
                        [record.candidate_ids[i] for i in corder],
                        [record.evaluator_ids[i] for i in eorder],
                        record.similarity[eorder], record.baseline[eorder],
                        record.ccs[np.ix_(corder, eorder)], record.labels[corder])
    validate_snapshot_state(new_state, cfg, allow_hidden_source=True)
    return new_record, new_state, safe[corder], maximum[corder]


def build_snapshots(records, ids, teacher, cfg, per_query, seed, epoch=0, augmented=False):
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    reachable = reachable_states(cfg)
    catalog = structural_catalog(cfg.zero_shots, cfg.k) if augmented else ()
    others = per_query - 1
    n_reachable = round(others * cfg.snapshot_reachable_fraction) if augmented else others
    for rank, idx in enumerate(ids):
        states = [State((1 << cfg.n)-1, (1 << cfg.k)-1)]
        for j in range(n_reachable):
            position = (epoch * len(ids) + rank) * max(1, n_reachable) + j
            states.append(reachable[(position * 53) % len(reachable)])
        for j in range(others - n_reachable):
            position = ((epoch * len(ids) + rank) * max(1, others - n_reachable) + j)
            candidates, retrieved, evaluators = catalog[(position * 7919) % len(catalog)]
            states.append(complete_snapshot_state(candidates, retrieved, evaluators, cfg))
        for position, state in enumerate(states):
            r = records[idx]
            safe = teacher["safe"][idx] if teacher is not None else None
            maximum = teacher["maximum"][idx] if teacher is not None else None
            if augmented and position:
                state = augment_structure(state, cfg, rng)
                if rng.rand() < cfg.snapshot_missing_fraction:
                    state = augment_measurements(state, cfg, rng)
                if rng.rand() < cfg.snapshot_permutation_fraction:
                    if teacher is None:
                        dummy = np.zeros(cfg.n, np.float32)
                        r, state, _, _ = permute_snapshot(r, state, dummy, dummy, cfg, rng)
                    else:
                        r, state, safe, maximum = permute_snapshot(r, state, safe, maximum, cfg, rng)
            xs.append(observation(r, state, cfg, allow_hidden_source=augmented))
            ys.append((np.r_[r.labels, present_mask(state, cfg)].astype(np.float32)
                       if teacher is None else snapshot_target(r, safe, maximum, state, cfg)))
    return np.stack(xs), np.stack(ys)


def snapshot_loss(logits, targets, cfg):
    n = cfg.n
    mask = targets[:, 3*n + 2:]
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets[:, :3*n+2], reduction="none")
    losses = [((raw[:, b*n:(b+1)*n] * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
              for b in range(3)]
    return losses[0] + cfg.auxiliary_weight * (losses[1] + losses[2] + raw[:, 3*n:].mean(0).sum())


def snapshot_summary(logits, targets, cfg):
    mask = targets[:, 3*cfg.n+2:].astype(bool)
    chosen = np.where(mask, logits[:, :cfg.n], -np.inf).argmax(1)
    selected = targets[np.arange(len(targets)), chosen]
    p = 1 / (1 + np.exp(-np.clip(logits[:, :cfg.n], -40, 40)))
    return {"snapshots": len(targets), "top1": float(selected.mean()),
            "brier": float(((p - targets[:, :cfg.n])**2)[mask].mean())}


def temperature_vector(temperatures, cfg):
    return np.r_[np.repeat(temperatures[:3], cfg.n), temperatures[3:]].astype(np.float32)


def fit_temperatures(logits, targets, cfg):
    temps = np.ones(5, np.float32)
    if not cfg.calibrate:
        return temps
    masks = targets[:, 3*cfg.n+2:].astype(bool)
    for block in range(5):
        sl = slice(block*cfg.n, (block+1)*cfg.n) if block < 3 else slice(3*cfg.n+block-3, 3*cfg.n+block-2)
        z, y = logits[:, sl], targets[:, sl]
        if block < 3:
            z, y = z[masks], y[masks]
        candidates = np.geomspace(.25, 4., 41)
        nlls = [np.mean(np.logaddexp(0, z/t) - y*z/t) for t in candidates]
        temps[block] = candidates[int(np.argmin(nlls))]
    return temps


def fit_snapshot(records, splits, teacher, cfg, device):
    seed_everything(cfg.seed + 200)
    vx, vy = build_snapshots(records, splits["dev"], teacher, cfg,
                             cfg.dev_snapshots_per_query, cfg.seed + 202)
    model = ResNet(cfg.input_dim, 3*cfg.n+2, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best, weights, stale, history = (-1., -math.inf), None, 0, []
    catalog_size = len(structural_catalog(cfg.zero_shots, cfg.k))
    print(f"Snapshot training: {len(reachable_states(cfg))} application states; "
          f"{catalog_size} coherent structures; {cfg.snapshots_per_query} samples/question/epoch.")
    for epoch in range(cfg.snapshot_epochs):
        x, y = build_snapshots(records, splits["supervised"], teacher, cfg,
                               cfg.snapshots_per_query, cfg.seed + 201 + epoch,
                               epoch=epoch, augmented=True)
        loader = make_loader(x, y, cfg.batch_size)
        model.train()
        total = 0.
        for bx, by in loader:
            opt.zero_grad(set_to_none=True)
            loss = snapshot_loss(model(bx.to(device)), by.to(device), cfg)
            loss.backward()
            opt.step()
            total += float(loss.item())
        logits = predict_array(model, vx, device)
        metrics = snapshot_summary(logits, vy, cfg)
        candidate = (metrics["top1"], -metrics["brier"])
        history.append({"epoch": epoch + 1, "loss": total/len(loader), **metrics})
        if candidate > best:
            best, weights, stale, best_epoch = candidate, cpu_state(model), 0, epoch + 1
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % cfg.print_every == 0:
            print(f"Snapshot epoch {epoch+1:3d} | loss {total/len(loader):.4f} | "
                  f"development Top-1 {metrics['top1']:.1%} | Brier {metrics['brier']:.4f}")
        if stale >= cfg.patience:
            break
    model.load_state_dict(weights)
    logits = predict_array(model, vx, device)
    temps = fit_temperatures(logits, vy, cfg)
    print(f"Snapshot selected epoch {best_epoch}; development Top-1 {best[0]:.1%}.")
    off_per_query = (cfg.snapshots_per_query - 1) - round(
        (cfg.snapshots_per_query - 1) * cfg.snapshot_reachable_fraction)
    covered = min(catalog_size, len(history) * len(splits["supervised"]) * off_per_query)
    print(f"Coherent structural catalog visited during training: {covered}/{catalog_size} "
          f"(shared across questions; development remains reachable-only).")
    return {"weights": weights, "temperatures": temps, "history": history,
            "best_epoch": best_epoch, "input_dim": cfg.input_dim,
            "structural_catalog_size": catalog_size, "structural_visited": covered,
            "dev_summary": snapshot_summary(logits / temperature_vector(temps, cfg), vy, cfg)}


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


def decision_dataset_fingerprint(records, splits, teacher, predictor, cfg):
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema": 3, "revision": PIPELINE_REVISION,
                              "config": {key: value for key, value in asdict(cfg).items()
                                         if key not in {"resume", "output_dir", "device", "cpu_threads", "print_every",
                                                        "fixed_acquisition_orders", "continue_after_max", "decision_eval_batch_size",
                                                        "hf_repo", "hf_sync_minutes"}},
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
# schema 4 (an int32 source per row) converts in place. The label fingerprint is unchanged.
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
    fingerprint = decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
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


def decision_progress(label, completed, total, started, last_print, cfg, detail=""):
    now = time.monotonic()
    if completed == 1 or completed == total or completed % cfg.print_every == 0 or now - last_print >= 30:
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
    fingerprint = decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
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
    difference = np.asarray([np.mean(values) for values in grouped.values()], float)
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
            "acquisition_complete": (state.candidate_mask == (1 << cfg.n)-1 and
                                     state.evaluator_mask == (1 << cfg.k)-1),
            "final_max_recognized": max_recognized(probabilities, state, cfg),
            "correct": int(record.labels[selected]), "has_max": has_max,
            "exact_max": exact_max, "cost": state_cost(state, cfg),
            "full_cost": cfg.full_cost,
            "call_counts": acquisition_call_counts(bin(state.candidate_mask).count("1"),
                                                   bin(state.evaluator_mask).count("1"), cfg),
            "full_call_counts": acquisition_call_counts(cfg.n, cfg.k, cfg),
            "saved_fraction": 1 - state_cost(state, cfg)/cfg.full_cost,
            "reason": reason, "trajectory": trajectory}

def evaluate_policy(records, ids, teacher, predictor, cfg, head=None,
                    policy="supervised", trace=False):
    rows = []
    for idx in ids:
        record = records[idx]
        for order in permutations(range(cfg.zero_shots)):
            scenario, _, _, maximum = permute_zero_shots(
                record, teacher["scores"][idx], teacher["safe"][idx],
                teacher["maximum"][idx], order, cfg)
            # Evaluation measures the first predicted MAX stop, regardless of
            # the application wrapper's configurable continuation behavior.
            row = rollout(scenario, maximum, predictor, cfg, policy, head, trace,
                          continue_after_max=False)
            row["zero_shot_order"] = list(order)
            rows.append(row)
    return rows

def call_savings_summary(question_means):
    groups = {"all": question_means,
              "max_present": [row for row in question_means if row["has_max"]]}
    result = {}
    for name, group in groups.items():
        with_max = [row for row in group if row["has_max"]]
        metrics = {"n": len(group),
                   "exact_max_recovery": float(np.mean([row["exact_max"] for row in with_max])) if with_max else None,
                   "predicted_max_stop_rate": float(np.mean([row["predicted_max"] for row in group])) if group else None,
                   "wrong_predicted_max_stop_rate": float(np.mean([row["wrong_predicted_max"] for row in group])) if group else None}
        stop_weight = sum(row["predicted_max"] for row in group)
        metrics["predicted_max_stop_precision"] = (
            float(sum(row["predicted_max"] - row["wrong_predicted_max"] for row in group) / stop_weight)
            if stop_weight else None)
        for unit in ("solver_calls", "total_calls"):
            used = np.asarray([row["call_counts"][unit] for row in group])
            full = np.asarray([row["full_call_counts"][unit] for row in group])
            metrics[unit] = {"mean_full": float(full.mean()) if group else None,
                             "mean_used": float(used.mean()) if group else None,
                             "mean_saved": float((full-used).mean()) if group else None,
                             "total_saved": float((full-used).sum()) if group else None,
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


def policy_summary(rows):
    if not rows:
        return None
    by_question = {}
    for row in rows:
        by_question.setdefault(row["uid"], []).append(row)
    means = [{"correct": np.mean([r["correct"] for r in group]),
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
              "has_max": group[0]["has_max"],
              "exact_max": np.mean([r["exact_max"] for r in group]) if group[0]["has_max"] else None,
              "predicted_max": np.mean([r["reason"] == "predicted_max" for r in group]),
              "max_recognized": np.mean([r.get("first_max_recognition") is not None or
                                         r["reason"] == "predicted_max" for r in group]),
              "wrong_predicted_max": np.mean([r["reason"] == "predicted_max" and not r["exact_max"] for r in group]),
              "false_max": np.mean([r["reason"] == "predicted_max" for r in group])
              if not group[0]["has_max"] else None}
             for group in by_question.values()]
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
            "stop_reasons": dict(Counter(r["reason"] for r in rows))}


def savings_stop_category(row):
    if row["reason"] != "predicted_max":
        return "other"
    return "correct_max" if row["exact_max"] else "wrong_max"

def continuation_diagnostic(records, ids, teacher, predictor, cfg, head):
    reached, broken = [], []
    final_prefixes, completed = [], []
    per_rank, per_rank_final = [], []
    for idx in ids:
        question_prefix, question_lost = [], []
        question_final_prefix, question_complete = [], []
        for zero_order in permutations(range(cfg.zero_shots)):
            record, _, safe, maximum = permute_zero_shots(
                records[idx], teacher["scores"][idx], teacher["safe"][idx],
                teacher["maximum"][idx], zero_order, cfg)
            order = permuted_reference_order(teacher["scores"][idx], zero_order, cfg)
            full_cfg = copy.deepcopy(cfg)
            full_cfg.max_cost = None
            _, full = predictor.predict(record, State((1 << cfg.n)-1, (1 << cfg.k)-1), full_cfg)
            state, best_prefix, lost = State(), 0, False
            while True:
                hidden, probabilities = predictor.predict(record, state)
                flags = [goal_condition(rank, order, safe, maximum, probabilities,
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
        reached.append(float(np.mean(question_prefix)))
        broken.append(float(np.mean(question_lost)))
        final_prefixes.append(float(np.mean(question_final_prefix)))
        completed.append(float(np.mean(question_complete)))
        per_rank.append([float(np.mean(np.asarray(question_prefix) >= rank))
                          for rank in range(1, cfg.n+1)])
        per_rank_final.append([float(np.mean(np.asarray(question_final_prefix) >= rank))
                               for rank in range(1, cfg.n+1)])
    return {"n": len(ids), "orders_per_question": math.factorial(cfg.zero_shots),
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

def stage_snapshot_audit(records, ids, teacher, predictor, cfg):
    if not ids:
        return None
    x, y = build_snapshots(records, ids, teacher, cfg, cfg.dev_snapshots_per_query,
                           cfg.seed + 401)
    logits = predict_array(predictor.model, x, predictor.device) / predictor.temperatures
    metrics = snapshot_summary(logits, y, cfg)
    p = 1/(1+np.exp(-np.clip(logits, -40, 40)))
    for name, col in (("safe", 3*cfg.n), ("max", 3*cfg.n+1)):
        positive = p[:, col] >= cfg.recognition_threshold
        true = y[:, col].astype(bool)
        metrics[name + "_precision"] = float(true[positive].mean()) if positive.any() else None
        metrics[name + "_recall"] = float(positive[true].mean()) if true.any() else None
    sx, sy = build_snapshots(records, ids, teacher, cfg, cfg.dev_snapshots_per_query,
                             cfg.seed + 402, augmented=True)
    stress = predict_array(predictor.model, sx, predictor.device) / predictor.temperatures
    metrics["augmented_stress"] = snapshot_summary(stress, sy, cfg)
    return metrics

def policy_report(records, ids, teacher, predictor, cfg, head, title):
    section(title)
    if not ids:
        print("No eligible questions; metrics are unavailable.")
        return None, {}
    rows = {name: evaluate_policy(records, ids, teacher, predictor, cfg, head,
                                  policy=name, trace=True)
            for name in evaluation_methods(cfg)}
    summary = {name: policy_summary(data) for name, data in rows.items()}
    print(f"{'Method':<26} | {'Top-1 correct':>13} | {'Exact MAX':>9} | {'Mean cost':>10} | {'Saved':>7}")
    for name, item in summary.items():
        exact = "--" if item["exact_max_recovery"] is None else f"{item['exact_max_recovery']:.1%}"
        print(f"{name:<26} | {item['top1_correct']:>12.1%} | {exact:>9} | "
              f"{item['mean_cost']:>10.1f} | {item['mean_saved_fraction']:>6.1%}")
    print("MAX recognized during acquisition: " +
          ", ".join(f"{name}={item['max_recognition_rate']:.1%}" for name, item in summary.items()))
    print(f"Cost unit: {cfg.cost_unit}. MAX recognition uses predicted signals, not historical MAX labels.")
    print("Evaluation stops each learned and fixed rollout at predicted MAX; "
          "the application wrapper may choose a different stopping point.")
    print("API-call savings versus the complete pool (generation + measurement solves + their graders):")
    print(f"{'Method':<26} | {'Questions':<12} | {'N':>5} | {'Mean API':>9} | {'Saved/q':>9} | {'True MAX/q':>10} | {'Wrong MAX/q':>11} | {'Other/q':>9} | {'Saved':>7} | {'Exact MAX':>9}")
    for name, item in summary.items():
        for cohort, metrics in item["call_savings"].items():
            calls = metrics["total_calls"]
            values = ["--" if calls[key] is None else f"{calls[key]:.1f}"
                      for key in ("mean_used", "mean_saved", "mean_saved_by_correct_max_stop",
                                  "mean_saved_by_wrong_max_stop", "mean_saved_by_other_stop")]
            saved = "--" if calls["saved_fraction"] is None else f"{calls['saved_fraction']:.1%}"
            exact = "--" if metrics["exact_max_recovery"] is None else f"{metrics['exact_max_recovery']:.1%}"
            print(f"{name:<26} | {cohort:<12} | {metrics['n']:>5} | {values[0]:>9} | {values[1]:>9} | {values[2]:>10} | {values[3]:>11} | {values[4]:>9} | {saved:>7} | {exact:>9}")
    print("Saved/q = True MAX/q + Wrong MAX/q + Other/q; each uses all questions in that row as the denominator.")
    print("Other/q includes budget limits and ended custom sequences; only True MAX/q reflects a correct MAX-triggered stop.")
    print("On predicted-MAX stops only (including incorrect stops):")
    print(f"{'Method':<26} | {'Questions':<12} | {'MAX stop %':>10} | {'API/stop':>9} | {'Saved/stop':>10} | {'Correct MAX':>11}")
    for name, item in summary.items():
        for cohort, metrics in item["call_savings"].items():
            calls = metrics["total_calls"]
            values = ["--" if calls[key] is None else f"{calls[key]:.1f}"
                      for key in ("mean_used_on_predicted_max_stop", "mean_saved_on_predicted_max_stop")]
            rate = "--" if metrics["predicted_max_stop_rate"] is None else f"{metrics['predicted_max_stop_rate']:.1%}"
            precision = "--" if metrics["predicted_max_stop_precision"] is None else f"{metrics['predicted_max_stop_precision']:.1%}"
            print(f"{name:<26} | {cohort:<12} | {rate:>10} | {values[0]:>9} | {values[1]:>10} | {precision:>11}")
    return {"policies": summary, "cost_unit": cfg.cost_unit,
            "fixed_acquisition_sequences": fixed_acquisition_sequences(cfg),
            "learned_continues_after_max": False,
            "inference_continues_after_max_default": cfg.continue_after_max,
            "early_stop_rule": {"reason": "predicted_max", "threshold": cfg.recognition_threshold,
                                "signals": ["global_MAX", "global_SAFE", "selected_MAX", "selected_SAFE"]},
            "call_accounting": "Recorded generation, baseline/cross measurement solves, and their grading calls; excludes retrieval and target-answer grading.",
            "snapshot_diagnostics": stage_snapshot_audit(records, ids, teacher, predictor, cfg),
            "continuation": continuation_diagnostic(records, ids, teacher, predictor, cfg, head)}, rows

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
    files from the mirror once a role is merged. Run one session per repo.
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
                                   ignore_patterns=[".gitattributes"])
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
                                       delete_patterns=["decision_labels/*/*.pt", "decision_shards/*/*"])
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
        contract_cfg = asdict(cfg)
        for key in ("resume", "output_dir", "device", "cpu_threads", "print_every", "fixed_acquisition_orders",
                    "continue_after_max",
                    "decision_eval_batch_size", "hf_repo", "hf_sync_minutes",
                    "train_file", "test_files"):  # Logs are identified by the sources below.
            contract_cfg.pop(key)
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
              f"(schema {DECISION_LABEL_SCHEMA}, one byte per row plus one bitmap per question) "
              "and never writes audit shards.", flush=True)
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
        if self.cfg.resume and path.exists():
            checkpoint = load_checkpoint(path)
            if (checkpoint.get("action_count") != self.cfg.action_count or
                    checkpoint.get("dataset_fingerprint") != decision_dataset_fingerprint(
                        self.records, self.splits, self.teacher, self.predictor, self.cfg)):
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
        trajectory_path = self.out / "final_trajectories.jsonl"
        tmp = trajectory_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for name in names:
                ids = self.splits[name]
                ranking = teacher_report(self.records, ids, self.cfg,
                                         self.teacher["scores"], self.heuristics,
                                         f"RANKING | {name} ({len(ids)} eligible questions)")
                policy, rows = policy_report(self.records, ids, self.teacher, self.predictor,
                                             self.cfg, self.head,
                                             f"ACQUISITION | {name}")
                results[name] = {"ranking": ranking, "adaptive": policy}
                if rows:
                    results[name]["paired_top1"] = {
                        baseline: paired_interval(rows["supervised"], rows[baseline],
                                                   self.cfg.bootstrap_samples, self.cfg.seed)
                        for baseline in rows if baseline != "supervised"}
                for method, method_rows in rows.items():
                    for row in method_rows:
                        handle.write(json.dumps({"benchmark": name, "method": method, **row},
                                                allow_nan=False) + "\n")
        os.replace(tmp, trajectory_path)
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
        print(f"Saved reports, trajectories, checkpoints, and inference bundle to {self.out}")
        print("These are cached historical outcomes, not live API or Kaggle evidence.")
        self.results = results
        self.push("Stage 4 complete: reports and trajectories")
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
