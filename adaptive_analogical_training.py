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

PIPELINE_REVISION = 4  # Shared training questions across all three stages.


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
    output_dir: str = "/kaggle/working/adaptive_analogical_shared_training_v4_run"
    resume: bool = True             # Reuse completed stages with identical contracts.
    seed: int = 75
    device: str = "auto"           # CUDA when available (including Kaggle), otherwise CPU.
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
    cost_unit: str = "solver_calls" # solver_calls / total_calls (incl. evaluator graders)
    max_cost: Optional[float] = None # None = cost of the complete pool.
    decision_hidden: int = 64
    decision_epochs: int = 100
    decision_lr: float = 1e-4
    decision_batch_size: int = 128
    decision_eval_batch_size: int = 4096  # Frozen dev forwards; no optimizer changes.
    decision_cache_mb: int = 4096    # Shared CPU tensor cap; 0 streams memory-mapped shards.
    # Evaluation only: built-in order names or explicit [ZS1, R1, OS1, ...] sequences.
    fixed_acquisition_orders: dict = field(default_factory=lambda: {
        "fixed": "interleaved",
        "fixed_evaluators_first": "evaluators_first",
        "fixed_zero_shots_first": "zero_shots_first",
    })
    bootstrap_samples: int = 1000
    print_every: int = 10

    @property
    def n(self):
        return self.zero_shots + self.k

    @property
    def action_count(self):
        return 2 * self.k + 1

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
        assert self.decision_eval_batch_size >= 2 and self.decision_cache_mb >= 0
        assert self.snapshots_per_query >= 2 and self.dev_snapshots_per_query >= 2
        assert 0 <= self.snapshot_reachable_fraction <= 1
        assert 0 <= self.snapshot_missing_fraction <= 1
        assert 0 <= self.snapshot_permutation_fraction <= 1
        assert all(0 <= p <= 1 for p in (self.candidate_mask_fraction,
                                         self.evaluator_mask_fraction,
                                         self.hidden_source_fraction))
        assert self.print_every > 0
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


def resolve_training_device(requested="auto"):
    """Choose one shared device for all three neural training stages."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable. Enable Kaggle Settings > "
                             "Accelerator > GPU, or set device='auto' for CPU fallback.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} GPU(s) are visible.")
    return device


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


def atomic_torch(path, value, *, compress=False):
    """Preserve completed artifacts and remove partial writes, including on ENOSPC."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        # A Python file handle exposes filesystem errors that the C++ path writer
        # can otherwise obscure behind an iostream/unexpected-position error.
        with tmp.open("wb") as handle:
            if compress:
                with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=1, mtime=0) as zipped:
                    torch.save(value, zipped)
            else:
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
    if compressed_checkpoint(path):
        with gzip.open(path, "rb") as handle:
            return torch.load(handle, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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


def fit_teachers(records, splits, cfg, device, out):
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
        ckpt = train_teacher(records, train_ids, val_ids, cfg, device, cfg.seed + f,
                             f"Teacher fold {f+1}/{cfg.teacher_folds}")
        predictions = teacher_predict(ckpt, records, heldout.tolist(), cfg, device)
        oof.update({int(i): p for i, p in zip(heldout, predictions)})
        ckpt["heldout_ids"] = [records[i].uid for i in heldout]
        atomic_torch(out / f"teacher_fold_{f+1}.pt", ckpt)
        provenance.append({k: ckpt[k] for k in ("train_ids", "validation_ids", "heldout_ids")})
    final = train_teacher(records, splits["supervised"], splits["dev"], cfg, device,
                          cfg.seed + 100, "Final teacher")
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
    if 0 <= action < cfg.k:
        if state.evaluator_mask & (1 << action):
            return None
        return State(state.candidate_mask, state.evaluator_mask | (1 << action))
    if action == cfg.k:
        missing = np.flatnonzero(~mask[:cfg.zero_shots])
        if not len(missing):
            return None
        return State(state.candidate_mask | (1 << int(missing[0])), state.evaluator_mask)
    source = action - cfg.k - 1
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


def stopping_reason(probabilities, state, cfg):
    if all(value >= cfg.recognition_threshold
           for value in max_recognition_signals(probabilities, state, cfg).values()):
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
    return ([f"add_evaluator_R{j+1}" for j in range(cfg.k)] +
            ["add_zero_shot"] +
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
        return all(probabilities[j] >= threshold for j in
                   (3*n, 3*n+1, n+i, 2*n+i))
    compare = rank == 0 or cfg.later_rank_filter
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
        else:
            if (rank == 0 or cfg.later_rank_filter) and rank < n - 1:
                met &= probabilities[:, i] > max(full_probabilities[j] for j in order[rank+1:])
            if rank > 0 and safe[i]:
                met &= probabilities[:, n+i] >= threshold
        goal[:, rank] = met
    return goal


def visible_state_key(record, state, cfg, allow_hidden_source=False):
    # Candidate IDs and hidden order are audit information, not model features.
    observed = observation(record, state, cfg, allow_hidden_source)
    return (state.candidate_mask, state.evaluator_mask, hashlib.sha256(observed.tobytes()).hexdigest())


def aggregate_decision_rows(scenarios, cfg, transition_table=None):
    """Choose common actions until acquired evidence distinguishes the futures.

    Each visible group contains equiprobable historical orders with the same
    visible observation. Goals never split a group or enter the head's input.
    Search cost ends at goal attainment, a broken prior goal, or action exhaustion.
    """
    groups, cases = {}, []
    # Actions and costs depend on the structure/config, never on historical order.
    unique_states = {state for scenario in scenarios for state in scenario["states"]}
    _, _, cached_valid, cached_costs = decision_structure(
        cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    valid_by_state = {state: cached_valid[state] if state in cached_valid else valid_actions(state, cfg)
                      for state in unique_states}
    actions_by_state = {state: np.flatnonzero(valid) for state, valid in valid_by_state.items()}
    costs = {state: cached_costs[state] if state in cached_costs else state_cost(state, cfg)
             for state in unique_states}
    prefixes = [np.logical_and.accumulate(scenario["goal"], axis=1) for scenario in scenarios]
    goal_ranks = [prefix.sum(axis=1) for prefix in prefixes]
    full = State((1 << cfg.n)-1, (1 << cfg.k)-1)
    for sid, scenario in enumerate(scenarios):
        for j, state in enumerate(scenario["states"]):
            rank = int(goal_ranks[sid][j])
            valid = valid_by_state[state]
            status = ("COMPLETE" if rank == cfg.n else
                      "EXHAUSTED" if state == full else
                      "BUDGET" if not valid.any() else None)
            cases.append({"scenario": sid, "candidate_mask": state.candidate_mask,
                          "evaluator_mask": state.evaluator_mask, "status": status,
                          "evaluators": evaluator_names(state.evaluator_mask, cfg),
                          "goal_rank": None if rank == cfg.n else rank,
                          "training_row": None})
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

    rows = []
    case_index = {(case["scenario"], scenario["index"][State(case["candidate_mask"], case["evaluator_mask"])]): case
                  for case in cases for scenario in [scenarios[case["scenario"]]]}
    # The graph only adds items. Solve all visible-state groups backwards, so
    # every future decision is the same policy used to label that future row.
    continuation = {}
    ordered = sorted(groups.items(), key=lambda item: -(
        bin(item[0][0]).count("1") + bin(item[0][1]).count("1")))
    for key, group in ordered:
        sid, j, _ = group[0]
        scenario = scenarios[sid]
        hidden, state = scenario["hidden"][j], scenario["states"][j]
        valid = valid_by_state[state]
        active = [(s, index, rank) for s, index, rank in group
                  if case_index[s, index]["status"] is None]
        if not active:
            for s, index, _ in group:
                continuation[key, s, index] = (0., 0., 0.)
            continue
        values = np.zeros((cfg.action_count, 3), np.float64)
        values[:, 1:] = np.inf
        individual = {}
        for action in actions_by_state[state]:
            outcomes = []
            for s, index, rank in active:
                current = scenarios[s]
                successor = transitions[s][index][int(action)]
                nxt = current["states"][successor]
                delta = costs[nxt] - costs[state]
                flags = current["goal"][successor]
                if rank and not prefixes[s][successor, rank-1]:
                    outcome = (0., delta, 1.)
                elif flags[rank]:
                    outcome = (1., delta, 1.)
                else:
                    child = continuation[(current["keys"][successor], s, successor)]
                    outcome = (child[0], delta + child[1], 1. + child[2])
                individual[int(action), s, index] = outcome
                outcomes.append(outcome)
            values[action] = outcomes[0] if len(outcomes) == 1 else np.mean(outcomes, axis=0)
        preferred = actions_by_state[state]
        for column, maximize in ((0, True), (1, False), (2, False)):
            scores = values[preferred, column]
            best = scores.max() if maximize else scores.min()
            preferred = preferred[(scores == best) | (np.abs(scores - best) <= 1e-9)]
        chosen = int(preferred[0])  # Concrete tied continuation is deterministic.
        for s, index, _ in active:
            continuation[key, s, index] = individual[chosen, s, index]
        reach, cost, steps = values[chosen]
        if reach <= 0:
            for s, index, _ in active:
                case_index[s, index]["status"] = "UNREACHABLE"
            continue
        target = np.zeros(cfg.action_count, np.float32)
        target[list(preferred)] = 1 / len(preferred)
        ranks = Counter(row[2] for row in active)
        rows.append({"uid": scenario["record"].uid, "state_key": key,
                     "h": hidden, "target": target, "valid": valid,
                     "goal_rank": next(iter(ranks)) if len(ranks) == 1 else -1,
                     "goal_rank_distribution": dict(ranks), "reach": float(reach),
                     "expected_cost": float(cost), "expected_steps": float(steps),
                     "action_reach": values[:, 0], "action_expected_cost": values[:, 1],
                     "action_expected_steps": values[:, 2]})
        for s, index, _ in active:
            case_index[s, index]["status"] = "ACTION"
            case_index[s, index]["training_row"] = len(rows)-1
    return rows, cases


def build_decision_rows(record, scores, safe, maximum, predictor, cfg):
    all_states, transitions, _, costs = decision_structure(
        cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    states = [state for state in all_states if costs[state] <= cfg.budget + 1e-6]
    scenarios = []
    for zero_order in permutations(range(cfg.zero_shots)):
        r, s, y_safe, y_max = permute_zero_shots(record, scores, safe, maximum, zero_order, cfg)
        scenarios.append(decision_scenario(
            r, s, y_safe, y_max, predictor, cfg, states,
            reference_order=permuted_reference_order(scores, zero_order, cfg)))
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
                                                        "fixed_acquisition_orders", "decision_eval_batch_size",
                                                        "decision_cache_mb"}},
                              "predictor": predictor_fingerprint(predictor),
                              "data_sha256": data_digest(records),
                              "roles": {role: [records[i].uid for i in splits[role]]
                                        for role in ("policy", "dev")}}, sort_keys=True).encode())
    for role in ("policy", "dev"):
        for idx in splits[role]:
            for key in ("scores", "safe", "maximum"):
                digest.update(np.asarray(teacher[key][idx], dtype=np.float32).tobytes())
    return digest.hexdigest()


def decision_shard_paths(records, splits, out):
    return {role: [out / "decision_shards" / role /
                   f"{position:06d}_{hashlib.sha256(records[idx].uid.encode()).hexdigest()[:12]}.pt"
                   for position, idx in enumerate(splits[role])]
            for role in ("policy", "dev")}


def shard_coverage(shard, cfg):
    statuses = Counter(case["status"] for case in shard["cases"])
    structures = structural_states(cfg.zero_shots, cfg.k)
    expected = {(sid, state.candidate_mask, state.evaluator_mask)
                for sid in range(math.factorial(cfg.zero_shots)) for state in structures}
    actual = {(case["scenario"], case["candidate_mask"], case["evaluator_mask"])
              for case in shard["cases"]}
    if (sum(statuses.values()) != shard["expected_cases"] or
            len(actual) != len(shard["cases"]) or actual != expected):
        raise ValueError("Decision shard has incomplete state/order coverage.")
    allowed = {"OUT_OF_BUDGET", "COMPLETE", "EXHAUSTED", "BUDGET", "ACTION", "UNREACHABLE"}
    if set(statuses) - allowed:
        raise ValueError("Decision shard contains an unknown status.")
    for case in shard["cases"]:
        index = case["training_row"]
        if (case["status"] == "ACTION") != (index is not None):
            raise ValueError("Decision shard action mapping is incomplete.")
        if index is not None and not 0 <= index < len(shard["rows"]):
            raise ValueError("Decision shard has an invalid training-row reference.")
    for row in shard["rows"]:
        if (len(row["target"]) != cfg.action_count or
                len(row["valid"]) != cfg.action_count or
                not np.isclose(row["target"].sum(), 1) or
                np.any(row["target"][~row["valid"]])):
            raise ValueError("Decision shard has an invalid masked action target.")
    return {"questions": 1, "structural_states": shard["structural_states"],
            "state_order_cases": shard["expected_cases"], "action_rows": len(shard["rows"]),
            "status_counts": dict(statuses),
            "actionable_by_target_rank": dict(Counter(str(case["goal_rank"] + 1)
                for case in shard["cases"] if case["status"] == "ACTION"))}


def combine_coverage(parts):
    statuses, ranks = Counter(), Counter()
    for part in parts:
        statuses.update(part["status_counts"])
        ranks.update(part["actionable_by_target_rank"])
    return {"questions": len(parts),
            "structural_states_per_question": parts[0]["structural_states"] if parts else 0,
            "state_order_cases": sum(p["state_order_cases"] for p in parts),
            "action_rows": sum(p["action_rows"] for p in parts),
            "status_counts": dict(statuses), "actionable_by_target_rank": dict(ranks)}


def pack_decision_rows(rows, cfg):
    """Only the three arrays consumed by the head; audit rows stay in the source shard."""
    return {key: torch.from_numpy(np.stack([row[key] for row in rows])) if rows else
            torch.empty((0, width), dtype=dtype)
            for key, width, dtype in (("h", cfg.hidden_dim, torch.float32),
                                      ("target", cfg.action_count, torch.float32),
                                      ("valid", cfg.action_count, torch.bool))}


def load_training_checkpoint(path):
    """Map only the packed, ordinary tensor companion; the OS pages it on demand."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def ensure_training_companion(path, packed):
    """Reuse only tensors identical to the validated audit rows; repair derived caches."""
    if path.exists():
        try:
            previous = load_training_checkpoint(path)
            matches = (set(previous) == set(packed) and all(
                isinstance(previous[key], torch.Tensor) and
                previous[key].dtype == value.dtype and torch.equal(previous[key], value)
                for key, value in packed.items()))
            del previous  # Close mappings before replacement, including on Windows.
            if matches:
                return True
        except Exception:
            # This derived cache is reconstructed from the validated source shard.
            pass
    atomic_torch(path, packed)
    return False


def available_host_memory():
    """Available RAM, including reclaimable pages; no mandatory new dependency."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
    return None


class PackedDecisionCache:
    """Bounded whole-role tensor reuse, with mmap streaming when a role cannot fit.

    Shuffle accesses each question once per epoch, so an undersized LRU would
    thrash. Admit complete roles without using the training RNG, policy first.
    Never retain audit cases or eagerly copy mapped tensors into RAM.
    """
    def __init__(self, paths, cfg):
        available = available_host_memory()
        requested = int(cfg.decision_cache_mb * 1024**2)
        self.budget = min(requested, available // 2) if available is not None else 0
        self.rows, self.bytes = {}, 0
        retained = []
        for role in ("policy", "dev"):
            # File bytes conservatively include tensor payload plus archive headers.
            required = sum(Path(path).stat().st_size for path in paths[role])
            if required > self.budget - self.bytes or not self.budget:
                continue
            for path in paths[role]:
                packed = load_training_checkpoint(path)
                self.rows[Path(path)] = packed
                self.bytes += sum(t.numel() * t.element_size() for t in packed.values())
            retained.append(role)
        print(f"Stage 3 tensor cache: {self.bytes / 1024**2:.1f} MiB retained, "
              f"{self.budget / 1024**2:.1f} MiB cap (at most half available host RAM); "
              f"roles {', '.join(retained) or 'none'}; other roles stream memory-mapped tensors.",
              flush=True)

    def get(self, path):
        path = Path(path)
        if path in self.rows:
            return self.rows[path]
        return load_training_checkpoint(path)


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


def train_decision_head(records, splits, teacher, predictor, cfg, device, out=None):
    if splits["policy"] != splits["supervised"]:
        raise ValueError("Stage 3 policy questions must match the Stage 1/2 supervised questions.")
    seed_everything(cfg.seed + 300)
    if out is None:
        raise ValueError("Stage 3 requires an output directory for question-sized shards.")
    out = Path(out)
    fingerprint = decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    frozen_before = predictor_fingerprint(predictor)
    paths = decision_shard_paths(records, splits, out)
    training_paths = {role: [path.with_suffix(".training.pt") for path in paths[role]]
                      for role in ("policy", "dev")}
    manifest_path = out / "decision_dataset_manifest.json"
    manifest = {"schema_version": 3, "implementation_revision": PIPELINE_REVISION,
                "fingerprint": fingerprint,
                "roles": {role: [str(path.relative_to(out)) for path in paths[role]]
                          for role in ("policy", "dev")}}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Decision dataset manifest has an incompatible teacher, predictor, split, or action contract.")
    # These exact temporary names are never resumed; the final .pt is the
    # atomic completion marker. Reclaim old failed writes before migration.
    reclaimed = 0
    for role in paths:
        for path in paths[role] + training_paths[role]:
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.exists():
                size = tmp.stat().st_size
                tmp.unlink()
                reclaimed += size
    print(f"Stage 3 storage: {out.resolve()} | "
          f"{shutil.disk_usage(out).free / 1024**3:.2f} GiB free; "
          f"removed {reclaimed / 1024**2:.2f} MiB of incomplete shard writes; "
          "losslessly compressed audit shards, packed training companions.", flush=True)
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    coverage = {}
    for role in ("policy", "dev"):
        parts = []
        started = last_print = time.monotonic()
        built = reused = 0
        print(f"Stage 3 {role} labels: building/validating {len(paths[role])} question shards.", flush=True)
        for position, (idx, path) in enumerate(zip(splits[role], paths[role]), 1):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                shard = load_checkpoint(path)
                if (shard.get("schema_version") != 3 or shard.get("fingerprint") != fingerprint
                        or shard.get("uid") != records[idx].uid):
                    raise ValueError(f"Incompatible decision shard: {path}")
                part = shard_coverage(shard, cfg)
                # Storage-only migration: keep all rows/cases and the exact
                # validated contract while reclaiming legacy pickle overhead.
                if not compressed_checkpoint(path):
                    atomic_torch(path, shard, compress=True)
                reused += 1
            else:
                rows, cases = build_decision_rows(records[idx], teacher["scores"][idx],
                                                  teacher["safe"][idx], teacher["maximum"][idx],
                                                  predictor, cfg)
                shard = {"schema_version": 3, "implementation_revision": PIPELINE_REVISION,
                         "fingerprint": fingerprint, "uid": records[idx].uid,
                         "structural_states": len(structural_states(cfg.zero_shots, cfg.k)),
                         "expected_cases": len(structural_states(cfg.zero_shots, cfg.k)) * math.factorial(cfg.zero_shots),
                         "rows": rows, "cases": cases}
                part = shard_coverage(shard, cfg)
                atomic_torch(path, shard, compress=True)
                built += 1
            parts.append(part)
            # Validate against the source, but do not rewrite an identical companion.
            ensure_training_companion(path.with_suffix(".training.pt"),
                                      pack_decision_rows(shard["rows"], cfg))
            del shard
            last_print = decision_progress(f"Stage 3 {role} labels", position, len(paths[role]),
                                           started, last_print, cfg,
                                           f" | built {built}, reused {reused}")
        coverage[role] = combine_coverage(parts)
        atomic_json(out / "decision_label_coverage.json", coverage)
        print(f"Stage 3 {role}: {coverage[role]['state_order_cases']} state/order cases, "
              f"{coverage[role]['action_rows']} shared action rows; "
              f"statuses {coverage[role]['status_counts']}; "
              f"actionable ranks {coverage[role]['actionable_by_target_rank']}.", flush=True)
        if not coverage[role]["action_rows"]:
            raise ValueError(f"No actionable Stage 3 rows for {role}: {coverage[role]}")
    model = SupervisedActionHead(cfg).to(device)
    cache = PackedDecisionCache(training_paths, cfg)
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
          "Labels are complete; no per-epoch generation.", flush=True)
    for epoch in range(start_epoch, cfg.decision_epochs):
        if stale >= cfg.patience:
            break
        epoch_started = time.monotonic()
        model.train()
        started = last_print = time.monotonic()
        train_total, train_count = 0., 0
        for position, path in enumerate(rng.permutation(training_paths["policy"]), 1):
            rows = cache.get(path)
            value, n = masked_action_loss(model, rows, rng.permutation(len(rows["h"])), cfg, device, opt)
            train_total, train_count = train_total + value, train_count + n
            del rows
            last_print = decision_progress(f"Stage 3 epoch {epoch+1} train", position, len(paths["policy"]),
                                           started, last_print, cfg, f" | rows {train_count}")
        model.eval()
        with torch.no_grad():
            started = last_print = time.monotonic()
            total, count = 0., 0
            for position, path in enumerate(training_paths["dev"], 1):
                rows = cache.get(path)
                value, n = masked_action_loss(model, rows, range(len(rows["h"])), cfg, device,
                                              batch_size=cfg.decision_eval_batch_size)
                total, count = total + value, count + n
                del rows
                last_print = decision_progress(f"Stage 3 epoch {epoch+1} dev", position, len(paths["dev"]),
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
        action = cfg.k if kind == "ZS" else number-1 if kind == "R" else cfg.k+number
        nxt = raw_next_state(state, action, cfg)
        if nxt is None or (kind == "ZS" and (nxt.candidate_mask ^ state.candidate_mask) != 1 << (number-1)):
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

def rollout(record, maximum, predictor, cfg, policy="supervised", head=None, trace=False):
    if policy not in evaluation_methods(cfg):
        raise ValueError(f"Unknown policy: {policy}")
    if policy == "full":
        full_cfg = copy.deepcopy(cfg)
        full_cfg.max_cost = None
        state = State((1 << cfg.n)-1, (1 << cfg.k)-1)
        _, probabilities = predictor.predict(record, state, full_cfg)
        reason, trajectory = "full_budget_reference", []
    else:
        state, reason, trajectory = State(), None, []
        sequence = (fixed_acquisition_sequences(cfg)[policy]["actions"].copy()
                    if policy in cfg.fixed_acquisition_orders else None)
        while reason is None:
            hidden, probabilities = predictor.predict(record, state)
            reason = stopping_reason(probabilities, state, cfg)
            if trace:
                step = {"candidate_mask": int(state.candidate_mask),
                        "evaluator_mask": int(state.evaluator_mask),
                        "evaluators": evaluator_names(state.evaluator_mask, cfg),
                        "candidates": [record.candidate_ids[i] for i in np.flatnonzero(present_mask(state, cfg))],
                        "selected": record.candidate_ids[selected_candidate(probabilities, state, cfg)],
                        "max_signals": max_recognition_signals(probabilities, state, cfg),
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
                item = (f"ZS{slot+1}" if slot is not None and slot < cfg.zero_shots else
                        f"OS{slot-cfg.zero_shots+1}" if slot is not None else f"R{action+1}")
                trajectory.append({**step, "action": int(action),
                                   "action_name": action_names(cfg)[action],
                                   "acquired_item": item,
                                   "acquired_candidate": record.candidate_ids[slot] if slot is not None else None,
                                   "acquired_evaluator": record.evaluator_ids[action] if slot is None else None,
                                   "cost_after": state_cost(nxt, cfg),
                                   "incremental_cost": state_cost(nxt, cfg) - state_cost(state, cfg),
                                   "valid_actions": np.flatnonzero(valid).tolist()})
            state = nxt
    selected = selected_candidate(probabilities, state, cfg)
    has_max = bool(maximum.any())
    exact_max = bool(maximum[selected]) if has_max else None
    return {"uid": record.uid, "group": record.group,
            "selected": record.candidate_ids[selected],
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
            row = rollout(scenario, maximum, predictor, cfg, policy, head, trace)
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
    per_rank = []
    for idx in ids:
        question_prefix, question_lost = [], []
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
                    break  # A later recovery cannot repair an invalid intermediate step.
                best_prefix = prefix
                valid = valid_actions(state, cfg)
                if not valid.any() or prefix == cfg.n:
                    break
                state = advance(state, greedy_action(head, hidden, valid, predictor.device), cfg)
            question_prefix.append(best_prefix)
            question_lost.append(lost)
        reached.append(float(np.mean(question_prefix)))
        broken.append(float(np.mean(question_lost)))
        per_rank.append([float(np.mean(np.asarray(question_prefix) >= rank))
                         for rank in range(1, cfg.n+1)])
    return {"n": len(ids), "orders_per_question": math.factorial(cfg.zero_shots),
            "mean_rank_prefix_reached": float(np.mean(reached)) if reached else None,
            "per_rank_reach": np.mean(per_rank, axis=0).tolist() if per_rank else None,
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
    print(f"Cost unit: {cfg.cost_unit}. Early stopping uses predicted MAX, not historical MAX labels.")
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
            "early_stop_rule": {"reason": "predicted_max", "threshold": cfg.recognition_threshold,
                                "signals": ["global_MAX", "global_SAFE", "selected_MAX", "selected_SAFE"]},
            "call_accounting": "Recorded generation, baseline/cross measurement solves, and their grading calls; excludes retrieval and target-answer grading.",
            "snapshot_diagnostics": stage_snapshot_audit(records, ids, teacher, predictor, cfg),
            "continuation": continuation_diagnostic(records, ids, teacher, predictor, cfg, head)}, rows

class Workflow:
    """Notebook stages; completed checkpoints resume, interrupted stages restart safely."""
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.validate()
        torch.set_num_threads(cfg.cpu_threads)
        seed_everything(cfg.seed)
        self.device = resolve_training_device(cfg.device)
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        contract_cfg = asdict(cfg)
        for key in ("resume", "output_dir", "device", "cpu_threads", "print_every", "fixed_acquisition_orders",
                    "decision_eval_batch_size", "decision_cache_mb"):
            contract_cfg.pop(key)
        sources = [{"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size,
                    "mtime_ns": Path(p).stat().st_mtime_ns} for p in [cfg.train_file] + cfg.test_files]
        contract = {"schema_version": 4, "implementation_revision": PIPELINE_REVISION,
                    "config": contract_cfg, "sources": sources}
        path = self.out / "contract.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            # Round-trip converts tuples to JSON lists consistently.
            if old != json.loads(json.dumps(contract)):
                raise ValueError("Output folder belongs to a different config/data contract. Choose a new output_dir.")
            if not cfg.resume:
                raise ValueError("Output folder already contains this run. Set resume=True or choose a new folder.")
        else:
            if any(self.out.iterdir()):
                raise ValueError("Use an empty output folder; existing unrecognized files will not be overwritten.")
            atomic_json(path, contract)
        self.records, self.splits = None, None

    def prepare(self):
        section(f"DATA AUDIT | device={self.device} | {self.cfg.n} candidates, {self.cfg.k} evaluators")
        cache_path = self.out / "compact_records.pt"
        if self.cfg.resume and cache_path.exists():
            cache = load_checkpoint(cache_path)
            self.records = [Record(**r) for r in cache["records"]]
            self.audit = cache["audit"]
            print("Using compact cached records (source path/size/mtime contract verified).")
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
        return self

    def train_teacher(self):
        if self.records is None:
            self.prepare()
        section(f"STAGE 1 | Masked joint correctness model and out-of-fold ranking | device={self.device}")
        path = self.out / "teacher_completed.pt"
        if self.cfg.resume and path.exists():
            self.teacher = load_checkpoint(path)
            print("Loaded completed teacher stage.")
        else:
            self.teacher = fit_teachers(self.records, self.splits, self.cfg, self.device, self.out)
            atomic_torch(path, self.teacher)
        self.heuristics = select_heuristics(self.records, self.splits["dev"], self.cfg)
        atomic_json(self.out / "heuristics_selected_on_dev.json", self.heuristics)
        ids = self.splits["supervised"]
        empty = int((self.teacher["safe"][ids].sum(1) == 0).sum())
        print(f"OOF labels: {len(ids)} supervised questions; empty SAFE/MAX: {empty}/{len(ids)}.")
        teacher_report(self.records, self.splits["dev"], self.cfg, self.teacher["scores"],
                       self.heuristics, "TEACHER DEVELOPMENT REPORT (used for model selection)")
        return self

    def train_snapshot(self):
        if not hasattr(self, "teacher"):
            self.train_teacher()
        section(f"STAGE 2 | Snapshot ResNet ({self.cfg.input_dim} inputs, {3*self.cfg.n+2} prediction outputs) | device={self.device}")
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
        return self

    def train_decision_head(self):
        if not hasattr(self, "predictor"):
            self.train_snapshot()
        section(f"STAGE 3 | Supervised acquisition labels and frozen-encoder head | device={self.device}")
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
                                             self.predictor, self.cfg, self.device, self.out)
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
        return self

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
        return results


def load_inference_bundle(path, device="cpu"):
    bundle = load_checkpoint(path)
    if bundle.get("schema_version") != 4 or bundle.get("implementation_revision") != PIPELINE_REVISION:
        raise ValueError(f"Expected a version 4 inference bundle with revision {PIPELINE_REVISION} training contract; old bundles are incompatible.")
    cfg = Config(**bundle["config"])
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
