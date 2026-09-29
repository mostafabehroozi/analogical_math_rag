"""Kaggle-compatible ranking -> snapshot -> supervised acquisition training.

The companion notebook embeds this module: uploading the notebook alone is enough.
No provider calls are made. Acquisition is simulated from recorded Layer-1 data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
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

PIPELINE_REVISION = 2  # Common decisions across indistinguishable zero-shot futures.


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
    output_dir: str = "/kaggle/working/adaptive_analogical_supervised_run"
    resume: bool = True             # Reuse completed stages with identical contracts.
    seed: int = 75
    device: str = "auto"
    cpu_threads: int = 2
    k: int = 5
    zero_shots: int = 3
    repeats: int = 5                # Must match the stored CCS denominator.
    split_fractions: tuple = (.60, .20, .10, .10)  # supervised / policy / dev / audit
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
    decision_augmented_fraction: float = .10
    bootstrap_samples: int = 1000
    print_every: int = 10

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
        assert len(self.split_fractions) == 4 and min(self.split_fractions) > 0
        assert abs(sum(self.split_fractions) - 1) < 1e-8
        assert 1 <= self.budget <= self.full_cost
        assert self.teacher_epochs > 0 and self.snapshot_epochs > 0 and self.patience > 0
        assert self.decision_epochs > 0 and self.decision_batch_size >= 2 and self.decision_lr > 0
        assert self.snapshots_per_query >= 2 and self.dev_snapshots_per_query >= 2
        assert 0 <= self.snapshot_reachable_fraction <= 1
        assert 0 <= self.snapshot_missing_fraction <= 1
        assert 0 <= self.snapshot_permutation_fraction <= 1
        assert all(0 <= p <= 1 for p in (self.candidate_mask_fraction,
                                         self.evaluator_mask_fraction,
                                         self.hidden_source_fraction))
        assert 0 <= self.decision_augmented_fraction <= .10
        assert self.print_every > 0


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
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    # Only load checkpoints created by this workflow, never untrusted .pt files.
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
    train = [i for i, r in enumerate(records) if r.benchmark == Path(cfg.train_file).stem]
    if len(train) < max(20, cfg.teacher_folds * 3):
        raise ValueError("Too few eligible training questions for four disjoint roles and teacher folds.")
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(train)
    cuts = np.rint(np.cumsum(cfg.split_fractions)[:-1] * len(train)).astype(int)
    parts = np.split(np.asarray(train), cuts)
    if min(map(len, parts)) < 2:
        raise ValueError("Every internal role must contain at least two question groups.")
    result = {name: p.tolist() for name, p in zip(("supervised", "policy", "dev", "audit"), parts)}
    for path in cfg.test_files:
        name = Path(path).stem
        result[name] = [i for i, r in enumerate(records) if r.benchmark == name]
    return result


def data_digest(records):
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps([r.uid, r.group, r.candidate_ids, r.evaluator_ids]).encode())
        for a in (r.similarity, r.baseline, r.ccs, r.labels):
            h.update(a.tobytes())
    return h.hexdigest()


def acquisition_cost(n, k, cfg):
    probes = cfg.repeats * k * (n + 1)
    return float(n + probes * (2 if cfg.cost_unit == "total_calls" else 1))


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
    full = State((1 << cfg.n)-1, cfg.k)
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
    full = State((1 << cfg.n)-1, cfg.k)
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
    candidates: int = 1             # Bit mask; ZS1 initially exists.
    evaluators: int = 0             # Add R1 through the normal acquisition action.


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
    active = (1 << state.evaluators) - 1
    if allow_hidden_source:
        cells = sum(active << (i * cfg.k) for i in range(cfg.n)
                    if state.candidates & (1 << i))
        result = SnapshotState(state.candidates, active, active, active,
                               active, active, cells, cells)
        validate_snapshot_state(result, cfg, allow_hidden_source=True)
        return result
    return complete_snapshot_state(state.candidates, active, active, cfg)


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


def reachable_states(cfg):
    """All states supported by the seven current application actions."""
    seen, pending = {State()}, [State()]
    for state in pending:
        for action in np.flatnonzero(valid_actions(state, cfg)):
            nxt = advance(state, int(action), cfg)
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return tuple(pending)


def present_mask(state, cfg):
    return np.asarray([(state.candidates >> i) & 1 for i in range(cfg.n)], bool)


def state_cost(state, cfg):
    if isinstance(state, SnapshotState):
        probes = cfg.repeats * (bin(state.baseline_attempted).count("1") + bin(state.ccs_attempted).count("1"))
        return float(bin(state.candidates).count("1") + probes * (2 if cfg.cost_unit == "total_calls" else 1))
    return acquisition_cost(int(present_mask(state, cfg).sum()), state.evaluators, cfg)


def raw_next_state(state, action, cfg):
    mask = present_mask(state, cfg)
    if action == 0:
        if state.evaluators >= cfg.k:
            return None
        return State(state.candidates, state.evaluators + 1)
    if action == 1:
        missing = np.flatnonzero(~mask[:cfg.zero_shots])
        if not len(missing):
            return None
        return State(state.candidates | (1 << int(missing[0])), state.evaluators)
    source = action - 2
    if not 0 <= source < state.evaluators:
        return None
    slot = cfg.zero_shots + source
    if mask[slot]:
        return None
    return State(state.candidates | (1 << slot), state.evaluators)


def valid_actions(state, cfg):
    result = np.zeros(cfg.action_count, bool)
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
        states = [State((1 << cfg.n)-1, cfg.k)]
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

    def predict_many(self, record, states, cfg_override=None, allow_hidden_source=False):
        """Batch frozen inference; only visible measurements enter the network."""
        cfg = cfg_override or self.cfg
        xs = np.stack([observation(record, s, cfg, allow_hidden_source)
                       for s in states])
        hs, ps = [], []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(xs), 1024):
                h = self.model.encode(torch.as_tensor(xs[start:start+1024], device=self.device))
                logits = self.model.output_layer(h).cpu().numpy()
                hs.append(h.cpu().numpy())
                ps.append(1 / (1 + np.exp(-np.clip(logits / self.temperatures, -40, 40))))
        hidden, probabilities = np.concatenate(hs), np.concatenate(ps)
        for row, state in enumerate(states):
            absent = ~present_mask(state, self.cfg)
            for block in range(3):
                probabilities[row, block*self.cfg.n:(block+1)*self.cfg.n][absent] = 0
        return hidden, probabilities


def selected_candidate(probabilities, state, cfg):
    return int(np.where(present_mask(state, cfg), probabilities[:cfg.n], -np.inf).argmax())


def stopping_reason(probabilities, state, cfg):
    i, n = selected_candidate(probabilities, state, cfg), cfg.n
    threshold = cfg.recognition_threshold
    if all(probabilities[j] >= threshold for j in (3*n, 3*n+1, n+i, 2*n+i)):
        return "predicted_max"
    if not valid_actions(state, cfg).any():
        full = state.candidates == (1 << cfg.n)-1 and state.evaluators == cfg.k
        return "pool_exhaustion" if full else "budget"
    return None

def greedy_action(head, hidden, valid, device):
    if not valid.any():
        raise ValueError("No valid acquisition action.")
    with torch.no_grad():
        logits = head(torch.as_tensor(hidden[None], device=device))[0].cpu().numpy()
    return int(np.where(valid, logits, -np.inf).argmax())


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
    if not state.candidates & (1 << i):
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
    hidden, probabilities = predictor.predict_many(
        record, states, allow_hidden_source=allow_hidden_source)
    full_cfg = copy.deepcopy(cfg)
    full_cfg.max_cost = None
    _, full_probability = predictor.predict(record, State((1 << cfg.n)-1, cfg.k), full_cfg)
    order = reference_order if reference_order is not None else ranked_indices(scores, retrieval_order(cfg))
    goal = np.asarray([[goal_condition(rank, order, safe, maximum, p,
                                       full_probability, state, cfg)
                        for rank in range(cfg.n)]
                       for state, p in zip(states, probabilities)], bool)
    keys = [visible_state_key(record, state, cfg, allow_hidden_source) for state in states]
    return {"record": record, "states": states, "hidden": hidden, "goal": goal,
            "keys": keys, "index": {state: j for j, state in enumerate(states)}}


def visible_state_key(record, state, cfg, allow_hidden_source=False):
    # Candidate IDs and hidden order are audit information, not model features.
    observed = observation(record, state, cfg, allow_hidden_source)
    return (state.candidates, state.evaluators, hashlib.sha256(observed.tobytes()).hexdigest())


def aggregate_decision_rows(scenarios, cfg, roots=None, artificial=()):
    """Choose common actions until acquired evidence distinguishes the futures.

    Each recursive group contains equiprobable historical orders with the same
    visible observation. Goals never split a group or enter the head's input.
    Search cost ends at goal attainment, a broken prior goal, or action exhaustion.
    """
    groups = {}
    artificial = set(artificial)
    for sid, scenario in enumerate(scenarios):
        for j, state in enumerate(scenario["states"]):
            if roots is not None and state not in roots:
                continue
            rank = next((r for r, met in enumerate(scenario["goal"][j]) if not met), cfg.n)
            if rank < cfg.n:
                groups.setdefault(scenario["keys"][j], []).append((sid, j, rank))

    transitions = {}
    for sid, scenario in enumerate(scenarios):
        for j, state in enumerate(scenario["states"]):
            transitions[sid, j] = {
                int(action): scenario["index"][advance(state, int(action), cfg)]
                for action in np.flatnonzero(valid_actions(state, cfg))}

    @lru_cache(maxsize=None)
    def solve(members):
        first_sid, first_j, _ = members[0]
        valid = np.zeros(cfg.action_count, bool)
        valid[list(transitions[first_sid, first_j])] = True
        values = np.zeros((cfg.action_count, 3), np.float64)
        values[:, 1:] = np.inf
        if not valid.any():
            return (0., 0., 0.), (), values
        for action in np.flatnonzero(valid):
            branches = {}
            reach, cost, steps = 0., 0., 0.
            for sid, j, rank in members:
                scenario = scenarios[sid]
                successor = transitions[sid, j][int(action)]
                state, nxt = scenario["states"][j], scenario["states"][successor]
                cost += state_cost(nxt, cfg) - state_cost(state, cfg)
                steps += 1
                flags = scenario["goal"][successor]
                if not flags[:rank].all():
                    continue  # This historical branch violates an earlier goal.
                if flags[rank]:
                    reach += 1
                else:
                    branches.setdefault(scenario["keys"][successor], []).append((sid, successor, rank))
            for branch in branches.values():
                continuation, _, _ = solve(tuple(branch))
                reach += len(branch) * continuation[0]
                cost += len(branch) * continuation[1]
                steps += len(branch) * continuation[2]
            values[action] = np.asarray([reach, cost, steps]) / len(members)
        preferred = np.flatnonzero(valid)
        for column, maximize in ((0, True), (1, False), (2, False)):
            scores = values[preferred, column]
            best = scores.max() if maximize else scores.min()
            preferred = preferred[np.isclose(scores, best, rtol=0, atol=1e-9)]
        # Keep genuine ties as soft targets; a concrete continuation uses the first index.
        return tuple(values[preferred[0]]), tuple(int(a) for a in preferred), values

    rows, skipped = [], 0
    for key, group in groups.items():
        sid, j, _ = group[0]
        scenario = scenarios[sid]
        hidden, state = scenario["hidden"][j], scenario["states"][j]
        valid = valid_actions(state, cfg)
        if not all(np.allclose(hidden, scenarios[s]["hidden"][index], atol=1e-6)
                   for s, index, _ in group):
            raise ValueError("One visible state has inconsistent frozen features.")
        (reach, cost, steps), preferred, values = solve(tuple(group))
        if reach <= 0:
            skipped += 1
            continue
        target = np.zeros(cfg.action_count, np.float32)
        target[list(preferred)] = 1 / len(preferred)
        ranks = Counter(row[2] for row in group)
        rows.append({"uid": scenario["record"].uid, "state_key": key,
                     "h": hidden, "target": target, "valid": valid,
                     "weight": .1 if state in artificial else 1.,
                     "goal_rank": next(iter(ranks)) if len(ranks) == 1 else -1,
                     "goal_rank_distribution": dict(ranks), "reach": float(reach),
                     "expected_cost": float(cost), "expected_steps": float(steps),
                     "action_reach": values[:, 0], "action_expected_cost": values[:, 1]})
    solve.cache_clear()
    return rows, skipped


def build_decision_rows(record, scores, safe, maximum, predictor, cfg, augmented=False):
    base_states = reachable_states(cfg)
    chosen = []
    if augmented and cfg.decision_augmented_fraction:
        rng = np.random.RandomState(cfg.seed + int(record.group[:8], 16))
        possible = []
        for state in base_states:
            sources = [j for j in range(cfg.k)
                       if state.candidates & (1 << (cfg.zero_shots+j)) and j < state.evaluators]
            if sources:
                artificial = State(state.candidates, int(rng.choice(sources)))
                if valid_actions(artificial, cfg).any():
                    possible.append(artificial)
        rng.shuffle(possible)
        chosen = list(dict.fromkeys(possible))[:round(len(base_states) * cfg.decision_augmented_fraction)]
    states = list(dict.fromkeys(list(base_states) + [s for root in chosen for s in states_from(root, cfg)]))
    scenarios = []
    for zero_order in permutations(range(cfg.zero_shots)):
        r, s, y_safe, y_max = permute_zero_shots(record, scores, safe, maximum, zero_order, cfg)
        scenarios.append(decision_scenario(
            r, s, y_safe, y_max, predictor, cfg, states, allow_hidden_source=bool(chosen),
            reference_order=permuted_reference_order(scores, zero_order, cfg)))
    rows, skipped = aggregate_decision_rows(scenarios, cfg,
                                           roots=set(base_states) | set(chosen), artificial=chosen)
    ordinary = [row for row in rows if row["weight"] == 1.]
    extra = [row for row in rows if row["weight"] < 1.]
    if extra:
        rng.shuffle(extra)
    return ordinary + extra[:int(len(ordinary) * cfg.decision_augmented_fraction)], skipped

class SupervisedActionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.decision_hidden), nn.ReLU(),
                                 nn.Linear(cfg.decision_hidden, cfg.action_count))

    def forward(self, hidden):
        return self.net(hidden)

def train_decision_head(records, splits, teacher, predictor, cfg, device, out=None):
    seed_everything(cfg.seed + 300)
    datasets, coverage = {}, {}
    for role, augment in (("policy", True), ("dev", False)):
        rows, unreachable = [], 0
        for idx in splits[role]:
            record_rows, missing = build_decision_rows(
                records[idx], teacher["scores"][idx], teacher["safe"][idx],
                teacher["maximum"][idx], predictor, cfg, augmented=augment)
            rows.extend(record_rows)
            unreachable += missing
        coverage[role] = {"questions": len(splits[role]), "labeled": len(rows),
                          "unreachable": unreachable,
                          "by_goal_rank": dict(Counter(row["goal_rank"] for row in rows))}
        if out is not None:
            atomic_json(out / "decision_label_coverage.json", coverage)
        if not rows:
            raise ValueError(f"No reachable decision labels for {role}: "
                             f"{unreachable} unreachable states across {len(splits[role])} questions "
                             f"at recognition threshold {cfg.recognition_threshold:.2f}.")
        datasets[role] = rows
        if out is not None:
            atomic_torch(out / f"decision_dataset_{role}.pt",
                         {"schema_version": 2, "implementation_revision": PIPELINE_REVISION,
                          "rows": rows, "coverage": coverage[role]})
    def arrays(rows):
        return (np.stack([row["h"] for row in rows]).astype(np.float32),
                np.stack([row["target"] for row in rows]).astype(np.float32),
                np.stack([row["valid"] for row in rows]),
                np.asarray([row["weight"] for row in rows], np.float32))
    train_h, train_y, train_v, train_w = arrays(datasets["policy"])
    dev_h, dev_y, dev_v, dev_w = arrays(datasets["dev"])
    del datasets
    model = SupervisedActionHead(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.decision_lr, weight_decay=cfg.weight_decay)
    best, weights, stale, history = math.inf, None, 0, []
    rng = np.random.RandomState(cfg.seed + 301)
    for epoch in range(cfg.decision_epochs):
        model.train()
        for batch in np.array_split(rng.permutation(len(train_h)),
                                    max(1, math.ceil(len(train_h) / cfg.decision_batch_size))):
            h = torch.as_tensor(train_h[batch], device=device)
            y = torch.as_tensor(train_y[batch], device=device)
            valid = torch.as_tensor(train_v[batch], device=device)
            weight = torch.as_tensor(train_w[batch], device=device)
            logits = model(h).masked_fill(~valid, -1e9)
            loss = (-(y * nn.functional.log_softmax(logits, dim=1)).sum(1) * weight).sum() / weight.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.as_tensor(dev_h, device=device)).masked_fill(
                ~torch.as_tensor(dev_v, device=device), -1e9)
            y = torch.as_tensor(dev_y, device=device)
            weight = torch.as_tensor(dev_w, device=device)
            dev_loss = float((-(y * nn.functional.log_softmax(logits, dim=1)).sum(1) * weight).sum()
                             / weight.sum())
        history.append({"epoch": epoch+1, "dev_loss": dev_loss})
        if dev_loss < best - 1e-7:
            best, weights, stale, best_epoch = dev_loss, cpu_state(model), 0, epoch+1
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    return {"weights": weights, "best_epoch": best_epoch, "dev_loss": best,
            "coverage": coverage, "history": history,
            "input_dim": cfg.hidden_dim, "action_count": cfg.action_count}

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


def fixed_order_actions(cfg):
    order = []
    for source in range(cfg.k):
        order.extend((0, 2 + source))
        if source < cfg.zero_shots - 1:
            order.append(1)
    order.extend([1] * max(0, cfg.zero_shots - cfg.k - 1))
    return order

def rollout(record, maximum, predictor, cfg, policy="supervised", head=None, trace=False):
    if policy == "full":
        full_cfg = copy.deepcopy(cfg)
        full_cfg.max_cost = None
        state = State((1 << cfg.n)-1, cfg.k)
        _, probabilities = predictor.predict(record, state, full_cfg)
        reason, trajectory = "full_budget_reference", []
    else:
        state, reason, trajectory = State(), None, []
        sequence = fixed_order_actions(cfg) if policy == "fixed" else None
        while reason is None:
            hidden, probabilities = predictor.predict(record, state)
            reason = stopping_reason(probabilities, state, cfg)
            if reason is not None:
                if trace:
                    trajectory.append({"candidate_mask": int(state.candidates),
                                       "evaluators": int(state.evaluators),
                                       "cost": state_cost(state, cfg), "action": None,
                                       "reason": reason})
                break
            valid = valid_actions(state, cfg)
            if policy == "fixed":
                action = sequence[0] if sequence else None
                if action is None or not valid[action]:
                    reason = "budget"
                    if trace:
                        trajectory.append({"candidate_mask": int(state.candidates),
                                           "evaluators": int(state.evaluators),
                                           "cost": state_cost(state, cfg), "action": None,
                                           "reason": reason, "blocked_fixed_action": action})
                    break
                sequence.pop(0)
            elif policy == "supervised":
                action = greedy_action(head, hidden, valid, predictor.device)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            if trace:
                trajectory.append({"candidate_mask": int(state.candidates),
                                   "evaluators": int(state.evaluators),
                                   "cost": state_cost(state, cfg), "action": int(action),
                                   "valid_actions": np.flatnonzero(valid).tolist()})
            state = advance(state, action, cfg)
    selected = selected_candidate(probabilities, state, cfg)
    has_max = bool(maximum.any())
    exact_max = bool(maximum[selected]) if has_max else None
    return {"uid": record.uid, "group": record.group,
            "selected": record.candidate_ids[selected],
            "correct": int(record.labels[selected]), "has_max": has_max,
            "exact_max": exact_max, "cost": state_cost(state, cfg),
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

def policy_summary(rows):
    if not rows:
        return None
    by_question = {}
    for row in rows:
        by_question.setdefault(row["uid"], []).append(row)
    means = [{"correct": np.mean([r["correct"] for r in group]),
              "cost": np.mean([r["cost"] for r in group]),
              "saved": np.mean([r["saved_fraction"] for r in group]),
              "has_max": group[0]["has_max"],
              "exact_max": np.mean([r["exact_max"] for r in group]) if group[0]["has_max"] else None,
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
            "stop_reasons": dict(Counter(r["reason"] for r in rows))}

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
            _, full = predictor.predict(record, State((1 << cfg.n)-1, cfg.k), full_cfg)
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
            for name in ("full", "fixed", "supervised")}
    summary = {name: policy_summary(data) for name, data in rows.items()}
    print(f"{'Method':<16} | {'Top-1 correct':>13} | {'Exact MAX':>9} | {'Mean calls':>10} | {'Saved':>7}")
    for name, item in summary.items():
        exact = "--" if item["exact_max_recovery"] is None else f"{item['exact_max_recovery']:.1%}"
        print(f"{name:<16} | {item['top1_correct']:>12.1%} | {exact:>9} | "
              f"{item['mean_cost']:>10.1f} | {item['mean_saved_fraction']:>6.1%}")
    return {"policies": summary,
            "snapshot_diagnostics": stage_snapshot_audit(records, ids, teacher, predictor, cfg),
            "continuation": continuation_diagnostic(records, ids, teacher, predictor, cfg, head)}, rows

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
        contract_cfg = asdict(cfg)
        for key in ("resume", "output_dir", "device", "cpu_threads", "print_every"):
            contract_cfg.pop(key)
        sources = [{"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size,
                    "mtime_ns": Path(p).stat().st_mtime_ns} for p in [cfg.train_file] + cfg.test_files]
        contract = {"schema_version": 3, "implementation_revision": PIPELINE_REVISION,
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
        print("Strict complete-pool analysis; exclusions are reported, never silently counted as wrong.")
        print("CCS uses recorded rates/configured attempts; per-trial API validity is not inferred from rates.")
        print("Exact normalized-text duplicates removed; near-duplicate/corpus contamination requires a separate audit.")
        atomic_json(self.out / "data_audit.json", self.audit)
        atomic_json(self.out / "split_manifest.json", {"data_sha256": data_digest(self.records),
                    "roles": {k: [self.records[i].uid for i in ids] for k,ids in self.splits.items()},
                    "groups": {r.uid: r.group for r in self.records}})
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
        return self

    def train_decision_head(self):
        if not hasattr(self, "predictor"):
            self.train_snapshot()
        section("STAGE 3 | Supervised acquisition labels and frozen-encoder head")
        path = self.out / "decision_head_completed.pt"
        if self.cfg.resume and path.exists():
            checkpoint = load_checkpoint(path)
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
            "schema_version": 3, "implementation_revision": PIPELINE_REVISION,
            "config": asdict(self.cfg),
            "snapshot": load_checkpoint(self.out / "snapshot_completed.pt"),
            "decision_head": checkpoint,
            "action_names": ["add_next_retrieved", "add_zero_shot"] +
            [f"one_shot_source_{j+1}" for j in range(self.cfg.k)],
            "cost_note": "Fixed-m cached bundles; no target-grading cost at deployment."})
        print(f"Decision head: epoch {checkpoint['best_epoch']}, "
              f"development loss {checkpoint['dev_loss']:.4f}.")
        return self

    def report(self):
        if not hasattr(self, "head"):
            self.train_decision_head()
        results = {"decision_training": {"best_epoch": self.decision_checkpoint["best_epoch"],
                                         "dev_loss": self.decision_checkpoint["dev_loss"],
                                         "coverage": self.decision_checkpoint["coverage"]}}
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
                        for baseline in ("full", "fixed")}
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
                    for method in ("full", "fixed", "supervised")}}
        atomic_json(self.out / "results.json", results)
        print(f"Saved reports, trajectories, checkpoints, and inference bundle to {self.out}")
        print("These are cached historical outcomes, not live API or Kaggle evidence.")
        self.results = results
        return results


def load_inference_bundle(path, device="cpu"):
    bundle = load_checkpoint(path)
    if bundle.get("schema_version") != 3:
        raise ValueError("Expected a version 3 supervised-acquisition bundle; old DQN bundles are incompatible.")
    cfg = Config(**bundle["config"])
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
