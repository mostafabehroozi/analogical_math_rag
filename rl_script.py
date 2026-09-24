import importlib.util
import subprocess
import sys

# Kaggle normally supplies NumPy/PyTorch. Only the small streaming reader may need installation.
if importlib.util.find_spec("ijson") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ijson"])
print("Streaming JSON reader ready. Enable Kaggle Internet only if installation is needed.")

"""Kaggle-compatible teacher -> snapshot -> frozen-encoder DQN training.

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
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



# Configuration and compact data
@dataclass
class Config:
    train_file: str = "/kaggle/working/downloaded_files/dir_5/numina_hard_run_log.json"
    test_files: list = field(default_factory=lambda: [
        "/kaggle/working/downloaded_files/dir_1/aime25_run_log.json",
        "/kaggle/working/downloaded_files/dir_2/aime26_run_log.json",
        "/kaggle/working/downloaded_files/dir_3/gsm8k_run_log.json",
        "/kaggle/working/downloaded_files/dir_4/math500_run_log.json",
    ])
    output_dir: str = "/kaggle/working/adaptive_analogical_early_run"
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
    auxiliary_weight: float = .25
    calibrate: bool = True          # Development-only temperature selection.
    stop_mode: str = "safe"         # safe / max / reliability
    stop_threshold: float = .95
    member_threshold: float = .90
    require_selected_member: bool = True
    cost_unit: str = "solver_calls" # solver_calls / total_calls (incl. evaluator graders)
    max_cost: Optional[float] = None # None = cost of the complete pool.
    cost_weight: float = .20
    early_quality_weight: float = .10
    rl_steps: int = 20000
    rl_hidden: int = 64
    rl_lr: float = 1e-4
    rl_batch_size: int = 64
    replay_size: int = 50000
    warmup: int = 1000
    target_update: int = 500
    eval_every: int = 1000
    exploration_fraction: float = .30
    epsilon_end: float = .05
    gamma: float = 1.0
    double_dqn: bool = False
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
        assert self.stop_mode in {"safe", "max", "reliability"}
        assert self.cost_unit in {"solver_calls", "total_calls"}
        assert 0 < self.stop_threshold <= 1 and 0 < self.member_threshold <= 1
        assert len(self.split_fractions) == 4 and min(self.split_fractions) > 0
        assert abs(sum(self.split_fractions) - 1) < 1e-8
        assert acquisition_cost(1, 1, self) <= self.budget <= self.full_cost
        assert self.teacher_epochs > 0 and self.snapshot_epochs > 0 and self.patience > 0
        assert self.rl_steps > 0 and self.rl_batch_size > 0 and self.eval_every > 0
        assert self.replay_size >= self.rl_batch_size and self.target_update > 0
        assert self.warmup >= 0 and self.snapshots_per_query >= 2
        assert self.dev_snapshots_per_query >= 2 and self.cost_weight >= 0
        assert self.early_quality_weight >= 0
        assert self.early_quality_weight == 0 or self.gamma == 1.0
        assert 0 <= self.snapshot_reachable_fraction <= 1
        assert 0 <= self.snapshot_missing_fraction <= 1
        assert 0 <= self.snapshot_permutation_fraction <= 1
        assert 0 < self.gamma <= 1 and 0 < self.exploration_fraction <= 1
        assert 0 <= self.epsilon_end <= 1 and self.print_every > 0


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



CFG = Config(
    # Data paths: edit these for your Kaggle dataset locations.
    train_file="/kaggle/working/downloaded_files/dir_5/numina_hard_run_log.json",
    test_files=[
        "/kaggle/working/downloaded_files/dir_1/aime25_run_log.json",
        "/kaggle/working/downloaded_files/dir_2/aime26_run_log.json",
        "/kaggle/working/downloaded_files/dir_3/gsm8k_run_log.json",
        "/kaggle/working/downloaded_files/dir_4/math500_run_log.json",
    ],
    output_dir="/kaggle/working/adaptive_analogical_early_run",
    resume=True,
    device="auto",                       # GPU if available; CPU also supported.
    seed=75,
    k=5, zero_shots=3, repeats=5,

    # Disjoint question roles: supervised / RL policy / development / final internal audit.
    split_fractions=(0.60, 0.20, 0.10, 0.10),
    teacher_folds=5,
    hidden_dim=128, residual_blocks=2, dropout=0.10,
    learning_rate=0.0003, weight_decay=1e-4, batch_size=128,
    teacher_epochs=100, snapshot_epochs=100, patience=15,
    snapshots_per_query=24, dev_snapshots_per_query=16,
    snapshot_reachable_fraction=0.50,    # Other half cycles through all coherent structures.
    snapshot_missing_fraction=0.25,      # Hide recorded measurements or mark attempts without results.
    snapshot_permutation_fraction=0.10,  # Permute slots together with evidence and labels.
    auxiliary_weight=0.25,              # Set 0 for correctness-only ablation.
    calibrate=True,                     # Development-only temperature fitting.

    # Fixed application policy and cost objective.
    stop_mode="safe",                   # "safe", "max", or "reliability"
    stop_threshold=0.95,
    member_threshold=0.90,
    require_selected_member=True,
    cost_unit="solver_calls",           # Or "total_calls" including evaluator grading.
    max_cost=None,                     # None=233 solver calls / 458 total calls at full pool.
    cost_weight=0.20,                   # Incremental API-call cost weight.
    early_quality_weight=0.10,          # Correct selected answer earlier within the budget.

    # Only the two acquisition heads are trained with RL.
    rl_steps=20000, rl_hidden=64, rl_lr=1e-4,
    rl_batch_size=64, replay_size=50000, warmup=1000,
    target_update=500, eval_every=1000,
    exploration_fraction=0.30, epsilon_end=0.05,
    gamma=1.0, double_dqn=False,
    bootstrap_samples=1000, print_every=10,
)

# Optional cheap end-to-end pilot. This does NOT provide publishable performance.
QUICK_PILOT = False
if QUICK_PILOT:
    CFG.output_dir += "_pilot"
    CFG.teacher_folds = 2
    CFG.teacher_epochs = 3
    CFG.snapshot_epochs = 3
    CFG.snapshots_per_query = 4
    CFG.dev_snapshots_per_query = 4
    CFG.rl_steps = 1000
    CFG.warmup = 100
    CFG.eval_every = 500
    CFG.bootstrap_samples = 100

print("Configuration ready. Finish running definition cells, then start Stage 0.")

# Teacher network, metrics, and cross-fitted labels
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


def teacher_features(r, cfg):
    x = np.zeros((cfg.n, 2 + 4 * cfg.k), np.float32)
    x[:cfg.zero_shots, 0] = 1
    x[:, 2:2 + cfg.k] = r.similarity
    x[:, 2 + cfg.k:2 + 2 * cfg.k] = r.baseline
    x[:, 2 + 2 * cfg.k:2 + 3 * cfg.k] = r.ccs
    for j in range(cfg.k):
        i = cfg.zero_shots + j
        x[i, 1] = r.similarity[j]
        x[i, 2 + 3 * cfg.k + j] = 1
    return x


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
    # Avoid singleton training batches with the original BatchNorm architecture.
    size = min(batch_size, len(x))
    if size < 2 and shuffle:
        raise ValueError("BatchNorm training requires at least two samples.")
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=max(1, size), shuffle=shuffle,
                      drop_last=shuffle and len(x) % size == 1)


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
    x = np.concatenate([teacher_features(records[i], cfg) for i in train_ids])
    y = np.concatenate([records[i].labels for i in train_ids])[:, None]
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1
    vx = np.concatenate([teacher_features(records[i], cfg) for i in val_ids])
    model = ResNet(x.shape[1], 1, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loader = make_loader((x - mean) / std, y, cfg.batch_size)
    best, best_state, stale, best_epoch = -1.0, None, 0, 0
    for epoch in range(cfg.teacher_epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(bx.to(device)), by.to(device))
            loss.backward()
            opt.step()
        pred = predict_array(model, (vx - mean) / std, device).reshape(-1, cfg.n)
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
    return {"weights": best_state, "mean": mean, "std": std, "best_epoch": best_epoch,
            "train_ids": [records[i].uid for i in train_ids],
            "validation_ids": [records[i].uid for i in val_ids]}


def teacher_predict(checkpoint, records, ids, cfg, device):
    model = ResNet(2 + 4 * cfg.k, 1, cfg).to(device)
    model.load_state_dict(checkpoint["weights"])
    x = np.concatenate([teacher_features(records[i], cfg) for i in ids])
    return predict_array(model, (x - checkpoint["mean"]) / checkpoint["std"], device).reshape(-1, cfg.n)


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



# Partial snapshots and supervised prediction
@dataclass(frozen=True)
class State:
    candidates: int = 1             # Bit mask; ZS1 initially exists.
    evaluators: int = 1             # Active retrieval prefix.


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


def validate_snapshot_state(state, cfg):
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
    if state.candidates >> cfg.zero_shots & ~state.retrieved:
        raise ValueError("one-shot candidate without retrieved source")
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
    active = (1 << state.evaluators) - 1
    return complete_snapshot_state(state.candidates, active, active, cfg)


@lru_cache(maxsize=4)
def structural_catalog(zero_shots, k):
    """Every nonempty, logically valid candidate/retrieval/evaluator structure."""
    rows = []
    for zero_mask in range(1 << zero_shots):
        for code in range(5 ** k):
            candidates, retrieved, evaluators, rest = zero_mask, 0, 0, code
            for i in range(k):
                choice, rest = rest % 5, rest // 5
                if choice:
                    retrieved |= 1 << i
                if choice in (2, 4):
                    evaluators |= 1 << i
                if choice in (3, 4):
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
    context = [spent / cfg.full_cost, (cfg.budget - spent) / cfg.full_cost,
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
    validate_snapshot_state(result, cfg)
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
    validate_snapshot_state(new_state, cfg)
    return new_record, new_state, safe[corder], maximum[corder]


def build_snapshots(records, ids, teacher, cfg, per_query, seed, epoch=0, augmented=False):
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    reachable = reachable_states(cfg)
    catalog = structural_catalog(cfg.zero_shots, cfg.k) if augmented else ()
    n_reachable = max(1, round(per_query * cfg.snapshot_reachable_fraction)) if augmented else per_query
    for rank, idx in enumerate(ids):
        states = [State()]
        for j in range(n_reachable - 1):
            position = (epoch * len(ids) + rank) * (n_reachable - 1) + j
            states.append(reachable[(position * 53) % len(reachable)])
        # The coprime stride cycles through the entire catalog across questions
        # and epochs, rather than sampling only the common structures.
        for j in range(per_query - n_reachable):
            position = ((epoch * len(ids) + rank) * (per_query - n_reachable) + j)
            candidates, retrieved, evaluators = catalog[(position * 7919) % len(catalog)]
            states.append(complete_snapshot_state(candidates, retrieved, evaluators, cfg))
        for state in states:
            r, safe, maximum = records[idx], teacher["safe"][idx], teacher["maximum"][idx]
            if augmented and rng.rand() < cfg.snapshot_missing_fraction:
                state = augment_measurements(state, cfg, rng)
            if augmented and rng.rand() < cfg.snapshot_permutation_fraction:
                r, state, safe, maximum = permute_snapshot(r, state, safe, maximum, cfg, rng)
            xs.append(observation(r, state, cfg))
            ys.append(snapshot_target(r, safe, maximum, state, cfg))
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
    off_per_query = cfg.snapshots_per_query - max(1, round(cfg.snapshots_per_query * cfg.snapshot_reachable_fraction))
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
        # No global prediction cache: hidden outcome changes cannot be hidden by stale keys.
        x = torch.as_tensor(observation(record, state, cfg_override or self.cfg)[None], device=self.device)
        with torch.no_grad():
            h = self.model.encode(x)
            logits = self.model.output_layer(h)[0].cpu().numpy()
        p = 1 / (1 + np.exp(-np.clip(logits / self.temperatures, -40, 40)))
        # Public prediction lights for nonexistent candidates are OFF. Their logits
        # remain excluded, rather than treated as negatives, in supervised training.
        absent = ~present_mask(state, self.cfg)
        for block in range(3):
            p[block*self.cfg.n:(block+1)*self.cfg.n][absent] = 0
        return h[0].cpu().numpy(), p


def selected_candidate(probabilities, state, cfg):
    return int(np.where(present_mask(state, cfg), probabilities[:cfg.n], -np.inf).argmax())


def stopping_reason(probabilities, state, cfg):
    i, n = selected_candidate(probabilities, state, cfg), cfg.n
    confident = False
    if cfg.stop_mode == "reliability":
        confident = probabilities[i] >= cfg.stop_threshold
    elif cfg.stop_mode == "safe":
        confident = probabilities[3*n] >= cfg.stop_threshold
        if cfg.require_selected_member:
            confident = confident and probabilities[n+i] >= cfg.member_threshold
    elif cfg.stop_mode == "max":
        confident = probabilities[3*n] >= cfg.stop_threshold and probabilities[3*n+1] >= cfg.stop_threshold
        if cfg.require_selected_member:
            confident = confident and probabilities[2*n+i] >= cfg.member_threshold
    if confident:
        return cfg.stop_mode
    if not valid_actions(state, cfg).any():
        full = state.candidates == (1 << cfg.n)-1 and state.evaluators == cfg.k
        return "pool_exhaustion" if full else "budget"
    return None



# Cached acquisition environment and head-only DQN
class CachedEnvironment:
    def __init__(self, record, predictor, cfg, early_quality_weight=None):
        self.record, self.predictor, self.cfg = record, predictor, cfg
        self.early_quality_weight = (cfg.early_quality_weight if early_quality_weight is None
                                     else early_quality_weight)
        self.initial_cost = state_cost(State(), cfg)
        self.early_area = 0.0
        self.state = State()
        self.h, self.probabilities = predictor.predict(record, self.state)
        self.reason = stopping_reason(self.probabilities, self.state, cfg)

    def step(self, action):
        if self.reason is not None:
            raise ValueError("Cannot acquire after application termination.")
        previous = state_cost(self.state, self.cfg)
        previous_correct = int(self.record.labels[selected_candidate(
            self.probabilities, self.state, self.cfg)])
        self.state = advance(self.state, action, self.cfg)
        self.h, self.probabilities = self.predictor.predict(self.record, self.state)
        self.reason = stopping_reason(self.probabilities, self.state, self.cfg)
        cost = state_cost(self.state, self.cfg) - previous
        span = self.cfg.budget - self.initial_cost
        self.early_area += previous_correct * cost
        reward = -self.cfg.cost_weight * cost / self.cfg.full_cost
        if self.early_quality_weight and span > 0:
            reward += self.early_quality_weight * previous_correct * cost / span
        if self.reason is not None:
            final_correct = int(self.record.labels[selected_candidate(
                self.probabilities, self.state, self.cfg)])
            reward += float(final_correct)
            if self.early_quality_weight and span > 0:
                reward += self.early_quality_weight * final_correct * (
                    self.cfg.budget - state_cost(self.state, self.cfg)) / span
        return reward, self.reason is not None

    def early_quality(self):
        final_correct = int(self.record.labels[selected_candidate(
            self.probabilities, self.state, self.cfg)])
        span = self.cfg.budget - self.initial_cost
        if span <= 0:
            return float(final_correct)
        return float((self.early_area + final_correct * (
            self.cfg.budget - state_cost(self.state, self.cfg))) / span)


class ActionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.rl_hidden), nn.ReLU(),
                                 nn.Linear(cfg.rl_hidden, cfg.action_count))

    def forward(self, h):
        return self.net(h)


class Replay:
    def __init__(self, cfg):
        size = cfg.replay_size
        self.h = np.zeros((size, cfg.hidden_dim), np.float32)
        self.nh = np.zeros_like(self.h)
        self.actions = np.zeros(size, np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.done = np.zeros(size, bool)
        self.next_valid = np.zeros((size, cfg.action_count), bool)
        self.position, self.count = 0, 0

    def add(self, h, action, reward, next_h, done, next_valid):
        j = self.position
        self.h[j], self.nh[j] = h, next_h
        self.actions[j], self.rewards[j], self.done[j] = action, reward, done
        self.next_valid[j] = next_valid
        self.position = (j + 1) % len(self.actions)
        self.count = min(self.count + 1, len(self.actions))


def bellman_target(reward, done, next_valid, target_values, gamma, online_values=None):
    values = target_values.masked_fill(~next_valid, -torch.inf)
    if torch.any((~done) & (~next_valid.any(dim=1))):
        raise ValueError("Nonterminal transition has no valid next action.")
    if online_values is None:
        future = values.max(dim=1).values
    else:
        best = online_values.masked_fill(~next_valid, -torch.inf).argmax(dim=1)
        future = target_values.gather(1, best[:, None]).squeeze(1)
    future = torch.where(done, torch.zeros_like(future), future)
    return reward + gamma * future


def greedy_action(head, h, valid, device):
    with torch.no_grad():
        q = head(torch.as_tensor(h[None], device=device))[0].cpu().numpy()
    return int(np.where(valid, q, -np.inf).argmax())


def snapshot_audit_row(env, trace=False):
    cfg, record = env.cfg, env.record
    selected = selected_candidate(env.probabilities, env.state, cfg)
    present = present_mask(env.state, cfg)
    row = {"cost": state_cost(env.state, cfg),
           "offline_selected_correct": int(record.labels[selected]),
           "offline_acquired_oracle": bool(record.labels[present].any())}
    if trace:
        row.update({"candidate_mask": int(env.state.candidates),
                    "evaluators": int(env.state.evaluators),
                    "selected": record.candidate_ids[selected], "action": None,
                    "valid_actions": [],
                    "predictions": env.probabilities.tolist(), "reason": env.reason})
    return row


def budget_curve(snapshots, cfg):
    initial = state_cost(State(), cfg)
    points = {}
    for fraction in (0., .25, .50, .75, 1.):
        limit = initial + fraction * (cfg.budget - initial)
        current = snapshots[0]
        for state in snapshots[1:]:
            if state["cost"] > limit + 1e-6:
                break
            current = state
        points[str(fraction)] = {
            "selected_correct": current["offline_selected_correct"],
            "acquired_oracle": int(current["offline_acquired_oracle"])}
    return points


def rollout(record, predictor, cfg, policy="rl", head=None, rng=None, trace=False):
    env = CachedEnvironment(record, predictor, cfg)
    snapshots = [snapshot_audit_row(env, trace=trace)]
    while env.reason is None:
        valid = valid_actions(env.state, cfg)
        available = np.flatnonzero(valid)
        if policy == "rl":
            action = greedy_action(head, env.h, valid, predictor.device)
        elif policy == "random":
            action = int(rng.choice(available))
        elif policy == "evaluator_first":
            action = int(available[0])
        elif policy == "candidate_first":
            priority = list(range(1, cfg.action_count)) + [0]
            action = next(a for a in priority if valid[a])
        elif policy == "cheapest":
            action = min(available, key=lambda a: state_cost(advance(env.state, int(a), cfg), cfg))
        else:
            raise ValueError(policy)
        if trace:
            snapshots[-1]["action"] = int(action)
            snapshots[-1]["valid_actions"] = available.tolist()
        env.step(action)
        snapshots.append(snapshot_audit_row(env, trace=trace))
    chosen = selected_candidate(env.probabilities, env.state, cfg)
    cost = state_cost(env.state, cfg)
    return {"uid": record.uid, "group": record.group, "selected": record.candidate_ids[chosen],
            "correct": int(record.labels[chosen]), "cost": cost,
            "utility": float(record.labels[chosen] - cfg.cost_weight*cost/cfg.full_cost),
            "early_quality": env.early_quality(), "budget_curve": budget_curve(snapshots, cfg),
            "reason": env.reason, "confidence": float(env.probabilities[chosen]),
            "safe_probability": float(env.probabilities[3*cfg.n]),
            "max_probability": float(env.probabilities[3*cfg.n+1]),
            "candidates": int(present_mask(env.state, cfg).sum()), "evaluators": env.state.evaluators,
            "acquired_oracle": bool(record.labels[present_mask(env.state, cfg)].any()),
            "full_oracle": bool(record.labels.any()), "zero_shot": chosen < cfg.zero_shots,
            "trajectory": snapshots if trace else []}


def policy_summary(rows):
    if not rows:
        return None
    correct = np.asarray([r["correct"] for r in rows], float)
    costs = np.asarray([r["cost"] for r in rows], float)
    confident = np.asarray([r["reason"] in {"safe", "max", "reliability"} for r in rows])
    return {"n": len(rows), "top1": float(correct.mean()), "mean_cost": float(costs.mean()),
            "median_cost": float(np.median(costs)), "p90_cost": float(np.percentile(costs, 90)),
            "p95_cost": float(np.percentile(costs, 95)),
            "utility": float(np.mean([r["utility"] for r in rows])),
            "early_quality": (float(np.mean([r["early_quality"] for r in rows]))
                              if all("early_quality" in r for r in rows) else None),
            "budget_curve": ({key: {
                field: float(np.mean([r["budget_curve"][key][field] for r in rows]))
                for field in ("selected_correct", "acquired_oracle")}
                for key in rows[0]["budget_curve"]}
                if all("budget_curve" in r for r in rows) else None),
            "confident_n": int(confident.sum()), "confident_errors": int((1-correct[confident]).sum()),
            "confident_error": float((1-correct[confident]).mean()) if confident.any() else None,
            "forced_fraction": float((~confident).mean()),
            "acquired_oracle": float(np.mean([r["acquired_oracle"] for r in rows])),
            "full_oracle": float(np.mean([r["full_oracle"] for r in rows])),
            "zero_shot_fraction": float(np.mean([r["zero_shot"] for r in rows])),
            "stop_reasons": dict(Counter(r["reason"] for r in rows))}


def evaluate_policy(records, ids, predictor, cfg, head=None, policy="rl", trace=False):
    rows = []
    for idx in ids:
        # Same target gets the same random baseline independent of file/report ordering.
        salt = int(records[idx].group[:8], 16)
        rng = np.random.RandomState((cfg.seed + salt) % (2**32-1))
        rows.append(rollout(records[idx], predictor, cfg, policy, head, rng, trace))
    return rows


def policy_rank(metrics, cfg):
    return (metrics["top1"],
            cfg.early_quality_weight * metrics["early_quality"]
            - cfg.cost_weight * metrics["mean_cost"] / cfg.full_cost)


def fit_dqn(records, splits, predictor, cfg, out, variant="refined"):
    if variant not in {"baseline", "refined"}:
        raise ValueError(variant)
    reward_weight = 0.0 if variant == "baseline" else cfg.early_quality_weight
    seed_everything(cfg.seed + 300)
    rng = np.random.RandomState(cfg.seed + 301)
    device = predictor.device
    online = ActionHead(cfg).to(device)
    target = copy.deepcopy(online).eval().requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=cfg.rl_lr)
    replay = Replay(cfg)
    before = cpu_state(predictor.model)
    active = [i for i in splits["policy"]
              if CachedEnvironment(records[i], predictor, cfg, reward_weight).reason is None]
    if not active:
        raise ValueError("Every policy-training initial state already stops. Adjust stopping settings "
                         "on development data and use a new output directory; there is no RL decision to train.")
    best, best_weights, updates, history = (-math.inf, -math.inf), cpu_state(online), 0, []
    env = CachedEnvironment(records[int(rng.choice(active))], predictor, cfg, reward_weight)
    for step in range(1, cfg.rl_steps + 1):
        valid = valid_actions(env.state, cfg)
        eps = 1 - (1-cfg.epsilon_end)*min(1., step/(cfg.rl_steps*cfg.exploration_fraction))
        action = int(rng.choice(np.flatnonzero(valid))) if rng.rand() < eps else greedy_action(online, env.h, valid, device)
        h = env.h.copy()
        reward, done = env.step(action)
        replay.add(h, action, reward, env.h, done, valid_actions(env.state, cfg))
        if replay.count >= max(cfg.warmup, cfg.rl_batch_size):
            b = rng.choice(replay.count, cfg.rl_batch_size, replace=False)
            tensor = lambda x: torch.as_tensor(x, device=device)
            bh, nh = tensor(replay.h[b]), tensor(replay.nh[b])
            with torch.no_grad():
                y = bellman_target(tensor(replay.rewards[b]), tensor(replay.done[b]),
                                   tensor(replay.next_valid[b]), target(nh), cfg.gamma,
                                   online(nh) if cfg.double_dqn else None)
            q = online(bh).gather(1, tensor(replay.actions[b])[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10)
            opt.step()
            updates += 1
            if updates % cfg.target_update == 0:
                target.load_state_dict(online.state_dict())
        if done:
            env = CachedEnvironment(records[int(rng.choice(active))], predictor, cfg, reward_weight)
        if step % cfg.eval_every == 0 or step == cfg.rl_steps:
            metrics = policy_summary(evaluate_policy(records, splits["dev"], predictor, cfg, online))
            history.append({"step": step, "updates": updates, **metrics})
            print(f"{variant} DQN step {step:6d} | dev accuracy {metrics['top1']:.1%} | "
                  f"cost {metrics['mean_cost']:.1f}/{cfg.full_cost:.0f} | "
                  f"early quality {metrics['early_quality']:.3f}")
            rank = policy_rank(metrics, cfg)
            if rank > best:
                best, best_weights, best_step = rank, cpu_state(online), step
                atomic_torch(out / f"rl_{variant}_in_progress.pt", {"weights": best_weights,
                             "step": step, "completed": False, "variant": variant})
    if updates == 0:
        raise ValueError("No RL updates occurred: lower warmup/batch size or increase rl_steps.")
    for key, value in predictor.model.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[key]):
            raise AssertionError("Frozen snapshot predictor changed during RL.")
    return {"weights": best_weights, "best_step": best_step, "updates": updates,
            "variant": variant, "reward_weight": reward_weight, "dev_rank": best,
            "history": history, "frozen_verified": True, "completed": True}



# Reports, artifacts, and stage orchestration
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
    difference = np.array([x["correct"]-y["correct"] for x, y in zip(a, b)], float)
    rng = np.random.RandomState(seed)
    sampled = [float(rng.choice(difference, len(difference), replace=True).mean())
               for _ in range(max(1, repetitions))]
    return {"delta": float(difference.mean()),
            "ci95": np.percentile(sampled, [2.5, 97.5]).tolist(),
            "benefits": int((difference > 0).sum()), "harms": int((difference < 0).sum())}


def full_snapshot_rows(records, ids, predictor, cfg):
    full_cfg = copy.deepcopy(cfg)
    full_cfg.max_cost = None
    state = State((1 << cfg.n)-1, cfg.k)
    rows = []
    scores = {}
    for idx in ids:
        _, p = predictor.predict(records[idx], state, full_cfg)
        scores[idx] = p[:cfg.n]
        i = selected_candidate(p, state, cfg)
        r = records[idx]
        rows.append({"uid": r.uid, "group": r.group, "selected": r.candidate_ids[i],
                     "correct": int(r.labels[i]), "cost": cfg.full_cost,
                     "utility": float(r.labels[i] - cfg.cost_weight), "reason": "full_budget_reference",
                     "confidence": float(p[i]), "candidates": cfg.n, "evaluators": cfg.k,
                     "acquired_oracle": bool(r.labels.any()), "full_oracle": bool(r.labels.any()),
                     "zero_shot": i < cfg.zero_shots, "trajectory": []})
    return rows, scores


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


def stage_snapshot_audit(records, ids, teacher, predictor, cfg):
    if not ids:
        return None
    x, y = build_snapshots(records, ids, teacher, cfg, cfg.dev_snapshots_per_query, cfg.seed + 401)
    logits = predict_array(predictor.model, x, predictor.device) / predictor.temperatures
    metrics = snapshot_summary(logits, y, cfg)
    p = 1/(1+np.exp(-np.clip(logits, -40, 40)))
    for name, col in (("safe", 3*cfg.n), ("max", 3*cfg.n+1)):
        positive = p[:, col] >= cfg.stop_threshold
        true = y[:, col].astype(bool)
        metrics[name + "_precision"] = float(true[positive].mean()) if positive.any() else None
        metrics[name + "_recall"] = float(positive[true].mean()) if true.any() else None
    metrics["predicted_max_without_safe"] = float(((p[:, 3*cfg.n+1] >= cfg.stop_threshold)
                                                    & (p[:, 3*cfg.n] < cfg.stop_threshold)).mean())
    sx, sy = build_snapshots(records, ids, teacher, cfg, cfg.dev_snapshots_per_query,
                             cfg.seed + 402, augmented=True)
    stress_logits = predict_array(predictor.model, sx, predictor.device) / predictor.temperatures
    metrics["augmented_stress"] = snapshot_summary(stress_logits, sy, cfg)
    return metrics


def policy_report(records, ids, teacher, predictor, cfg, head, title,
                  baseline_head=None, refined_head=None):
    section(title)
    if not ids:
        print("No eligible questions: report is unavailable, not zero-filled.")
        return None, {}
    rows = {p: evaluate_policy(records, ids, predictor, cfg, policy=p, trace=True)
            for p in ("evaluator_first", "candidate_first", "random", "cheapest")}
    rows["rl_baseline"] = evaluate_policy(records, ids, predictor, cfg,
                                            baseline_head or head, trace=True)
    rows["rl_refined"] = evaluate_policy(records, ids, predictor, cfg,
                                           refined_head or head, trace=True)
    selected_variant = "rl_refined" if head is refined_head else "rl_baseline"
    rows["rl"] = rows[selected_variant]
    rows["full_snapshot"], scores = full_snapshot_rows(records, ids, predictor, cfg)
    summary = {p: policy_summary(v) for p, v in rows.items()}
    print(f"Questions: {len(ids)} | Stop mode: {cfg.stop_mode} | Budget: {cfg.budget:.0f} {cfg.cost_unit}")
    print(f"{'Acquisition policy':<23} | {'Top-1':>7} | {'Mean cost':>10} | {'P90 cost':>8} | {'Early':>7} | {'Conf.err':>8}")
    print("-"*91)
    for name in ("full_snapshot", "evaluator_first", "candidate_first", "random",
                 "cheapest", "rl_baseline", "rl_refined", "rl"):
        m = summary[name]
        err = "--" if m["confident_error"] is None else f"{m['confident_error']:.1%}"
        early = "--" if m["early_quality"] is None else f"{m['early_quality']:.3f}"
        print(f"{name:<23} | {m['top1']:6.1%} | {m['mean_cost']:10.1f} | {m['p90_cost']:8.1f} | "
              f"{early:>7} | {err:>8}")
    rl = summary["rl"]
    interval = wilson_interval(rl["confident_errors"], rl["confident_n"])
    if interval:
        print(f"RL confident-stop errors: {rl['confident_errors']}/{rl['confident_n']} "
              f"(Wilson 95% interval {interval[0]:.1%}..{interval[1]:.1%}); "
              f"forced stops: {rl['forced_fraction']:.1%}.")
    else:
        print("RL had no confidence-triggered stops; all stops were forced by budget/pool exhaustion.")
    paired = {name: paired_interval(rows["rl"], rows[name], cfg.bootstrap_samples, cfg.seed)
              for name in ("random", "full_snapshot")}
    d = paired["full_snapshot"]
    print(f"RL vs full: accuracy delta {100*d['delta']:+.2f}pp "
          f"[95% paired bootstrap {100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}pp].")
    if cfg.budget < cfg.full_cost:
        print("Full-snapshot reference exceeds the acquisition cap; compare accuracy and cost together.")
    return {"policies": summary, "paired": paired,
            "full_snapshot_ranking": ranking_metrics(records, ids, scores, cfg, list(range(cfg.n))),
            "snapshot_diagnostics": stage_snapshot_audit(records, ids, teacher, predictor, cfg)}, rows


def plot_budget_curves(summary, cfg, best_simple, path):
    """Offline last-completed-snapshot curves; full evidence is an endpoint only."""
    initial = state_cost(State(), cfg)
    fractions = (0., .25, .50, .75, 1.)
    xs = [initial + f * (cfg.budget - initial) for f in fractions]
    colors = {best_simple: "#64748b", "rl_baseline": "#2563eb", "rl_refined": "#ea580c"}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="370" '
             'viewBox="0 0 1000 370">', '<rect width="1000" height="370" fill="white"/>']
    xmax = max(cfg.full_cost, cfg.budget)
    for panel, (field, title, full_field) in enumerate((
            ("selected_correct", "Selected answer correct", "top1"),
            ("acquired_oracle", "Correct candidate available", "acquired_oracle"))):
        left = 68 + 495 * panel
        right, top, bottom = left + 390, 55, 285
        xcoord = lambda value: left + (value - initial) * (right - left) / max(1, xmax - initial)
        ycoord = lambda value: bottom - value * (bottom - top)
        parts.append(f'<text x="{left}" y="27" font-size="17" font-family="sans-serif">{title}</text>')
        for value in (0., .25, .5, .75, 1.):
            y = ycoord(value)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                         'stroke="#e2e8f0"/>')
            parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" '
                         f'font-size="11" font-family="sans-serif">{value:.0%}</text>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#475569"/>')
        for value in sorted({initial, cfg.budget, cfg.full_cost}):
            x = xcoord(value)
            parts.append(f'<text x="{x:.1f}" y="{bottom+18}" text-anchor="middle" '
                         f'font-size="11" font-family="sans-serif">{value:g}</text>')
        for name, color in colors.items():
            curve = summary[name]["budget_curve"]
            if curve is None:
                continue
            pairs = [(xcoord(x), ycoord(curve[str(f)][field])) for x, f in zip(xs, fractions)]
            coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in pairs)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
                         for x, y in pairs)
        full_y = ycoord(summary["full_snapshot"][full_field])
        full_x = xcoord(cfg.full_cost)
        parts.append(f'<text x="{full_x:.1f}" y="{full_y+5:.1f}" text-anchor="middle" '
                     'font-size="18" fill="black">×</text>')
        parts.append(f'<text x="{(left+right)/2:.1f}" y="{bottom+42}" text-anchor="middle" '
                     f'font-size="12" font-family="sans-serif">Completed cost ({cfg.cost_unit})</text>')
    for j, (name, color) in enumerate(colors.items()):
        x = 85 + 220 * j
        parts.append(f'<line x1="{x}" y1="350" x2="{x+18}" y2="350" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+24}" y="354" font-size="12" font-family="sans-serif">{name}</text>')
    parts.append('<text x="750" y="354" font-size="12" font-family="sans-serif">× full evidence endpoint</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


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
        contract = {"schema_version": 2, "config": contract_cfg, "sources": sources}
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
        section("STAGE 1 | Full-evidence ResNet teacher and out-of-fold SAFE/MAX labels")
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
        print(f"Application stopping: {self.cfg.stop_mode}, threshold={self.cfg.stop_threshold}, "
              f"selected-member gate={self.cfg.require_selected_member}.")
        return self

    def train_rl(self):
        if not hasattr(self, "predictor"):
            self.train_snapshot()
        section("STAGE 3 | Baseline and early-quality masked "
                + ("Double DQN" if self.cfg.double_dqn else "DQN"))
        checkpoints, heads = {}, {}
        for variant in ("baseline", "refined"):
            path = self.out / f"rl_{variant}_completed.pt"
            if self.cfg.resume and path.exists():
                checkpoints[variant] = load_checkpoint(path)
                print(f"Loaded completed {variant} RL stage.")
            else:
                checkpoints[variant] = fit_dqn(self.records, self.splits,
                                               self.predictor, self.cfg, self.out, variant)
                atomic_torch(path, checkpoints[variant])
            heads[variant] = ActionHead(self.cfg).to(self.device)
            heads[variant].load_state_dict(checkpoints[variant]["weights"])
            heads[variant].eval()
        self.baseline_head, self.refined_head = heads["baseline"], heads["refined"]
        dev = {name: policy_summary(evaluate_policy(self.records, self.splits["dev"],
                                                 self.predictor, self.cfg, policy=name))
               for name in ("evaluator_first", "candidate_first", "random", "cheapest")}
        dev["rl_baseline"] = policy_summary(evaluate_policy(
            self.records, self.splits["dev"], self.predictor, self.cfg, self.baseline_head))
        dev["rl_refined"] = policy_summary(evaluate_policy(
            self.records, self.splits["dev"], self.predictor, self.cfg, self.refined_head))
        reference = max(("evaluator_first", "candidate_first", "random", "cheapest",
                         "rl_baseline"), key=lambda name: policy_rank(dev[name], self.cfg))
        accepted = policy_rank(dev["rl_refined"], self.cfg) > policy_rank(dev[reference], self.cfg)
        selected = "refined" if accepted else "baseline"
        self.head = heads[selected]
        checkpoint = checkpoints[selected]
        self.selection = {"accepted": accepted, "exported": f"rl_{selected}",
                          "strongest_reference": reference,
                          "best_simple": max(("evaluator_first", "candidate_first", "random",
                                              "cheapest"), key=lambda name: policy_rank(dev[name], self.cfg)),
                          "development": dev}
        atomic_json(self.out / "rl_selection.json", self.selection)
        atomic_torch(self.out / "rl_completed.pt", checkpoint)
        print(f"Refined accepted: {accepted}; development reference: {reference}; "
              f"best simple policy: {self.selection['best_simple']}; "
              f"exported: rl_{selected} at step {checkpoint['best_step']}.")
        # Self-contained inference bundle; only this head is learned by RL.
        atomic_torch(self.out / "inference_bundle.pt", {
            "schema_version": 2, "config": asdict(self.cfg),
            "snapshot": load_checkpoint(self.out / "snapshot_completed.pt"),
            "rl": checkpoint, "rl_variants": checkpoints, "selection": self.selection,
            "action_names": ["add_evaluator", "add_zero_shot"] +
            [f"one_shot_source_{j+1}" for j in range(self.cfg.k)],
            "cost_note": "Fixed-m cached bundles; no target-grading cost at deployment."})
        return self

    def report(self):
        if not hasattr(self, "head"):
            self.train_rl()
        results = {"rl_selection": self.selection}
        external = [Path(p).stem for p in self.cfg.test_files]
        names = ["audit"] + external
        trajectory_path = self.out / "final_trajectories.jsonl"
        tmp = trajectory_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for name in names:
                ids = self.splits[name]
                ranking = teacher_report(self.records, ids, self.cfg, self.teacher["scores"], self.heuristics,
                                         f"TEACHER | {name} ({len(ids)} eligible questions)")
                policy, rows = policy_report(self.records, ids, self.teacher, self.predictor, self.cfg,
                                             self.head, f"ADAPTIVE EVALUATION | {name}",
                                             self.baseline_head, self.refined_head)
                results[name] = {"ranking": ranking, "adaptive": policy}
                for method, method_rows in rows.items():
                    for row in method_rows:
                        handle.write(json.dumps({"benchmark": name, "method": method, **row}, allow_nan=False) + "\n")
        os.replace(tmp, trajectory_path)
        available = [name for name in external if results[name]["adaptive"] is not None]
        if available:
            section(f"TEACHER EXTERNAL MEAN | {len(available)}/{len(external)} nonempty eligible benchmarks")
            print(f"{'Ranking method':<43} | {'Mean AP':>8} | {'Top-1':>8} | {'AP delta':>9}")
            ranking_macro = {}
            base = np.mean([results[b]["ranking"]["Baseline (Retrieval Order)"]["ap"] for b in available])
            for method in results[available[0]]["ranking"]:
                ap = float(np.mean([results[b]["ranking"][method]["ap"] for b in available]))
                acc = float(np.mean([results[b]["ranking"][method]["top1"] for b in available]))
                ranking_macro[method] = {"ap": ap, "top1": acc}
                print(f"{method:<43} | {ap:8.4f} | {acc:7.1%} | {100*(ap-base):+8.2f}pp")
            results["teacher_external_macro"] = {"included": available, "methods": ranking_macro}
            section(f"EQUAL-WEIGHT EXTERNAL MEAN | {len(available)}/{len(external)} nonempty eligible benchmarks")
            print(f"{'Acquisition policy':<23} | {'Top-1':>8} | {'Mean cost':>10} | {'Saving':>8}")
            macro = {}
            for method in ("full_snapshot", "evaluator_first", "candidate_first", "random",
                           "cheapest", "rl_baseline", "rl_refined", "rl"):
                vals = [results[b]["adaptive"]["policies"][method] for b in available]
                acc = float(np.mean([v["top1"] for v in vals]))
                cost = float(np.mean([v["mean_cost"] for v in vals]))
                macro[method] = {"top1": acc, "mean_cost": cost}
                print(f"{method:<23} | {acc:7.1%} | {cost:10.1f} | {1-cost/self.cfg.full_cost:7.1%}")
            results["external_macro"] = {"included": available,
                                         "excluded_empty": sorted(set(external)-set(available)), "methods": macro}
            weights = np.array([len(self.splits[b]) for b in available], float)
            results["external_micro"] = {method: {
                "top1": float(np.average([results[b]["adaptive"]["policies"][method]["top1"]
                                           for b in available], weights=weights)),
                "mean_cost": float(np.average([results[b]["adaptive"]["policies"][method]["mean_cost"]
                                                for b in available], weights=weights))}
                for method in macro}
        atomic_json(self.out / "results.json", results)
        if results["audit"]["adaptive"] is not None:
            plot_budget_curves(results["audit"]["adaptive"]["policies"], self.cfg,
                               self.selection["best_simple"], self.out / "accuracy_vs_cost_audit.svg")
        print(f"\nSaved compact reports, trajectories, stage checkpoints, and inference bundle to {self.out}")
        print("Accuracy is over eligible complete records; cached policy evaluation is not live API validation.")
        print("Prior benchmark use is not erased by the new split. SAFE/MAX thresholds are not certainty guarantees.")
        self.results = results
        return results


def load_inference_bundle(path, device="cpu"):
    bundle = load_checkpoint(path)
    cfg = Config(**bundle["config"])
    predictor = FrozenPredictor(bundle["snapshot"], cfg, torch.device(device))
    head = ActionHead(cfg).to(device)
    head.load_state_dict(bundle["rl"]["weights"])
    head.eval()
    return cfg, predictor, head


def run_pipeline(cfg=None):
    work = Workflow(cfg or Config())
    work.prepare().train_teacher().train_snapshot().train_rl()
    work.report()
    return work



WORK = Workflow(CFG)
WORK.prepare();

WORK.train_teacher();

WORK.train_snapshot();

WORK.train_rl();

RESULTS = WORK.report()

print("Artifacts:", WORK.out)
for path in sorted(WORK.out.iterdir()):
    print(f"  {path.name:<36} {path.stat().st_size / 1024**2:8.2f} MB")

# Reload for later cached inference (optional):
# config, frozen_predictor, action_head = load_inference_bundle(
#     WORK.out / "inference_bundle.pt", device=str(WORK.device)
# )
