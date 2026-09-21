"""Complete Kaggle-cell code for zero/one/few-shot context selection.

Inputs are paired normal Layer-1 and Layer-1-grouping run logs.  Each question
becomes one variable-length list of context options:

* exactly one zero-shot option: the first successfully answered ``zs_*`` item;
* one option for every successfully answered one-shot retrieval;
* one option for every successfully answered few-shot grouping candidate.

The single teacher label is the correct option with the smallest group size.
Ties use the largest sum of retrieval cosine similarities.  If no option is
correct, the zero-shot option is the teacher fallback.  Correctness is never an
input feature; it is used only to construct the teacher and report accuracy.
"""

import copy
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. CENTRAL CONFIGURATION
# ==========================================
TRAIN_LAYER1_FILE_PATH = (
    r"/kaggle/working/downloaded_files/dir_5/numina_hard_run_log.json"
)
TRAIN_GROUPING_FILE_PATH = (
    r"/kaggle/working/downloaded_files/dir_6/numina_hard_grouping_run_log.json"
)

# Pair every benchmark's normal Layer-1 JSON with its grouping JSON.
TEST_FILE_PAIRS = [
    {
        "name": "aime25",
        "layer1": r"/kaggle/working/downloaded_files/dir_1/aime25_run_log.json",
        "grouping": r"/kaggle/working/downloaded_files/dir_7/aime25_grouping_run_log.json",
    },
    {
        "name": "aime26",
        "layer1": r"/kaggle/working/downloaded_files/dir_2/aime26_run_log.json",
        "grouping": r"/kaggle/working/downloaded_files/dir_8/aime26_grouping_run_log.json",
    },
    {
        "name": "gsm8k",
        "layer1": r"/kaggle/working/downloaded_files/dir_3/gsm8k_run_log.json",
        "grouping": r"/kaggle/working/downloaded_files/dir_9/gsm8k_grouping_run_log.json",
    },
    {
        "name": "math500",
        "layer1": r"/kaggle/working/downloaded_files/dir_4/math500_run_log.json",
        "grouping": r"/kaggle/working/downloaded_files/dir_10/math500_grouping_run_log.json",
    },
]

HIDDEN_DIM = 128
NUM_RESIDUAL_BLOCKS = 2
DROPOUT_RATE = 0.10
EPOCHS = 100
PATIENCE = 20
LR = 0.0003
WEIGHT_DECAY = 1e-4
QUERY_BATCH_SIZE = 16
VALIDATION_FRACTION = 0.15
SEED = 75
STRICT_RETRIEVAL_MATCH = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. SCHEMA AND LABEL HELPERS
# ==========================================
def _dictionary_get(mapping, key, default=None):
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    text_key = str(key)
    if text_key in mapping:
        return mapping[text_key]
    try:
        int_key = int(key)
    except (TypeError, ValueError):
        return default
    return mapping.get(int_key, default)


def _query_text(record, state):
    return str(
        record.get("target_query_text")
        or state.get("target_query_data", {}).get("query_text")
        or ""
    ).strip()


def _load_states(file_path, state_key):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    records = OrderedDict()
    if isinstance(raw, dict) and isinstance(raw.get("queries"), dict):
        iterable = [
            ({"target_query_original_hard_list_idx": query_id}, state)
            for query_id, state in raw["queries"].items()
        ]
    elif isinstance(raw, list):
        iterable = []
        for fallback_index, record in enumerate(raw):
            if not isinstance(record, dict):
                continue
            state = record.get(state_key)
            if state is None and state_key == "layer1_base_execution_state":
                state = record if "candidate_set" in record else None
            if state is None and state_key == "layer1_grouping_state":
                state = record if "candidate_set" in record else None
            if isinstance(state, dict):
                wrapped = dict(record)
                wrapped.setdefault("target_query_original_hard_list_idx", fallback_index)
                iterable.append((wrapped, state))
    else:
        raise ValueError(f"Unsupported JSON schema in {file_path}")

    for record, state in iterable:
        query_id = str(record.get("target_query_original_hard_list_idx"))
        if query_id in records:
            raise ValueError(f"Duplicate query ID {query_id!r} in {file_path}")
        records[query_id] = {
            "state": state,
            "query_text": _query_text(record, state),
        }
    return records


def _candidate_label(labels, candidate_id):
    entry = _dictionary_get(labels, candidate_id, {})
    if not isinstance(entry, dict) or not isinstance(entry.get("is_correct"), bool):
        return None
    return bool(entry["is_correct"])


def _is_successful_candidate(candidate):
    return isinstance(candidate, dict) and bool(candidate.get("candidate_text"))


def _zero_shot_sort_key(candidate_id):
    text = str(candidate_id)
    if text.startswith("zs_"):
        try:
            return (0, int(text[3:]))
        except ValueError:
            pass
    return (1, text)


def _first_answered_zero_shot(candidate_set, labels):
    zero_ids = []
    for candidate_id, candidate in candidate_set.items():
        source = str(candidate.get("source_exemplar_idx"))
        if source in {"-1", "None"} or str(candidate_id).startswith("zs_"):
            zero_ids.append(candidate_id)
    for candidate_id in sorted(zero_ids, key=_zero_shot_sort_key):
        candidate = candidate_set[candidate_id]
        label = _candidate_label(labels, candidate_id)
        if _is_successful_candidate(candidate) and label is not None:
            return candidate_id, candidate, label
    return None


def _retrieval_signature(state):
    return [str(item.get("corpus_index")) for item in state.get("retrieved_set", [])]


def _paired_states(layer1_path, grouping_path):
    layer1 = _load_states(layer1_path, "layer1_base_execution_state")
    grouping = _load_states(grouping_path, "layer1_grouping_state")
    shared_ids = [query_id for query_id in layer1 if query_id in grouping]
    if not shared_ids:
        raise ValueError(
            f"No shared query IDs between {layer1_path} and {grouping_path}."
        )
    return layer1, grouping, shared_ids


def _cross_eval_vector(cross_eval, candidate_id, evaluator_ids):
    row = _dictionary_get(cross_eval, candidate_id, {})
    return [float(_dictionary_get(row, evaluator_id, 0.0) or 0.0) for evaluator_id in evaluator_ids]


def _one_shot_by_source(candidate_set, labels, evaluator_ids):
    by_source = {}
    for evaluator_id in evaluator_ids:
        for candidate_id, candidate in candidate_set.items():
            if str(candidate.get("source_exemplar_idx")) != evaluator_id:
                continue
            label = _candidate_label(labels, candidate_id)
            if _is_successful_candidate(candidate) and label is not None:
                by_source[evaluator_id] = (candidate_id, candidate, label)
                break
    return by_source


def _make_features(
    member_ids,
    similarity_by_id,
    evaluator_similarities,
    evaluator_baselines,
    aggregate_ccs,
    max_k,
    is_zero,
    ccs_coverage,
):
    member_scores = [similarity_by_id[member_id] for member_id in member_ids]
    group_size = len(member_ids)
    if member_scores:
        similarity_sum = float(sum(member_scores))
        similarity_mean = float(np.mean(member_scores))
        similarity_min = float(min(member_scores))
        similarity_max = float(max(member_scores))
    else:
        similarity_sum = similarity_mean = similarity_min = similarity_max = 0.0

    member_set = set(member_ids)
    member_mask = [
        1.0 if evaluator_id in member_set and evaluator_id != "PAD" else 0.0
        for evaluator_id in evaluator_similarities["ids"]
    ]
    scalar_features = [
        float(is_zero),
        float(group_size / max_k),
        similarity_sum,
        similarity_mean,
        similarity_min,
        similarity_max,
        float(ccs_coverage),
    ]
    return (
        scalar_features
        + evaluator_similarities["values"]
        + evaluator_baselines
        + aggregate_ccs
        + member_mask
    )


def build_question_options(layer1_record, grouping_record, max_k, strict=True):
    layer1_state = layer1_record["state"]
    grouping_state = grouping_record["state"]
    layer1_signature = _retrieval_signature(layer1_state)
    grouping_signature = _retrieval_signature(grouping_state)
    if layer1_signature != grouping_signature:
        message = (
            "Layer-1 and grouping retrievals differ: "
            f"{layer1_signature} != {grouping_signature}"
        )
        if strict:
            raise ValueError(message)
        return None

    if (
        layer1_record["query_text"]
        and grouping_record["query_text"]
        and layer1_record["query_text"] != grouping_record["query_text"]
    ):
        if strict:
            raise ValueError("Paired query IDs contain different target question text.")
        return None

    retrieved = layer1_state.get("retrieved_set", [])
    if not retrieved:
        return None
    actual_k = len(retrieved)
    evaluator_ids = [str(item.get("corpus_index")) for item in retrieved]
    similarities = [float(item.get("similarity_score") or 0.0) for item in retrieved]
    similarity_by_id = dict(zip(evaluator_ids, similarities))
    baselines_map = layer1_state.get("intrinsic_baselines", {})
    baselines = [float(_dictionary_get(baselines_map, item, 0.0) or 0.0) for item in evaluator_ids]

    while len(evaluator_ids) < max_k:
        evaluator_ids.append("PAD")
        similarities.append(0.0)
        baselines.append(0.0)
    evaluator_info = {"ids": evaluator_ids, "values": similarities}

    candidate_set = layer1_state.get("candidate_set", {})
    labels = layer1_state.get("ground_truth_labels", {})
    cross_eval = layer1_state.get("cross_evaluation_matrix", {})
    zero = _first_answered_zero_shot(candidate_set, labels)
    if zero is None:
        return None

    real_evaluator_ids = evaluator_ids[:actual_k]
    one_shot = _one_shot_by_source(candidate_set, labels, real_evaluator_ids)
    options = []

    zero_id, _zero_candidate, zero_label = zero
    zero_ccs = _cross_eval_vector(cross_eval, zero_id, evaluator_ids)
    options.append(
        {
            "option_id": f"zero::{zero_id}",
            "source_candidate_id": str(zero_id),
            "group_size": 0,
            "member_ids": [],
            "similarity_sum": 0.0,
            "is_correct": zero_label,
            "features": _make_features(
                [], similarity_by_id, evaluator_info, baselines, zero_ccs,
                max_k, True, 1.0,
            ),
        }
    )

    for evaluator_id in real_evaluator_ids:
        if evaluator_id not in one_shot:
            continue
        candidate_id, _candidate, label = one_shot[evaluator_id]
        ccs = _cross_eval_vector(cross_eval, candidate_id, evaluator_ids)
        options.append(
            {
                "option_id": f"one::{candidate_id}",
                "source_candidate_id": str(candidate_id),
                "group_size": 1,
                "member_ids": [evaluator_id],
                "similarity_sum": similarity_by_id[evaluator_id],
                "is_correct": label,
                "features": _make_features(
                    [evaluator_id], similarity_by_id, evaluator_info, baselines,
                    ccs, max_k, False, 1.0,
                ),
            }
        )

    grouping_candidates = grouping_state.get("candidate_set", {})
    grouping_labels = grouping_state.get("ground_truth_labels", {})
    sortable_groups = []
    for candidate_id, candidate in grouping_candidates.items():
        label = _candidate_label(grouping_labels, candidate_id)
        if not _is_successful_candidate(candidate) or label is None:
            continue
        member_ids = [str(value) for value in candidate.get("source_exemplar_indices", [])]
        if len(member_ids) < 2 or any(member_id not in similarity_by_id for member_id in member_ids):
            if strict:
                raise ValueError(
                    f"Grouping candidate {candidate_id!r} does not map to the Layer-1 retrieval set."
                )
            continue
        positions = tuple(candidate.get("source_retrieval_positions", []))
        sortable_groups.append((len(member_ids), positions, str(candidate_id), candidate, label, member_ids))

    for group_size, _positions, candidate_id, _candidate, label, member_ids in sorted(sortable_groups):
        member_vectors = []
        for member_id in member_ids:
            if member_id in one_shot:
                member_candidate_id = one_shot[member_id][0]
                member_vectors.append(
                    _cross_eval_vector(cross_eval, member_candidate_id, evaluator_ids)
                )
        if member_vectors:
            aggregate_ccs = np.mean(np.asarray(member_vectors), axis=0).tolist()
        else:
            aggregate_ccs = [0.0] * max_k
        coverage = len(member_vectors) / group_size
        options.append(
            {
                "option_id": f"group::{candidate_id}",
                "source_candidate_id": candidate_id,
                "group_size": group_size,
                "member_ids": member_ids,
                "similarity_sum": float(sum(similarity_by_id[item] for item in member_ids)),
                "is_correct": label,
                "features": _make_features(
                    member_ids, similarity_by_id, evaluator_info, baselines,
                    aggregate_ccs, max_k, False, coverage,
                ),
            }
        )

    if len(options) < 2:
        return None

    correct_indices = [index for index, option in enumerate(options) if option["is_correct"]]
    if correct_indices:
        teacher_index = min(
            correct_indices,
            key=lambda index: (
                options[index]["group_size"],
                -options[index]["similarity_sum"],
                index,
            ),
        )
    else:
        teacher_index = 0

    return {
        "options": options,
        "teacher_index": teacher_index,
        "has_correct_option": bool(correct_indices),
        "query_text": layer1_record["query_text"] or grouping_record["query_text"],
    }


def discover_global_max_k(file_pairs):
    max_k = 0
    for pair in file_pairs:
        layer1 = _load_states(pair["layer1"], "layer1_base_execution_state")
        for record in layer1.values():
            max_k = max(max_k, len(record["state"].get("retrieved_set", [])))
    if max_k < 1:
        raise ValueError("No Layer-1 retrieval records were found.")
    return max_k


def load_pair_dataset(layer1_path, grouping_path, max_k, namespace):
    layer1, grouping, shared_ids = _paired_states(layer1_path, grouping_path)
    dataset = OrderedDict()
    skipped = 0
    for query_id in shared_ids:
        built = build_question_options(
            layer1[query_id], grouping[query_id], max_k, STRICT_RETRIEVAL_MATCH
        )
        if built is None:
            skipped += 1
            continue
        dataset[f"{namespace}::{query_id}"] = built
    print(
        f"Loaded {namespace}: {len(dataset)} usable paired queries; "
        f"{skipped} skipped; {len(layer1) - len(shared_ids)} missing grouping partners."
    )
    return dataset


# ==========================================
# 3. LISTWISE TABULAR RESNET
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, values):
        residual = values
        values = self.relu(self.bn1(self.fc1(values)))
        values = self.dropout(values)
        values = self.bn2(self.fc2(values))
        return self.relu(values + residual)


class ContextSelectorResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.10, num_blocks=2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, values):
        values = self.input_layer(values)
        for block in self.res_blocks:
            values = block(values)
        return self.output_layer(values).squeeze(-1)


def _feature_tensor(question, mean_tensor, std_tensor):
    values = torch.tensor(
        [option["features"] for option in question["options"]],
        dtype=torch.float32,
        device=device,
    )
    return (values - mean_tensor) / std_tensor


def evaluate_selector(model, dataset, query_ids, mean_tensor, std_tensor):
    model.eval()
    teacher_hits = []
    answer_hits = []
    reciprocal_ranks = []
    selected_sizes = []
    teacher_sizes = []
    zero_hits = []
    oracle_hits = []
    with torch.no_grad():
        for query_id in query_ids:
            question = dataset[query_id]
            logits = model(_feature_tensor(question, mean_tensor, std_tensor))
            ranked = torch.argsort(logits, descending=True).cpu().tolist()
            selected_index = ranked[0]
            teacher_index = question["teacher_index"]
            teacher_hits.append(float(selected_index == teacher_index))
            answer_hits.append(float(question["options"][selected_index]["is_correct"]))
            reciprocal_ranks.append(1.0 / (ranked.index(teacher_index) + 1))
            selected_sizes.append(question["options"][selected_index]["group_size"])
            teacher_sizes.append(question["options"][teacher_index]["group_size"])
            zero_hits.append(float(question["options"][0]["is_correct"]))
            oracle_hits.append(float(question["has_correct_option"]))
    return {
        "queries": len(query_ids),
        "teacher_top1": float(np.mean(teacher_hits)) if teacher_hits else 0.0,
        "teacher_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "selected_answer_accuracy": float(np.mean(answer_hits)) if answer_hits else 0.0,
        "zero_shot_accuracy": float(np.mean(zero_hits)) if zero_hits else 0.0,
        "oracle_coverage": float(np.mean(oracle_hits)) if oracle_hits else 0.0,
        "mean_selected_group_size": float(np.mean(selected_sizes)) if selected_sizes else 0.0,
        "mean_teacher_group_size": float(np.mean(teacher_sizes)) if teacher_sizes else 0.0,
    }


def print_report(name, metrics):
    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)
    print(f"Usable paired queries       : {metrics['queries']}")
    print(f"Exact teacher choice Top-1  : {metrics['teacher_top1']:.4f}")
    print(f"Teacher reciprocal rank     : {metrics['teacher_mrr']:.4f}")
    print(f"Selected-answer accuracy    : {metrics['selected_answer_accuracy']:.4f}")
    print(f"First-zero-shot accuracy    : {metrics['zero_shot_accuracy']:.4f}")
    print(f"Candidate oracle coverage   : {metrics['oracle_coverage']:.4f}")
    print(f"Mean selected group size    : {metrics['mean_selected_group_size']:.3f}")
    print(f"Mean teacher group size     : {metrics['mean_teacher_group_size']:.3f}")


# ==========================================
# 4. TRAIN ONCE, EVALUATE EACH BENCHMARK
# ==========================================
def main():
    print(f"Using compute device: {device.type.upper()}")
    train_pair = {
        "name": "train",
        "layer1": TRAIN_LAYER1_FILE_PATH,
        "grouping": TRAIN_GROUPING_FILE_PATH,
    }
    all_pairs = [train_pair] + TEST_FILE_PAIRS
    max_k = discover_global_max_k(all_pairs)
    input_dim = 7 + 4 * max_k
    print(f"Global retrieval width K={max_k}; feature dimension={input_dim}")

    training_dataset = load_pair_dataset(
        TRAIN_LAYER1_FILE_PATH,
        TRAIN_GROUPING_FILE_PATH,
        max_k,
        "train",
    )
    test_datasets = {
        pair["name"]: load_pair_dataset(
            pair["layer1"], pair["grouping"], max_k, pair["name"]
        )
        for pair in TEST_FILE_PAIRS
    }
    if len(training_dataset) < 2:
        raise ValueError("At least two usable training queries are required.")
    if any(not dataset for dataset in test_datasets.values()):
        empty = [name for name, dataset in test_datasets.items() if not dataset]
        raise ValueError(f"No usable paired test queries for: {empty}")

    shuffled_ids = list(training_dataset)
    random.Random(SEED).shuffle(shuffled_ids)
    validation_count = max(1, int(round(len(shuffled_ids) * VALIDATION_FRACTION)))
    validation_count = min(validation_count, len(shuffled_ids) - 1)
    validation_ids = shuffled_ids[:validation_count]
    train_ids = shuffled_ids[validation_count:]
    print(f"Train queries={len(train_ids)}; validation queries={len(validation_ids)}")

    train_features = torch.tensor(
        [
            option["features"]
            for query_id in train_ids
            for option in training_dataset[query_id]["options"]
        ],
        dtype=torch.float32,
        device=device,
    )
    train_mean = train_features.mean(dim=0, keepdim=True)
    train_std = train_features.std(dim=0, keepdim=True)
    train_std[~torch.isfinite(train_std) | (train_std == 0)] = 1.0

    model = ContextSelectorResNet(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        num_blocks=NUM_RESIDUAL_BLOCKS,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_weights = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_ids = list(train_ids)
        random.Random(SEED + epoch).shuffle(epoch_ids)
        optimizer.zero_grad()
        pending_losses = []
        for position, query_id in enumerate(epoch_ids, start=1):
            question = training_dataset[query_id]
            logits = model(_feature_tensor(question, train_mean, train_std))
            target = torch.tensor(
                [question["teacher_index"]], dtype=torch.long, device=device
            )
            pending_losses.append(criterion(logits.unsqueeze(0), target))
            if len(pending_losses) == QUERY_BATCH_SIZE or position == len(epoch_ids):
                torch.stack(pending_losses).mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                pending_losses = []

        validation_metrics = evaluate_selector(
            model, training_dataset, validation_ids, train_mean, train_std
        )
        current_accuracy = validation_metrics["teacher_top1"]
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}: validation teacher Top-1="
                f"{current_accuracy:.4f}"
            )
        if patience_counter >= PATIENCE:
            print(
                f"Early stop at epoch {epoch + 1}; best validation teacher "
                f"Top-1={best_accuracy:.4f}"
            )
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    print_report(
        "HELD-OUT TRAINING-SOURCE VALIDATION",
        evaluate_selector(
            model, training_dataset, validation_ids, train_mean, train_std
        ),
    )
    benchmark_metrics = {}
    for name, dataset in test_datasets.items():
        metrics = evaluate_selector(
            model, dataset, list(dataset), train_mean, train_std
        )
        benchmark_metrics[name] = metrics
        print_report(f"TEST BENCHMARK: {name}", metrics)

    macro = {
        key: float(np.mean([metrics[key] for metrics in benchmark_metrics.values()]))
        for key in next(iter(benchmark_metrics.values()))
        if key != "queries"
    }
    macro["queries"] = sum(metrics["queries"] for metrics in benchmark_metrics.values())
    print_report("EQUAL-WEIGHT MEAN ACROSS TEST BENCHMARKS", macro)
    print("\nContext-selector training and evaluation complete.")


if __name__ == "__main__":
    main()
