"""Offline synthetic timing of the two-phase Stage 3 at the default geometry.

Run: python -B benchmark_adaptive_stage3.py --output stage3_two_phase_performance_report.json
No provider calls, dataset downloads, or Kaggle runtime claims. The predictor is untrained, so how many
goals are reachable (and with it the share of informative rows) says nothing about real logs.
"""
import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

import adaptive_analogical_training as current


def timed(call):
    started = time.perf_counter()
    value = call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return value, time.perf_counter() - started


def synthetic_question(cfg, index):
    rng = np.random.RandomState(810 + index)
    record = current.Record(
        f"synthetic::{index}", str(index), "synthetic", [f"c{i}" for i in range(cfg.n)],
        [f"e{i}" for i in range(cfg.k)], np.sort(rng.rand(cfg.k).astype(np.float32))[::-1].copy(),
        (rng.randint(cfg.repeats + 1, size=cfg.k) / cfg.repeats).astype(np.float32),
        (rng.randint(cfg.repeats + 1, size=(cfg.n, cfg.k)) / cfg.repeats).astype(np.float32),
        (rng.rand(cfg.n) > .5).astype(np.float32))
    scores = rng.rand(cfg.n).astype(np.float32)
    safe, maximum = current.teacher_labels(scores, record.labels, cfg)
    return record, scores, safe, maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("stage3_two_phase_performance_report.json"))
    args = parser.parse_args()
    if args.questions < 2 or args.repeats < 1:
        parser.error("--questions needs at least 2 and --repeats at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    cfg = current.Config()
    model = current.ResNet(cfg.input_dim, 3 * cfg.n + 2, cfg)
    predictor = current.FrozenPredictor({"weights": current.cpu_state(model), "temperatures": np.ones(5)},
                                        cfg, device)
    questions = [synthetic_question(cfg, i) for i in range(args.questions)]
    head = current.SupervisedActionHead(cfg).to(device)
    cases = current.decision_case_count(cfg)
    report = {"evidence": "synthetic microbenchmark with an untrained predictor; not a live Kaggle run",
              "device": str(device), "torch_version": torch.__version__,
              "config": {"k": cfg.k, "zero_shots": cfg.zero_shots, "decision_batch_size": cfg.decision_batch_size,
                         "decision_eval_batch_size": cfg.decision_eval_batch_size},
              "state_order_cases_per_question": cases, "questions": []}
    # Warm the shared, question-independent tables so each figure is the per-question increment.
    current.build_route_labels(*questions[0], predictor, cfg)
    current.build_finetune_labels(*questions[0], predictor, head, cfg)
    with tempfile.TemporaryDirectory(prefix="adaptive_stage3_benchmark_") as temp:
        for index, question in enumerate(questions):
            entry = {"question": index}
            route, entry["route_label_seconds"] = timed(lambda: current.build_route_labels(*question, predictor, cfg))
            fine, entry["finetune_label_seconds"] = timed(
                lambda: current.build_finetune_labels(*question, predictor, head, cfg))
            for name, labels in (("route", route), ("finetune", fine)):
                path = Path(temp) / f"{name}.pt"
                current.atomic_torch(path, {"schema_version": current.DECISION_LABEL_SCHEMA, **labels})
                entry[f"{name}_file_bytes"] = path.stat().st_size
            entry["rows"] = route["coverage"]["action_rows"]
            entry["informative_rows"] = route["coverage"]["informative_rows"]
            entry["rows_moved_by_finetuning"] = fine["coverage"]["changed_rows"]
            report["questions"].append(entry)
            print(json.dumps(entry), flush=True)
        # One training question and one development question of a head epoch.
        record, scores, safe, maximum = questions[1]
        route = current.build_route_labels(record, scores, safe, maximum, predictor, cfg)
        values = route["options"].numpy()
        source = current.label_rows(values, "pretrain")
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg.decision_lr,
                                     fused=device.type == "cuda")
        epoch = {}
        for repeat in range(args.repeats + 1):
            rows, assemble = timed(lambda: current.decision_training_rows(record, source, values[source],
                                                                          "pretrain", predictor, cfg))
            order = np.random.RandomState(repeat).permutation(len(source))
            _, train = timed(lambda: current.masked_action_loss(head, rows, order, cfg, device, optimizer))
            with torch.no_grad():
                _, dev = timed(lambda: current.masked_action_loss(head, rows, range(len(source)), cfg, device,
                                                                  batch_size=cfg.decision_eval_batch_size))
            if repeat:  # The first pass warms the device.
                for key, value in (("assemble", assemble), ("train_steps", train), ("dev_pass", dev)):
                    epoch.setdefault(key, []).append(value)
        report["head_epoch_seconds_per_question"] = {key: statistics.median(values) for key, values in epoch.items()}
        # Peak Python/NumPy allocations of one question's label build.
        for name, build in (("route", lambda: current.build_route_labels(*questions[1], predictor, cfg)),
                            ("finetune", lambda: current.build_finetune_labels(*questions[1], predictor, head, cfg))):
            tracemalloc.start()
            build()
            report[f"{name}_label_peak_mib"] = tracemalloc.get_traced_memory()[1] / 2**20
            tracemalloc.stop()
    for key in ("route_label_seconds", "finetune_label_seconds", "route_file_bytes", "finetune_file_bytes", "rows"):
        report[f"median_{key}"] = statistics.median(entry[key] for entry in report["questions"])
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "questions"}, indent=2))


if __name__ == "__main__":
    main()
