"""Offline synthetic CPU comparison against the pre-optimization Stage 3 source.

Run: python -B benchmark_adaptive_stage3.py --output stage3_performance_report.json
No provider calls, dataset downloads, or Kaggle/GPU runtime claims.
"""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import adaptive_analogical_training as current


def timed(call):
    started = time.perf_counter()
    value = call()
    return value, time.perf_counter() - started


def check_labels(expected, actual):
    old_rows, old_cases = expected
    rows, cases = actual
    assert cases == old_cases and len(rows) == len(old_rows)
    for before, after in zip(old_rows, rows):
        assert before.keys() == after.keys()
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(before[key], after[key])
            else:
                assert before[key] == after[key], key


def measure_files_and_head(rows, cases, cfg, directory, repeats):
    audit, packed = directory / "audit.pt", directory / "training.pt"
    current.atomic_torch(audit, {"rows": rows, "cases": cases})
    current.atomic_torch(packed, current.pack_decision_rows(rows, cfg))
    measurements = {"audit_file_bytes": audit.stat().st_size,
                    "training_file_bytes": packed.stat().st_size}
    for name, path in (("audit", audit), ("training", packed)):
        times = [timed(lambda: current.load_checkpoint(path))[1] for _ in range(repeats)]
        measurements[name + "_load_seconds_median"] = statistics.median(times)
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="8afb15c")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("stage3_performance_report.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = Path(__file__).resolve().parent
    source = subprocess.run(["git", "show", f"{args.baseline_ref}:adaptive_analogical_training.py"],
                            cwd=root, check=True, capture_output=True, text=True).stdout
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = current.Config()
    model = current.ResNet(cfg.input_dim, 3*cfg.n+2, cfg)
    checkpoint = {"weights": current.cpu_state(model), "temperatures": np.ones(5)}
    report = {"evidence": "synthetic CPU microbenchmark; not a live Kaggle/GPU run",
              "baseline_ref": args.baseline_ref, "torch_version": torch.__version__,
              "cpu_threads": 1, "config": {"k": cfg.k, "zero_shots": cfg.zero_shots,
                                           "decision_batch_size": cfg.decision_batch_size},
              "questions": []}
    with tempfile.TemporaryDirectory(prefix="adaptive_stage3_benchmark_") as temp:
        directory = Path(temp)
        baseline_path = directory / "baseline.py"
        baseline_path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("adaptive_stage3_baseline", baseline_path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        predictors = {"baseline": baseline.FrozenPredictor(checkpoint, baseline.Config(), "cpu"),
                      "optimized": current.FrozenPredictor(checkpoint, cfg, "cpu")}
        for index in range(args.repeats):
            rng = np.random.RandomState(810 + index)
            data = dict(uid=f"synthetic_{index}", group=f"synthetic_{index}", benchmark="synthetic",
                        candidate_ids=[f"c{i}" for i in range(cfg.n)],
                        evaluator_ids=[f"e{i}" for i in range(cfg.k)],
                        similarity=np.linspace(.9, .5, cfg.k, dtype=np.float32),
                        baseline=(rng.randint(cfg.repeats+1, size=cfg.k)/cfg.repeats).astype(np.float32),
                        ccs=(rng.randint(cfg.repeats+1, size=(cfg.n, cfg.k))/cfg.repeats).astype(np.float32),
                        labels=np.ones(cfg.n, np.float32))
            scores = np.arange(cfg.n, 0, -1, dtype=np.float32)
            safe, maximum = np.ones(cfg.n, np.float32), np.zeros(cfg.n, np.float32)
            maximum[0] = 1
            results, entry = {}, {"question": index}
            # Alternate measurement order to reduce systematic cache-order bias.
            names = ("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")
            for name in names:
                module = baseline if name == "baseline" else current
                record = module.Record(**data)
                module_cfg = baseline.Config() if name == "baseline" else cfg
                results[name], seconds = timed(lambda: module.build_decision_rows(
                    record, scores, safe, maximum, predictors[name], module_cfg))
                entry[name + "_label_seconds"] = seconds
            check_labels(results["baseline"], results["optimized"])
            rows, cases = results["optimized"]
            entry.update(action_rows=len(rows), state_order_cases=len(cases), exact_label_match=True)
            print(json.dumps(entry), flush=True)
            report["questions"].append(entry)
        assert rows, "Need actionable rows for head benchmark"
        report["files"] = measure_files_and_head(rows, cases, cfg, directory, max(5, args.repeats))
        order = np.random.RandomState(91).permutation(len(rows))
        packed = current.pack_decision_rows(rows, cfg)
        torch.manual_seed(19)
        head = current.SupervisedActionHead(cfg)
        # Warm optimizer imports before timing either implementation.
        torch.optim.Adam(head.parameters(), lr=cfg.decision_lr)
        head_results, times = {}, {"baseline": [], "optimized": []}
        for name, module, inputs in (("baseline", baseline, rows), ("optimized", current, packed)):
            for _ in range(args.repeats):
                trained = current.SupervisedActionHead(cfg)
                trained.load_state_dict(head.state_dict())
                opt = torch.optim.Adam(trained.parameters(), lr=cfg.decision_lr, weight_decay=cfg.weight_decay)
                result, seconds = timed(lambda: module.masked_action_loss(trained, inputs, order, cfg, "cpu", opt))
                times[name].append(seconds)
                head_results[name] = (result, current.cpu_state(trained))
        before, after = head_results["baseline"], head_results["optimized"]
        assert before[0] == after[0]
        assert all(torch.equal(value, after[1][key]) for key, value in before[1].items())
        report["head"] = {name + "_train_seconds_median": statistics.median(values)
                          for name, values in times.items()}
        report["head"]["exact_loss_and_weights_match"] = True
    old = statistics.median(q["baseline_label_seconds"] for q in report["questions"])
    new = statistics.median(q["optimized_label_seconds"] for q in report["questions"])
    report["label_seconds_median"] = {"baseline": old, "optimized": new, "speedup": old/new}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
