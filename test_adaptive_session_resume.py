"""Offline checks that a run survives a Kaggle session: content-hashed input contract,
fold-level Stage 1 resume, and the Hugging Face mirror of the output folder."""
import copy
import fnmatch
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import adaptive_analogical_training as a
from test_adaptive_analogical_training import raw_record


class FakeHub:
    """In-memory stand-in for HfApi with the Hub's pattern semantics."""
    def __init__(self):
        self.repos, self.messages, self.fail_uploads = {}, [], 0

    def repo_exists(self, repo_id, *, repo_type=None, token=None):
        return repo_id in self.repos

    def create_repo(self, repo_id, *, repo_type=None, private=None, exist_ok=False, **kwargs):
        assert repo_type == "dataset" and private and exist_ok
        self.repos.setdefault(repo_id, {})

    def snapshot_download(self, repo_id, *, repo_type=None, local_dir=None, ignore_patterns=None, **kwargs):
        local = Path(local_dir)
        for relative, data in self.repos[repo_id].items():
            if any(fnmatch.fnmatch(relative, pattern) for pattern in ignore_patterns or ()):
                continue
            target = local / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        # The real download keeps per-file metadata under local_dir/.cache/huggingface.
        bookkeeping = local / ".cache" / "huggingface" / "download" / "contract.json.metadata"
        bookkeeping.parent.mkdir(parents=True, exist_ok=True)
        bookkeeping.write_text("etag")
        return str(local)

    def upload_folder(self, *, repo_id, folder_path, repo_type=None, commit_message=None,
                      ignore_patterns=None, delete_patterns=None, **kwargs):
        if self.fail_uploads:
            self.fail_uploads -= 1
            raise ConnectionError("simulated upload failure")
        folder, remote = Path(folder_path), self.repos.setdefault(repo_id, {})
        local = {}
        for path in folder.rglob("*"):
            relative = path.relative_to(folder).as_posix()
            if not path.is_file() or relative.startswith(".cache/"):
                continue
            if any(fnmatch.fnmatch(relative, pattern) for pattern in ignore_patterns or ()):
                continue
            local[relative] = path.read_bytes()
        # Remote files matching delete_patterns go unless this upload re-adds them.
        for relative in list(remote):
            if relative not in local and any(fnmatch.fnmatch(relative, pattern)
                                             for pattern in delete_patterns or ()):
                del remote[relative]
        remote.update(local)
        self.messages.append(commit_message)
        return f"https://hub/{repo_id}/commit/{len(self.messages)}"


def small_config(root, **overrides):
    train, external = root / "train.json", root / "external.json"
    cfg = a.Config(train_file=str(train), test_files=[str(external)],
                   use_test_files_for_dev_and_audit=True,
                   output_dir=str(root / "run"), device="cpu", cpu_threads=1,
                   k=2, zero_shots=2, teacher_folds=2, hidden_dim=8, residual_blocks=1,
                   batch_size=16, teacher_epochs=2, snapshot_epochs=2, decision_epochs=2,
                   patience=2, snapshots_per_query=4, dev_snapshots_per_query=4,
                   decision_batch_size=16, recognition_threshold=.1,
                   bootstrap_samples=20, **overrides)
    if not train.exists():
        train.write_text(json.dumps([raw_record(i, cfg) for i in range(40)]))
        external.write_text(json.dumps([raw_record(i, cfg, prefix="external") for i in range(3)]))
    return cfg


def test_contract_identifies_input_logs_by_content_not_path_or_mtime(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(tmp_path)
    a.Workflow(cfg)
    path = tmp_path / "run" / "contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert all(set(source) == {"path", "name", "bytes", "sha256"} for source in contract["sources"])
    assert contract["sources"][0]["sha256"] == a.file_digest(cfg.train_file)
    assert not {"hf_repo", "hf_sync_minutes", "train_file", "test_files"} & set(contract["config"])
    # A new Kaggle session downloads the same logs again: new mtime, same bytes.
    train = Path(cfg.train_file)
    data = train.read_bytes()
    train.write_bytes(data)
    os.utime(train, (1, 1))
    a.Workflow(cfg)
    # The same file in another directory is the same run; its name is the benchmark identity.
    moved = tmp_path / "elsewhere" / "train.json"
    moved.parent.mkdir()
    shutil.copyfile(train, moved)
    moved_cfg = copy.deepcopy(cfg)
    moved_cfg.train_file = str(moved)
    a.Workflow(moved_cfg)
    renamed = tmp_path / "elsewhere" / "numina.json"
    shutil.copyfile(train, renamed)
    renamed_cfg = copy.deepcopy(cfg)
    renamed_cfg.train_file = str(renamed)
    with pytest.raises(ValueError, match="different config/data contract"):
        a.Workflow(renamed_cfg)
    # Different bytes of the same size are another run.
    tampered = bytearray(data)
    tampered[1] ^= 1
    train.write_bytes(bytes(tampered))
    with pytest.raises(ValueError, match="different config/data contract"):
        a.Workflow(cfg)


def test_legacy_mtime_contract_is_upgraded_only_when_it_still_matches(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(tmp_path)
    a.Workflow(cfg)
    path = tmp_path / "run" / "contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    # Earlier code kept the configured paths in the config and bound each log to its mtime.
    legacy = dict(contract,
                  config={**contract["config"], "train_file": cfg.train_file, "test_files": list(cfg.test_files)},
                  sources=[{"path": source["path"], "bytes": source["bytes"],
                            "mtime_ns": Path(source["path"]).stat().st_mtime_ns}
                           for source in contract["sources"]])
    path.write_text(json.dumps(legacy), encoding="utf-8")
    a.Workflow(cfg)
    assert json.loads(path.read_text(encoding="utf-8")) == contract
    # A legacy contract from another session cannot be verified and is rejected as before.
    stale = dict(legacy, sources=[dict(source, mtime_ns=source["mtime_ns"] + 1) for source in legacy["sources"]])
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="different config/data contract"):
        a.Workflow(cfg)


def test_mirror_settings_are_runtime_only_and_validated():
    cfg = a.Config(hidden_dim=4)
    records = [a.Record(str(i), str(i), "synthetic", list(range(cfg.n)), list(range(cfg.k)),
                        np.ones(cfg.k, np.float32), np.ones(cfg.k, np.float32),
                        np.ones((cfg.n, cfg.k), np.float32), np.ones(cfg.n, np.float32))
               for i in range(2)]
    teacher = {name: np.ones((2, cfg.n), np.float32) for name in ("scores", "safe", "maximum")}
    predictor = SimpleNamespace(model=torch.nn.Linear(4, 4), temperatures=np.ones(5))
    splits = {"policy": [0], "dev": [1]}
    first = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    mirrored = copy.deepcopy(cfg)
    mirrored.hf_repo, mirrored.hf_sync_minutes = "user/run", 1
    mirrored.validate()
    assert a.decision_dataset_fingerprint(records, splits, teacher, predictor, mirrored) == first
    with pytest.raises(ValueError, match="hf_repo"):
        a.Config(hf_repo="not a repository id").validate()
    with pytest.raises(AssertionError):
        a.Config(hf_sync_minutes=0).validate()


def test_stage3_fingerprint_ignores_log_paths_and_reproduces_the_path_bound_form():
    cfg = a.Config(hidden_dim=4)
    records = [a.Record(f"bench::{i}", str(i), "bench", list(range(cfg.n)), list(range(cfg.k)),
                        np.full(cfg.k, .5, np.float32), np.full(cfg.k, .4, np.float32),
                        np.full((cfg.n, cfg.k), .2, np.float32), np.ones(cfg.n, np.float32))
               for i in range(3)]
    teacher = {name: np.arange(3 * cfg.n, dtype=np.float32).reshape(3, cfg.n) / 10
               for name in ("scores", "safe", "maximum")}
    linear = torch.nn.Linear(4, 4)
    with torch.no_grad():
        linear.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)
        linear.bias.zero_()
    predictor = SimpleNamespace(model=linear, temperatures=np.ones(5))
    splits = {"policy": [0, 1], "dev": [2]}
    current = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    moved = copy.deepcopy(cfg)
    moved.train_file = "/content/working/logs/" + Path(cfg.train_file).name
    moved.test_files = ["/content/working/logs/" + Path(path).name for path in cfg.test_files]
    assert a.decision_dataset_fingerprint(records, splits, teacher, predictor, moved) == current
    # Code through e4b82f0 also hashed the configured paths; this is the value it computed.
    legacy = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg,
                                            [cfg.train_file] + cfg.test_files)
    assert legacy == "4bf5e8fa5b7d05e1bb5afe79cc01ad489572e73b222dc98d6de31c08d17a7591" != current
    assert a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg,
                                          [moved.train_file] + moved.test_files) not in (legacy, current)


def test_output_sync_pushes_changed_files_pulls_into_empty_folders_and_survives_failures(
        tmp_path, monkeypatch, capsys):
    hub = FakeHub()
    out = tmp_path / "out"
    out.mkdir()
    sync = a.OutputSync(out, "user/run", 1000, api=hub)
    assert sync.pull() == "new"
    (out / "a.json").write_text("1")
    (out / "partial.pt.tmp").write_bytes(b"never uploaded")
    (out / "decision_labels" / "policy").mkdir(parents=True)
    (out / "decision_labels" / "policy" / "000000_q.pt").write_bytes(b"question")
    assert sync.push("first", force=True) is True
    assert set(hub.repos["user/run"]) == {"a.json", "decision_labels/policy/000000_q.pt"}
    # Periodic pushes wait for the interval; stage-end pushes do not.
    (out / "b.json").write_text("2")
    assert sync.push("too soon") is False and "b.json" not in hub.repos["user/run"]
    sync.last -= sync.interval + 1
    assert sync.push("interval elapsed") is True and "b.json" in hub.repos["user/run"]
    # Merging a role removes its per-question files from the mirror as well.
    shutil.rmtree(out / "decision_labels" / "policy")
    (out / "decision_labels" / "policy.pt").write_bytes(b"merged")
    assert sync.push("merged", force=True) is True
    assert set(hub.repos["user/run"]) == {"a.json", "b.json", "decision_labels/policy.pt"}
    # A fresh session pulls the mirror into an empty folder without download bookkeeping,
    # and without the traced Stage 4 rows an earlier revision mirrored (derived, superseded).
    hub.repos["user/run"]["final_trajectories.jsonl"] = b"traced rows"
    fresh = tmp_path / "fresh"
    pulled = a.OutputSync(fresh, "user/run", 1000, api=hub)
    assert pulled.pull() == "pulled"
    assert (fresh / "decision_labels" / "policy.pt").read_bytes() == b"merged"
    assert not (fresh / ".cache").exists() and not (fresh / "partial.pt.tmp").exists()
    assert not (fresh / "final_trajectories.jsonl").exists()
    assert pulled.pull() == "local"
    # Failures never discard local work: periodic pushes warn, required pushes raise.
    monkeypatch.setattr(a.time, "sleep", lambda seconds: None)
    hub.fail_uploads = 3
    capsys.readouterr()
    assert sync.push("flaky", force=True) is False
    assert "WARNING: upload to user/run failed" in capsys.readouterr().out
    hub.fail_uploads = 3
    with pytest.raises(RuntimeError, match="Upload to user/run failed"):
        sync.push("needed", force=True, required=True)
    hub.fail_uploads = 2
    assert sync.push("retried", force=True) is True and sync.last_error is None
    assert set(hub.repos["user/run"]) == {"a.json", "b.json", "decision_labels/policy.pt"}
    assert hub.messages == ["first", "interval elapsed", "merged", "retried"]
    assert "4 uploads" in sync.describe() and "last upload: retried" in sync.describe()


def test_interrupted_teacher_stage_resumes_completed_folds_exactly(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = small_config(tmp_path)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.output_dir = str(tmp_path / "reference")
    reference = a.Workflow(reference_cfg).prepare().train_teacher().teacher
    original = a.train_teacher
    names = []

    def interrupt_second_fold(records, train_ids, val_ids, config, device, seed, name):
        names.append(name)
        if len(names) == 2:
            raise KeyboardInterrupt("simulated session end")
        return original(records, train_ids, val_ids, config, device, seed, name)

    monkeypatch.setattr(a, "train_teacher", interrupt_second_fold)
    with pytest.raises(KeyboardInterrupt, match="simulated session end"):
        a.Workflow(cfg).prepare().train_teacher()
    run = tmp_path / "run"
    assert (run / "teacher_fold_1.pt").exists() and not (run / "teacher_fold_2.pt").exists()
    names.clear()
    monkeypatch.setattr(a, "train_teacher",
                        lambda *args, **kwargs: names.append(args[-1]) or original(*args, **kwargs))
    capsys.readouterr()
    resumed = a.Workflow(cfg).prepare().train_teacher().teacher
    log = capsys.readouterr().out
    assert names == ["Teacher fold 2/2", "Final teacher"]
    assert "Teacher fold 1/2" in log and "loaded completed checkpoint" in log
    assert resumed["folds"] == reference["folds"]
    for key in ("scores", "safe", "maximum"):
        np.testing.assert_array_equal(resumed[key], reference[key])
    assert all(torch.equal(value, resumed["teacher"]["weights"][name])
               for name, value in reference["teacher"]["weights"].items())
    # A fold file whose question roles differ is never reused.
    fold = a.load_checkpoint(run / "teacher_fold_1.pt")
    fold["heldout_ids"] = fold["heldout_ids"][::-1]
    a.atomic_torch(run / "teacher_fold_1.pt", fold)
    (run / "teacher_completed.pt").unlink()
    names.clear()
    a.Workflow(cfg).prepare().train_teacher()
    assert names == ["Teacher fold 1/2"]


def test_pipeline_resumes_from_the_mirror_in_a_fresh_session(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(1)
    hub = FakeHub()
    monkeypatch.setattr(a, "hub_api", lambda: hub)
    cfg = small_config(tmp_path, hf_repo="user/adaptive-run", hf_sync_minutes=1e-9)
    work = a.run_pipeline(cfg)
    remote = hub.repos["user/adaptive-run"]
    assert set(remote) == {
        "contract.json", "compact_records.pt", "data_audit.json", "split_manifest.json",
        "teacher_fold_1.pt", "teacher_fold_2.pt", "teacher_final.pt", "teacher_completed.pt",
        "heuristics_selected_on_dev.json", "snapshot_completed.pt",
        "decision_dataset_manifest.json", "decision_label_coverage.json",
        "decision_labels/policy.pt", "decision_labels/dev.pt",
        "decision_head_progress.pt", "decision_head_completed.pt", "inference_bundle.pt",
        "final_rollouts.jsonl", "results.json"}
    for relative, data in remote.items():
        assert (tmp_path / "run" / relative).read_bytes() == data
    assert hub.messages[0].startswith("Stage 0") and hub.messages[-1].startswith("Stage 4")
    stages = [int(message.split()[1].rstrip(":")) for message in hub.messages]
    assert stages == sorted(stages) and set(stages) == {0, 1, 2, 3, 4}
    assert any("teacher fold" in message for message in hub.messages)
    assert any("labels" in message and "questions" in message for message in hub.messages)
    assert any("labels complete" in message for message in hub.messages)
    assert any("epoch" in message for message in hub.messages)
    # A new session: empty working disk, logs downloaded again, same configuration.
    shutil.rmtree(tmp_path / "run")
    os.utime(cfg.train_file, (1, 1))
    for name in ("fit_teachers", "fit_snapshot", "train_decision_head"):
        monkeypatch.setattr(a, name, lambda *args, _name=name, **kwargs: pytest.fail(
            f"{_name} must resume from the mirror instead of training again"))
    capsys.readouterr()
    resumed = a.run_pipeline(cfg)
    log = capsys.readouterr().out
    assert "Mirror user/adaptive-run: downloaded" in log and "completed stages resume" in log
    assert "Loaded completed teacher stage." in log
    assert "Loaded completed snapshot stage." in log
    assert "Loaded completed supervised decision head." in log
    assert resumed.results["decision_training"] == work.results["decision_training"]
    assert resumed.results["audit"]["adaptive"]["policies"]["supervised"]["n"] == \
        work.results["audit"]["adaptive"]["policies"]["supervised"]["n"]
    assert hub.messages[-1].startswith("Stage 4") and resumed.sync.describe().startswith("Mirror user/adaptive-run")
    # A bad token or repository fails at Stage 0, before any training.
    shutil.rmtree(tmp_path / "run")
    monkeypatch.setattr(a.time, "sleep", lambda seconds: None)
    hub.fail_uploads = 3
    with pytest.raises(RuntimeError, match="Upload to user/adaptive-run failed"):
        a.Workflow(cfg).prepare()


def move_logs(cfg, folder):
    """A new session downloads the same logs into another directory."""
    folder.mkdir(parents=True)
    paths = []
    for path in [cfg.train_file] + cfg.test_files:
        target = folder / Path(path).name
        shutil.move(path, target)
        paths.append(str(target))
    moved = copy.deepcopy(cfg)
    moved.train_file, moved.test_files = paths[0], paths[1:]
    return moved


def forbid(monkeypatch, *names):
    for name in names:
        monkeypatch.setattr(a, name, lambda *args, _name=name, **kwargs: pytest.fail(
            f"{_name} must resume from saved files instead of running again"))


def stage3_files(run):
    """Every saved Stage 3 fingerprint: manifest, labels, progress, and completed head."""
    found = {"manifest": json.loads((run / "decision_dataset_manifest.json").read_text(encoding="utf-8"))["fingerprint"]}
    for path in sorted(run.glob("decision_labels/**/*.pt")) + [run / "decision_head_progress.pt"]:
        if path.exists():
            found[path.relative_to(run).as_posix()] = a.load_checkpoint(path)["fingerprint"]
    for name in ("decision_head_completed.pt", "inference_bundle.pt"):
        if (run / name).exists():
            checkpoint = a.load_checkpoint(run / name)
            found[name] = checkpoint.get("decision_head", checkpoint)["dataset_fingerprint"]
    return found


def bind_to_log_paths(run, current, legacy):
    """Stage 3 files as earlier code wrote them: identical, but fingerprinted with the log paths."""
    assert set(stage3_files(run).values()) == {current}
    manifest = run / "decision_dataset_manifest.json"
    a.atomic_json(manifest, dict(json.loads(manifest.read_text(encoding="utf-8")), fingerprint=legacy))
    for relative in stage3_files(run):
        if relative == "manifest":
            continue
        checkpoint = a.load_checkpoint(run / relative)
        target = checkpoint.get("decision_head", checkpoint)
        target["dataset_fingerprint" if "dataset_fingerprint" in target else "fingerprint"] = legacy
        a.atomic_torch(run / relative, checkpoint)
    return [relative for relative in stage3_files(run) if relative not in ("manifest", "inference_bundle.pt")]


def assert_same_head(actual, reference):
    assert actual["history"] == reference["history"] and actual["best_epoch"] == reference["best_epoch"]
    assert actual["coverage"] == reference["coverage"]
    assert all(torch.equal(value, actual["weights"][key]) for key, value in reference["weights"].items())


def test_moved_input_logs_resume_stages_0_to_3_without_relabeling_or_retraining(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = small_config(tmp_path)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.output_dir = str(tmp_path / "reference")
    reference = a.Workflow(reference_cfg).train_decision_head().decision_checkpoint
    run = tmp_path / "run"
    original = a.masked_action_loss

    def stop_in_second_epoch(model, rows, order, config, device, optimizer=None, **kwargs):
        if optimizer is not None and (run / "decision_head_progress.pt").exists():
            raise KeyboardInterrupt("simulated session end")
        return original(model, rows, order, config, device, optimizer, **kwargs)

    monkeypatch.setattr(a, "masked_action_loss", stop_in_second_epoch)
    with pytest.raises(KeyboardInterrupt, match="simulated session end"):
        a.Workflow(cfg).train_decision_head()
    assert a.load_checkpoint(run / "decision_head_progress.pt")["epoch"] == 1
    monkeypatch.setattr(a, "masked_action_loss", original)
    # The next session downloads the same logs elsewhere; the old directory is gone.
    moved = move_logs(cfg, tmp_path / "session2")
    assert not Path(cfg.train_file).exists()
    forbid(monkeypatch, "fit_teachers", "fit_snapshot", "build_decision_labels")
    capsys.readouterr()
    resumed = a.Workflow(moved).train_decision_head().decision_checkpoint
    log = capsys.readouterr().out
    assert "Using compact cached records" in log and "Loaded completed teacher stage." in log
    assert "Loaded completed snapshot stage." in log and "built 0, reused" in log
    assert "Stage 3 head resume: 1 completed epochs" in log and "Stage 3 fingerprint" not in log
    assert_same_head(resumed, reference)
    # The fingerprint no longer depends on the directory: the reference used the first one.
    assert set(stage3_files(run).values()) == {reference["dataset_fingerprint"]}
    # A third directory loads the completed head.
    again = move_logs(moved, tmp_path / "session3")
    forbid(monkeypatch, "train_decision_head")
    a.Workflow(again).train_decision_head()
    assert "Loaded completed supervised decision head." in capsys.readouterr().out


@pytest.mark.parametrize("moved", [False, True])
def test_path_bound_labels_from_earlier_code_upgrade_in_place(tmp_path, monkeypatch, capsys, moved):
    torch.set_num_threads(1)
    cfg = small_config(tmp_path)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.output_dir = str(tmp_path / "reference")
    reference = a.Workflow(reference_cfg).train_decision_head().decision_checkpoint
    run = tmp_path / "run"
    original, built = a.build_decision_labels, []

    def stop_on_third_question(record, *args, **kwargs):
        built.append(record.uid)
        if len(built) == 3:
            raise KeyboardInterrupt("simulated session end")
        return original(record, *args, **kwargs)

    monkeypatch.setattr(a, "build_decision_labels", stop_on_third_question)
    work = a.Workflow(cfg)
    with pytest.raises(KeyboardInterrupt, match="simulated session end"):
        work.train_decision_head()
    current = reference["dataset_fingerprint"]
    legacy = a.decision_dataset_fingerprint(work.records, work.splits, work.teacher, work.predictor, cfg,
                                            [cfg.train_file] + cfg.test_files)
    assert bind_to_log_paths(run, current, legacy) == [
        path.relative_to(run).as_posix() for path in a.question_paths(
            work.records, work.splits["policy"][:2], run / "decision_labels" / "policy")]
    if moved:
        # Only contract.json still names the directory the labels were built with.
        cfg = move_logs(cfg, tmp_path / "session2")
    built.clear()
    monkeypatch.setattr(a, "build_decision_labels",
                        lambda record, *args, **kwargs: built.append(record.uid) or original(record, *args, **kwargs))
    forbid(monkeypatch, "fit_teachers", "fit_snapshot")
    capsys.readouterr()
    resumed = a.Workflow(cfg).train_decision_head().decision_checkpoint
    log = capsys.readouterr().out
    assert "the manifest and 2 label, progress, or head files hashed the input log paths" in log
    assert "built 0, reused 1" in log and built == [
        work.records[i].uid for i in work.splits["policy"][2:] + work.splits["dev"]]
    assert_same_head(resumed, reference)
    assert set(stage3_files(run).values()) == {current}
    capsys.readouterr()
    a.Workflow(cfg).train_decision_head()
    log = capsys.readouterr().out
    assert "Loaded completed supervised decision head." in log and "Stage 3 fingerprint" not in log


def test_completed_head_from_earlier_code_upgrades_with_the_paths_its_bundle_recorded(tmp_path, monkeypatch, capsys):
    torch.set_num_threads(1)
    monkeypatch.chdir(tmp_path)
    # Relative paths: earlier code hashed these strings, while contract.json records resolved ones.
    cfg = move_logs(small_config(tmp_path), Path("logs"))
    work = a.Workflow(cfg).train_decision_head()
    current = work.decision_checkpoint["dataset_fingerprint"]
    legacy = a.decision_dataset_fingerprint(work.records, work.splits, work.teacher, work.predictor, cfg,
                                            [cfg.train_file] + cfg.test_files)
    run = tmp_path / "run"
    rebound = bind_to_log_paths(run, current, legacy)
    assert rebound == ["decision_labels/dev.pt", "decision_labels/policy.pt",
                       "decision_head_progress.pt", "decision_head_completed.pt"]
    moved = move_logs(cfg, Path("downloads"))
    forbid(monkeypatch, "fit_teachers", "fit_snapshot", "train_decision_head")
    # Neither the new paths nor the contract's resolved paths reproduce the old fingerprint,
    # so without the bundle's configured strings the files are rejected and left as they are.
    bundle = run / "inference_bundle.pt"
    hidden = bundle.rename(run / "bundle.hidden")
    saved = {relative: (run / relative).read_bytes() for relative in rebound + ["decision_dataset_manifest.json"]}
    capsys.readouterr()
    with pytest.raises(ValueError, match="Completed decision head has an incompatible"):
        a.Workflow(moved).train_decision_head()
    assert "NOTE: Stage 3 files match neither this run nor its path-bound fingerprint" in capsys.readouterr().out
    assert all((run / relative).read_bytes() == data for relative, data in saved.items())
    hidden.rename(bundle)
    resumed = a.Workflow(moved).train_decision_head()
    log = capsys.readouterr().out
    assert "the manifest and 4 label, progress, or head files hashed the input log paths" in log
    assert "Loaded completed supervised decision head." in log
    assert set(stage3_files(run).values()) == {current}
    assert_same_head(resumed.decision_checkpoint, work.decision_checkpoint)
    a.Workflow(moved).train_decision_head()
    log = capsys.readouterr().out
    assert "Loaded completed supervised decision head." in log and "Stage 3 fingerprint" not in log
