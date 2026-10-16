"""Offline checks of the versioned dataset store: the dataset phase reads the input logs into compact
records without any model, uploads them once under a dataset id and version, resumes per log, and is
downloaded instead of rebuilt; everything the models make stays in the run mirror."""
import copy
import fnmatch
import json
import shutil
from pathlib import Path

import pytest
import torch

import adaptive_analogical_training as a
from test_adaptive_session_resume import FakeHub, forbid, interrupt_after_save, small_config

DATASET_FILES = {"compact_records.pt", "dataset_manifest.json"}
TRAINERS = ("fit_teachers", "fit_snapshot", "train_on_views", "fit_action_head", "build_route_labels",
            "build_finetune_labels", "finetune_labels")
RUN_OUTPUT = {"contract.json", "data_audit.json", "split_manifest.json", "heuristics_selected_on_dev.json",
              "teacher_fold_1.pt", "teacher_fold_2.pt", "teacher_final.pt", "teacher_completed.pt",
              "snapshot_completed.pt", "decision_dataset_manifest.json", "decision_label_coverage.json",
              "decision_labels/pretrain_policy.pt", "decision_labels/pretrain_dev.pt",
              "decision_labels/finetune_policy.pt", "decision_labels/finetune_dev.pt",
              "decision_head_pretrain_progress.pt", "decision_head_pretrained.pt",
              "decision_head_finetune_progress.pt", "decision_head_completed.pt", "inference_bundle.pt",
              "final_rollouts.jsonl", "results.json"}


def dataset_config(root, **overrides):
    return small_config(root, dataset_id="adaptive-v9", dataset_repo="user/datasets", **overrides)


@pytest.fixture
def hub(monkeypatch):
    fake = FakeHub()
    monkeypatch.setattr(a, "hub_api", lambda: fake)
    monkeypatch.setattr(a.time, "sleep", lambda seconds: None)
    return fake


def stored(hub, version, repo="user/datasets", dataset_id="adaptive-v9"):
    prefix = f"{dataset_id}/{version}/"
    return {name[len(prefix):]: data for name, data in hub.repos[repo].items() if name.startswith(prefix)}


def remove_logs(cfg):
    for path in [cfg.train_file] + cfg.test_files:
        Path(path).unlink()


def same_records(left, right):
    return a.data_digest(left.records) == a.data_digest(right.records) and left.audit == right.audit


def test_dataset_settings_are_runtime_only_and_validated(tmp_path, hub):
    torch.set_num_threads(1)
    with pytest.raises(ValueError, match="go together"):
        a.Config(dataset_id="adaptive-v9").validate()
    with pytest.raises(ValueError, match="go together"):
        a.Config(dataset_repo="user/datasets").validate()
    for bad in ("", "-leading", "has space", "a/b", "..", "x" * 65):
        with pytest.raises(ValueError, match="dataset_id"):
            a.Config(dataset_id=bad, dataset_repo="user/datasets").validate()
    with pytest.raises(ValueError, match="dataset_repo"):
        a.Config(dataset_id="adaptive-v9", dataset_repo="not a repository").validate()
    a.Config(dataset_id="adaptive-v9.1_a", dataset_repo="user/datasets").validate()
    with pytest.raises(ValueError, match="must differ from hf_repo"):
        a.Config(dataset_id="adaptive-v9", dataset_repo="user/run", hf_repo="user/run").validate()
    # The id and repository name where a dataset lives; they never change the run contract or labels.
    plain = small_config(tmp_path)
    with_id = dataset_config(tmp_path)
    assert {"dataset_id", "dataset_repo"} <= set(a.NON_CONTRACT_SETTINGS)
    a.Workflow(plain)
    plain_contract = json.loads((tmp_path / "run" / "contract.json").read_text(encoding="utf-8"))
    assert not {"dataset_id", "dataset_repo"} & set(plain_contract["config"])
    # A run started before the dataset store existed keeps resuming once an id is added.
    assert a.Workflow(with_id).datasets.state == "new"
    assert json.loads((tmp_path / "run" / "contract.json").read_text(encoding="utf-8")) == plain_contract


def test_the_dataset_phase_trains_no_model_and_the_dataset_holds_only_the_records(tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path, hf_repo="user/run", hf_sync_minutes=1e-9)
    # No network may even be constructed, let alone trained, loaded, or run.
    construct = torch.nn.Module.__init__

    def no_network(self, *args, **kwargs):
        pytest.fail(f"The dataset phase constructed a {type(self).__name__}.")

    trainers = {name: getattr(a, name) for name in TRAINERS}
    monkeypatch.setattr(torch.nn.Module, "__init__", no_network)
    forbid(monkeypatch, *TRAINERS)
    work = a.Workflow(cfg).build_datasets()
    monkeypatch.setattr(torch.nn.Module, "__init__", construct)
    for name, function in trainers.items():
        monkeypatch.setattr(a, name, function)
    log = capsys.readouterr().out
    assert "trains, loads, or runs no model" in log and "complete in user/datasets" in log
    assert work.records and not hasattr(work, "teacher") and work.splits is None
    version = work.datasets.version
    contract = json.loads((tmp_path / "run" / "contract.json").read_text(encoding="utf-8"))
    identity = [{key: source[key] for key in ("name", "bytes", "sha256")} for source in contract["sources"]]
    # Only the record settings and the logs decide the version, not the training settings or revision.
    assert version == a.dataset_version(a.dataset_construction({"k": 2, "zero_shots": 2, "repeats": 5}, identity))
    remote = stored(hub, version)
    assert set(remote) == DATASET_FILES and set(hub.repos["user/datasets"]) == {
        f"adaptive-v9/{version}/{name}" for name in DATASET_FILES}
    manifest = json.loads(remote["dataset_manifest.json"])
    assert manifest["complete"] is True and manifest["schema_version"] == a.DATASET_SCHEMA == 2
    assert manifest["construction"]["config"] == {"k": 2., "zero_shots": 2., "repeats": 5.}
    assert manifest["construction"]["sources"] == identity and "implementation_revision" not in manifest["construction"]
    assert manifest["artifacts"] == {"compact_records.pt": {
        "bytes": len(remote["compact_records.pt"]), "sha256": a.file_digest(tmp_path / "run" / "compact_records.pt")}}
    assert manifest["info"] == {"records": len(work.records), "audit": work.audit}
    assert remote["compact_records.pt"] == (tmp_path / "run" / "compact_records.pt").read_bytes()
    assert not (tmp_path / "run" / a.DATASET_PROGRESS_DIR).exists()
    # The build uploaded before reading, after each log, and when sealed; no per-log file is left.
    # (Each message is logged once per repository it went to.)
    assert list(dict.fromkeys(message for message in hub.messages if message.startswith("Dataset")))[:4] == [
        "Dataset: build started", "Dataset: records of log 1/2", "Dataset: records of log 2/2",
        f"Dataset adaptive-v9 version {version}: complete"]
    assert set(hub.repos["user/run"]) == {"contract.json"}
    # Training makes every model-dependent file; the run mirror keeps them, the sealed dataset is untouched.
    uploaded = dict(hub.repos["user/datasets"])
    work.prepare().train_decision_head().report()
    assert set(hub.repos["user/run"]) == RUN_OUTPUT
    assert hub.repos["user/datasets"] == uploaded
    assert work.sync.describe().startswith("Mirror user/run")


def test_a_fresh_session_downloads_the_dataset_without_the_logs_and_never_reads_them(
        tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    built = a.Workflow(cfg).build_datasets()
    uploaded = dict(hub.repos["user/datasets"])
    # A new Kaggle session: empty working disk, and no input logs attached at all.
    fresh = copy.deepcopy(cfg)
    fresh.output_dir = str(tmp_path / "session2")
    remove_logs(cfg)
    forbid(monkeypatch, "read_log")
    capsys.readouterr()
    resumed = a.Workflow(fresh)
    assert resumed.datasets.version == built.datasets.version and resumed.datasets.state == "complete"
    resumed.build_datasets().prepare()
    log = capsys.readouterr().out
    assert "complete in user/datasets" in log and "Nothing is rebuilt" in log
    assert "Using compact cached records" in log and same_records(resumed, built)
    assert hub.repos["user/datasets"] == uploaded  # Nothing new was uploaded.
    contract = json.loads((tmp_path / "session2" / "contract.json").read_text(encoding="utf-8"))
    assert [source["name"] for source in contract["sources"]] == [
        Path(cfg.train_file).name, Path(cfg.test_files[0]).name]
    # Training needs only the downloaded records.
    resumed.train_teacher()
    assert hub.repos["user/datasets"] == uploaded


def test_training_settings_share_a_version_and_record_settings_or_logs_make_another(tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    first = a.Workflow(cfg).build_datasets()
    before = dict(stored(hub, first.datasets.version))
    # Training settings change the models, never the records: the stored dataset is downloaded.
    trained = copy.deepcopy(cfg)
    trained.output_dir = str(tmp_path / "trained")
    trained.teacher_epochs, trained.learning_rate, trained.decision_epochs, trained.seed = 3, .01, 5, 7
    forbid(monkeypatch, "read_log")
    second = a.Workflow(trained)
    assert second.datasets.version == first.datasets.version and second.datasets.state == "complete"
    assert same_records(second.build_datasets(), first)
    # A setting the records depend on is another version beside the first, which stays untouched.
    changed = copy.deepcopy(cfg)
    changed.output_dir = str(tmp_path / "changed")
    changed.repeats = 1
    capsys.readouterr()
    third = a.Workflow(changed)
    assert third.datasets.version != first.datasets.version and third.datasets.state == "new"
    assert f"{first.datasets.version} differs in config.repeats" in capsys.readouterr().out
    assert stored(hub, first.datasets.version) == before
    # Changed logs are another version too, and the first dataset still resolves when they are restored.
    train = Path(cfg.train_file)
    original = train.read_bytes()
    tampered = bytearray(original)
    tampered[1] ^= 1
    train.write_bytes(bytes(tampered))
    other = copy.deepcopy(cfg)
    other.output_dir = str(tmp_path / "other")
    assert a.Workflow(other).datasets.version not in {first.datasets.version, third.datasets.version}
    train.write_bytes(original)
    restored = copy.deepcopy(cfg)
    restored.output_dir = str(tmp_path / "restored")
    assert a.Workflow(restored).datasets.version == first.datasets.version


def test_an_interrupted_build_resumes_with_the_logs_it_had_not_finished(tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.output_dir, reference_cfg.dataset_id = str(tmp_path / "reference"), "reference"
    reference = a.Workflow(reference_cfg).build_datasets()
    read_log = a.read_log

    def end_in_second_log(path, *args, **kwargs):
        if Path(path).name == "external.json":
            raise KeyboardInterrupt("simulated session end")
        return read_log(path, *args, **kwargs)

    monkeypatch.setattr(a, "read_log", end_in_second_log)
    with pytest.raises(KeyboardInterrupt, match="simulated session end"):
        a.Workflow(cfg).build_datasets()
    (version,) = {name.split("/")[1] for name in hub.repos["user/datasets"] if name.startswith("adaptive-v9/")}
    partial = stored(hub, version)
    assert set(partial) == {"dataset_manifest.json", f"{a.DATASET_PROGRESS_DIR}/01_train.pt"}
    assert json.loads(partial["dataset_manifest.json"])["complete"] is False
    # The next session starts empty, downloads the first log's records, and reads only the second log.
    shutil.rmtree(tmp_path / "run")
    reads = []
    monkeypatch.setattr(a, "read_log", lambda path, *args, **kwargs: reads.append(Path(path).name) or
                        read_log(path, *args, **kwargs))
    capsys.readouterr()
    resumed = a.Workflow(cfg)
    assert resumed.datasets.state == "partial"
    resumed.build_datasets()
    log = capsys.readouterr().out
    assert "partly built in user/datasets" in log and "01_train.pt was read in an earlier session" in log
    assert reads == ["external.json"] and same_records(resumed, reference)
    complete = stored(hub, version)
    assert set(complete) == DATASET_FILES and json.loads(complete["dataset_manifest.json"])["complete"] is True
    assert not (tmp_path / "run" / a.DATASET_PROGRESS_DIR).exists()


def test_a_saved_log_is_reused_only_for_the_same_logs_and_record_settings(tmp_path, monkeypatch, capsys):
    cfg = small_config(tmp_path)
    sources = a.input_sources([cfg.train_file] + cfg.test_files)
    out = tmp_path / "out"
    expected = a.load_records(cfg)
    records, audit = a.load_records(cfg, a.RecordProgress(out, cfg, sources))
    assert a.data_digest(records) == a.data_digest(expected[0]) and audit == expected[1]
    assert sorted(path.name for path in (out / a.DATASET_PROGRESS_DIR).iterdir()) == ["01_train.pt", "02_external.pt"]
    # Saved logs are loaded, not read; equal numbers are equal settings.
    read_log = a.read_log
    forbid(monkeypatch, "read_log")
    same = copy.deepcopy(cfg)
    same.repeats = 5.
    again = a.load_records(same, a.RecordProgress(out, same, sources))
    assert a.data_digest(again[0]) == a.data_digest(records) and again[1] == audit
    # Another first log changes what the second keeps, so both are read again and replaced.
    monkeypatch.setattr(a, "read_log", read_log)
    other = [dict(sources[0], sha256="0" * 64), sources[1]]
    capsys.readouterr()
    a.load_records(cfg, a.RecordProgress(out, cfg, other))
    log = capsys.readouterr().out
    assert log.count("holds the records of other logs or settings") == 2


def test_a_session_ended_inside_model_training_continues_from_the_run_mirror(tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path, hf_repo="user/run")
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.output_dir = str(tmp_path / "reference")
    reference_cfg.dataset_id = reference_cfg.dataset_repo = reference_cfg.hf_repo = None
    a.Workflow(reference_cfg).build_datasets().prepare().train_snapshot()
    # The session ends inside Stage 2, a few batches after its first saved check.
    teacher_saves = (cfg.teacher_folds + 1) * cfg.teacher_epochs * cfg.dev_checks_per_epoch
    restore = interrupt_after_save(monkeypatch, teacher_saves + 1, 3)
    with pytest.raises(KeyboardInterrupt, match="simulated session end"):
        a.Workflow(cfg).build_datasets().prepare().train_snapshot()
    restore()
    mirror = hub.repos["user/run"]
    assert "teacher_completed.pt" in mirror and "snapshot_completed.pt" not in mirror
    assert "training_progress/snapshot.pt" in mirror
    assert not any(name.startswith("training_progress/teacher") for name in mirror)
    # The training state is model output: it never enters the dataset.
    (version,) = {name.split("/")[1] for name in hub.repos["user/datasets"]}
    assert set(stored(hub, version)) == DATASET_FILES
    # Kaggle restarted: an empty working folder, and the notebook runs from its first cell.
    shutil.rmtree(tmp_path / "run")
    forbid(monkeypatch, "fit_teachers", "read_log")
    capsys.readouterr()
    a.Workflow(cfg).build_datasets().prepare().train_snapshot()
    log = capsys.readouterr().out
    assert "Loaded completed teacher stage." in log and "Snapshot: resumed at epoch 0.500 from snapshot.pt" in log
    assert "snapshot_completed.pt" in mirror and not any(name.startswith(a.PROGRESS_DIR) for name in mirror)
    assert not (tmp_path / "run" / a.PROGRESS_DIR).exists()
    built, expected = (a.load_checkpoint(root / "snapshot_completed.pt") for root in (tmp_path / "run",
                                                                                      tmp_path / "reference"))
    assert built["history"] == expected["history"]
    assert all(torch.equal(value, built["weights"][key]) for key, value in expected["weights"].items())


def test_model_checkpoints_that_a_schema_1_dataset_kept_continue_in_the_run(tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path, hf_repo="user/run")
    # Schema 1 kept a build's model output in the dataset, under a hash of the whole run contract.
    legacy_cfg = copy.deepcopy(cfg)
    legacy_cfg.output_dir = str(tmp_path / "legacy")
    legacy_cfg.dataset_id = legacy_cfg.dataset_repo = legacy_cfg.hf_repo = None
    a.Workflow(legacy_cfg).build_datasets().prepare().train_pretrained_head()
    folder = tmp_path / "legacy"
    contract = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
    identity = [{key: source[key] for key in ("name", "bytes", "sha256")} for source in contract["sources"]]
    legacy = a.legacy_dataset_version(contract["config"], identity)
    kept = {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    kept = {relative: data for relative, data in kept.items() if relative == "compact_records.pt" or any(
        fnmatch.fnmatch(relative, pattern) for pattern in a.LEGACY_TRAINING_PATTERNS)}
    construction = json.loads(json.dumps(a.canonical_settings({
        "schema_version": 1, "implementation_revision": a.PIPELINE_REVISION,
        "config": contract["config"], "sources": identity})))
    manifest = {"schema_version": 1, "dataset_id": "adaptive-v9", "version": legacy,
                "construction": construction, "complete": True, "artifacts": {}, "info": {}}
    kept["dataset_manifest.json"] = json.dumps(manifest).encode()
    hub.repos["user/datasets"] = {f"adaptive-v9/{legacy}/{relative}": data for relative, data in kept.items()}
    before = dict(hub.repos["user/datasets"])
    forbid(monkeypatch, "fit_teachers", "fit_snapshot", "fit_action_head", "build_route_labels")
    capsys.readouterr()
    work = a.Workflow(cfg)
    log = capsys.readouterr().out
    assert "was built by dataset-store schema 1" in log and f"{legacy} differs in schema_version" in log
    assert work.datasets.version != legacy and work.datasets.state == "new"
    work.build_datasets().prepare().train_pretrained_head()
    log = capsys.readouterr().out
    assert "Loaded completed teacher stage." in log and "Loaded completed snapshot stage." in log
    assert "Loaded completed pretrained decision head." in log
    # The schema 1 version stays as it was, the new one holds the records, the run mirror the checkpoints.
    assert {name: data for name, data in hub.repos["user/datasets"].items()
            if name.startswith(f"adaptive-v9/{legacy}/")} == before
    assert set(stored(hub, work.datasets.version)) == DATASET_FILES
    assert {"teacher_completed.pt", "snapshot_completed.pt", "decision_head_pretrained.pt",
            "decision_labels/pretrain_policy.pt"} <= set(hub.repos["user/run"])
    # Once the run holds checkpoints of its own (here, or pulled from its mirror), they are newer: nothing is adopted.
    (tmp_path / "run" / "teacher_fold_1.pt").unlink()
    capsys.readouterr()
    a.Workflow(cfg)
    assert "schema 1" not in capsys.readouterr().out and not (tmp_path / "run" / "teacher_fold_1.pt").exists()
    shutil.rmtree(tmp_path / "run")
    capsys.readouterr()
    a.Workflow(cfg)
    assert "schema 1" not in capsys.readouterr().out and (tmp_path / "run" / "teacher_fold_1.pt").exists()


def test_records_built_before_the_dataset_store_are_uploaded_as_the_dataset_not_read_again(
        tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    legacy = small_config(tmp_path, hf_repo="user/run", hf_sync_minutes=1e-9)
    a.Workflow(legacy).build_datasets().prepare()
    assert "compact_records.pt" in hub.repos["user/run"]
    # The user now selects a dataset id. A new session pulls the run mirror, whose records are then adopted.
    shutil.rmtree(tmp_path / "run")
    adopting = copy.deepcopy(legacy)
    adopting.dataset_id, adopting.dataset_repo = "adaptive-v9", "user/datasets"
    forbid(monkeypatch, "read_log")
    capsys.readouterr()
    work = a.Workflow(adopting)
    assert work.datasets.state == "adopted" and "not rebuilt" in capsys.readouterr().out
    work.build_datasets()
    remote = stored(hub, work.datasets.version)
    assert set(remote) == DATASET_FILES and json.loads(remote["dataset_manifest.json"])["complete"] is True
    assert remote["compact_records.pt"] == (tmp_path / "run" / "compact_records.pt").read_bytes()


def test_an_uploaded_dataset_is_never_overwritten_or_silently_replaced(tmp_path, hub):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    a.Workflow(cfg).build_datasets()
    # resume=False would rebuild and overwrite the dataset that is already stored.
    fresh = copy.deepcopy(cfg)
    fresh.output_dir, fresh.resume = str(tmp_path / "fresh"), False
    with pytest.raises(ValueError, match="already in user/datasets"):
        a.Workflow(fresh)
    # A folder holding another dataset id, or a rebuilt artifact, is refused rather than mixed in.
    renamed = copy.deepcopy(cfg)
    renamed.dataset_id = "other-id"
    with pytest.raises(ValueError, match="holds dataset 'adaptive-v9'"):
        a.Workflow(renamed)
    run = tmp_path / "run"
    original = (run / "compact_records.pt").read_bytes()
    (run / "compact_records.pt").write_bytes(original + b"x")
    with pytest.raises(ValueError, match="compact_records.pt differs from dataset 'adaptive-v9'"):
        a.Workflow(cfg)
    (run / "compact_records.pt").write_bytes(original)
    assert a.Workflow(cfg).datasets.state == "complete"
    # A damaged download is detected the same way.
    version = a.Workflow(cfg).datasets.version
    hub.repos["user/datasets"][f"adaptive-v9/{version}/compact_records.pt"] += b"damaged"
    damaged = copy.deepcopy(cfg)
    damaged.output_dir = str(tmp_path / "damaged")
    with pytest.raises(ValueError, match="compact_records.pt differs from dataset"):
        a.Workflow(damaged)


def test_a_hub_outage_never_leads_to_a_rebuild_and_missing_logs_need_one_matching_version(
        tmp_path, hub, monkeypatch):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    forbid(monkeypatch, "read_log")
    # Not knowing whether the Hub holds the dataset is an error, not an empty Hub.
    hub.fail_reads = 9
    with pytest.raises(RuntimeError, match="Nothing is rebuilt while it is unknown"):
        a.Workflow(cfg)
    hub.fail_reads = 0
    # Without the logs a version cannot be identified unless exactly one complete version fits.
    detached = copy.deepcopy(cfg)
    detached.output_dir = str(tmp_path / "detached")
    remove_logs(cfg)
    with pytest.raises(ValueError, match="not available, and user/datasets holds 0 complete versions"):
        a.Workflow(detached)


def test_without_the_logs_the_one_complete_version_with_these_log_names_is_used(tmp_path, hub):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    built = a.Workflow(cfg).build_datasets()
    remove_logs(cfg)
    detached = copy.deepcopy(cfg)
    detached.output_dir = str(tmp_path / "detached")
    assert a.Workflow(detached).datasets.version == built.datasets.version
    # Other log names are not the same dataset, whatever the configuration says.
    renamed = copy.deepcopy(detached)
    renamed.output_dir = str(tmp_path / "renamed")
    renamed.test_files = [str(Path(renamed.test_files[0]).with_name("elsewhere.json"))]
    with pytest.raises(ValueError, match="not available"):
        a.Workflow(renamed)


def test_equal_settings_share_a_version_and_a_complete_dataset_is_read_only_here(tmp_path, hub):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    built = a.Workflow(cfg).build_datasets()
    same = copy.deepcopy(cfg)
    same.output_dir, same.repeats = str(tmp_path / "same"), 5.
    assert a.Workflow(same).datasets.version == built.datasets.version
    # A complete version is never written to by a session that only loaded it, even by a differing local file.
    before = dict(hub.repos["user/datasets"])
    session = copy.deepcopy(cfg)
    session.output_dir = str(tmp_path / "reader")
    reader = a.Workflow(session)
    (tmp_path / "reader" / a.DATASET_PROGRESS_DIR).mkdir()
    (tmp_path / "reader" / a.DATASET_PROGRESS_DIR / "01_train.pt").write_bytes(b"leftover")
    reader.build_datasets().prepare()
    reader.push("anything")
    assert hub.repos["user/datasets"] == before and "read-only" in reader.datasets.describe()
