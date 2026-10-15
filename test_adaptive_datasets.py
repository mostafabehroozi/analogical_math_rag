"""Offline checks of the versioned dataset store: what the three training phases build is uploaded
once under a dataset id and version, downloaded instead of rebuilt, and never mixed across versions."""
import copy
import json
import shutil
from pathlib import Path

import pytest
import torch

import adaptive_analogical_training as a
from test_adaptive_session_resume import FakeHub, assert_same_head, forbid, interrupt_after_save, small_config

DATASET_FILES = {
    "compact_records.pt",
    "teacher_fold_1.pt", "teacher_fold_2.pt", "teacher_final.pt", "teacher_completed.pt",
    "snapshot_completed.pt", "decision_head_pretrained.pt",
    "decision_dataset_manifest.json", "decision_label_coverage.json",
    "decision_labels/pretrain_policy.pt", "decision_labels/pretrain_dev.pt",
    "decision_labels/finetune_policy.pt", "decision_labels/finetune_dev.pt",
    "decision_head_pretrain_progress.pt", "dataset_manifest.json"}
BUILDERS = ("fit_teachers", "fit_snapshot", "fit_action_head", "build_route_labels", "build_finetune_labels")


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


def test_build_uploads_the_dataset_under_its_id_and_version_and_keeps_training_output_in_the_run_mirror(
        tmp_path, hub, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path, hf_repo="user/run", hf_sync_minutes=1e-9)
    work = a.Workflow(cfg).prepare().build_datasets()
    version = work.datasets.version
    contract = json.loads((tmp_path / "run" / "contract.json").read_text(encoding="utf-8"))
    identity = [{key: source[key] for key in ("name", "bytes", "sha256")} for source in contract["sources"]]
    assert version == a.dataset_version(a.dataset_construction(contract["config"], identity))
    remote = stored(hub, version)
    assert set(remote) == DATASET_FILES and set(hub.repos["user/datasets"]) == {
        f"adaptive-v9/{version}/{name}" for name in DATASET_FILES}
    for relative, data in remote.items():
        assert (tmp_path / "run" / relative).read_bytes() == data
    manifest = json.loads(remote["dataset_manifest.json"])
    assert manifest["complete"] is True and manifest["dataset_id"] == "adaptive-v9"
    assert manifest["version"] == version and manifest["implementation_revision"] == a.PIPELINE_REVISION
    assert manifest["construction"]["sources"] == identity
    assert set(manifest["artifacts"]) == set(a.dataset_artifacts())
    for relative, meta in manifest["artifacts"].items():
        assert meta == {"bytes": len(remote[relative]), "sha256": a.file_digest(tmp_path / "run" / relative)}
    assert manifest["info"]["records"] == len(work.records)
    assert manifest["info"]["questions"]["dev"] == len(work.splits["dev"])
    # The run mirror carries only what the dataset does not: the contract and training output.
    assert set(hub.repos["user/run"]) == {"contract.json", "data_audit.json", "split_manifest.json",
                                          "heuristics_selected_on_dev.json"}
    log = capsys.readouterr().out
    assert "complete in user/datasets" in log and "The labels come from trained models" in log
    work.train_decision_head().report()
    assert set(hub.repos["user/run"]) == {
        "contract.json", "data_audit.json", "split_manifest.json",
        "heuristics_selected_on_dev.json", "decision_head_finetune_progress.pt", "decision_head_completed.pt",
        "inference_bundle.pt", "final_rollouts.jsonl", "results.json"}
    assert stored(hub, version) == remote
    assert "Dataset adaptive-v9 version" in work.sync.describe() and "Mirror user/run" in work.sync.describe()


def test_a_fresh_session_downloads_the_dataset_without_the_logs_and_never_rebuilds_it(
        tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    built = a.Workflow(cfg).prepare().build_datasets().train_decision_head()
    version = built.datasets.version
    uploaded = dict(hub.repos["user/datasets"])
    # A new Kaggle session: empty working disk, and no input logs attached at all.
    fresh = copy.deepcopy(cfg)
    fresh.output_dir = str(tmp_path / "session2")
    remove_logs(cfg)
    # Only fine-tuning may train: it reads the downloaded labels and heads.
    forbid(monkeypatch, *(name for name in BUILDERS if name != "fit_action_head"))
    capsys.readouterr()
    resumed = a.Workflow(fresh)
    assert resumed.datasets.version == version and resumed.datasets.state == "complete"
    resumed.prepare().build_datasets()
    log = capsys.readouterr().out
    assert "complete in user/datasets" in log and "Nothing is rebuilt" in log and "nothing trains" in log
    assert "Using compact cached records" in log and "Loaded completed teacher stage." in log
    assert "Loaded completed snapshot stage." in log and "Loaded completed pretrained decision head." in log
    resumed.train_decision_head()
    assert_same_head(resumed.decision_checkpoint, built.decision_checkpoint)
    assert resumed.pretrained_checkpoint["weights"].keys() == built.pretrained_checkpoint["weights"].keys()
    assert hub.repos["user/datasets"] == uploaded  # Nothing new was uploaded.
    contract = json.loads((tmp_path / "session2" / "contract.json").read_text(encoding="utf-8"))
    assert [source["name"] for source in contract["sources"]] == [
        Path(cfg.train_file).name, Path(cfg.test_files[0]).name]


def test_the_same_id_with_another_construction_is_another_version_beside_the_first(tmp_path, hub, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    first = a.Workflow(cfg).prepare().build_datasets()
    before = dict(stored(hub, first.datasets.version))
    changed = copy.deepcopy(cfg)
    changed.output_dir = str(tmp_path / "second")
    changed.teacher_epochs = 3
    capsys.readouterr()
    second = a.Workflow(changed)
    log = capsys.readouterr().out
    assert second.datasets.version != first.datasets.version and second.datasets.state == "new"
    assert f"{first.datasets.version} differs in config.teacher_epochs" in log
    second.prepare()
    # The first version is untouched, the second starts as a partial dataset of its own.
    assert stored(hub, first.datasets.version) == before
    partial = json.loads(stored(hub, second.datasets.version)["dataset_manifest.json"])
    assert partial["complete"] is False and partial["version"] == second.datasets.version
    assert set(stored(hub, second.datasets.version)) == {"compact_records.pt", "dataset_manifest.json"}
    # Changed logs are another version too, and the first dataset still resolves when they are restored.
    train = Path(cfg.train_file)
    original = train.read_bytes()
    tampered = bytearray(original)
    tampered[1] ^= 1
    train.write_bytes(bytes(tampered))
    third = copy.deepcopy(cfg)
    third.output_dir = str(tmp_path / "third")
    assert a.Workflow(third).datasets.version not in {first.datasets.version, second.datasets.version}
    train.write_bytes(original)
    restored = copy.deepcopy(cfg)
    restored.output_dir = str(tmp_path / "restored")
    assert a.Workflow(restored).datasets.version == first.datasets.version


def test_an_interrupted_build_resumes_from_the_partial_dataset_on_the_hub(tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    original = a.fit_snapshot

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("simulated session end")

    monkeypatch.setattr(a, "fit_snapshot", interrupted)
    with pytest.raises(KeyboardInterrupt, match="simulated session end"):
        a.Workflow(cfg).prepare().build_datasets()
    (version,) = {name.split("/")[1] for name in hub.repos["user/datasets"]}
    partial = stored(hub, version)
    assert "teacher_completed.pt" in partial and "snapshot_completed.pt" not in partial
    assert json.loads(partial["dataset_manifest.json"])["complete"] is False
    # The next session starts empty, downloads the teacher stage, and trains only what is missing.
    shutil.rmtree(tmp_path / "run")
    monkeypatch.setattr(a, "fit_snapshot", original)
    forbid(monkeypatch, "fit_teachers")
    capsys.readouterr()
    resumed = a.Workflow(cfg)
    assert resumed.datasets.state == "partial"
    resumed.prepare().build_datasets()
    log = capsys.readouterr().out
    assert "partly built in user/datasets" in log and "Loaded completed teacher stage." in log
    complete = stored(hub, version)
    assert set(complete) == DATASET_FILES and json.loads(complete["dataset_manifest.json"])["complete"] is True


def test_a_session_ended_inside_model_training_continues_from_the_state_on_the_hub(
        tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.output_dir, reference_cfg.dataset_id = str(tmp_path / "reference"), "reference"
    reference = a.Workflow(reference_cfg).prepare().build_datasets()
    # The session ends inside Stage 2, a few batches after its first saved check.
    teacher_saves = (cfg.teacher_folds + 1) * cfg.teacher_epochs * cfg.dev_checks_per_epoch
    restore = interrupt_after_save(monkeypatch, teacher_saves + 1, 3)
    with pytest.raises(KeyboardInterrupt, match="simulated session end"):
        a.Workflow(cfg).prepare().build_datasets()
    restore()
    (version,) = {name.split("/")[1] for name in hub.repos["user/datasets"] if name.startswith("adaptive-v9/")}
    partial = stored(hub, version)
    assert "teacher_completed.pt" in partial and "snapshot_completed.pt" not in partial
    assert "training_progress/snapshot.pt" in partial
    assert not any(name.startswith("training_progress/teacher") for name in partial)
    # Kaggle restarted: an empty working folder, and the notebook runs from its first cell.
    shutil.rmtree(tmp_path / "run")
    forbid(monkeypatch, "fit_teachers")
    capsys.readouterr()
    resumed = a.Workflow(cfg)
    assert resumed.datasets.state == "partial"
    resumed.prepare().build_datasets()
    log = capsys.readouterr().out
    assert "Loaded completed teacher stage." in log and "Snapshot: resumed at epoch 0.500 from snapshot.pt" in log
    complete = stored(hub, version)
    assert set(complete) == DATASET_FILES and json.loads(complete["dataset_manifest.json"])["complete"] is True
    assert not (tmp_path / "run" / a.PROGRESS_DIR).exists()
    built, expected = (a.load_checkpoint(root / "snapshot_completed.pt") for root in (tmp_path / "run",
                                                                                      tmp_path / "reference"))
    assert built["history"] == expected["history"]
    assert all(torch.equal(value, built["weights"][key]) for key, value in expected["weights"].items())


def test_a_run_built_before_the_dataset_store_is_uploaded_as_a_dataset_not_rebuilt(tmp_path, hub, monkeypatch, capsys):
    torch.set_num_threads(1)
    legacy = small_config(tmp_path, hf_repo="user/run", hf_sync_minutes=1e-9)
    a.Workflow(legacy).prepare().build_datasets()
    assert DATASET_FILES - {"dataset_manifest.json"} <= set(hub.repos["user/run"])
    # The user now selects a dataset id. A new session pulls the run mirror, whose files are then adopted.
    shutil.rmtree(tmp_path / "run")
    adopting = copy.deepcopy(legacy)
    adopting.dataset_id, adopting.dataset_repo = "adaptive-v9", "user/datasets"
    forbid(monkeypatch, *BUILDERS)
    capsys.readouterr()
    work = a.Workflow(adopting)
    assert work.datasets.state == "adopted" and "not rebuilt" in capsys.readouterr().out
    work.prepare().build_datasets()
    remote = stored(hub, work.datasets.version)
    assert set(remote) == DATASET_FILES and json.loads(remote["dataset_manifest.json"])["complete"] is True
    assert remote["teacher_completed.pt"] == (tmp_path / "run" / "teacher_completed.pt").read_bytes()
    assert a.file_digest(tmp_path / "run" / "decision_head_pretrained.pt") == json.loads(
        remote["dataset_manifest.json"])["artifacts"]["decision_head_pretrained.pt"]["sha256"]


def test_an_uploaded_dataset_is_never_overwritten_or_silently_replaced(tmp_path, hub):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    a.Workflow(cfg).prepare().build_datasets()
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
    original = (run / "snapshot_completed.pt").read_bytes()
    (run / "snapshot_completed.pt").write_bytes(original + b"x")
    with pytest.raises(ValueError, match="snapshot_completed.pt differs from dataset 'adaptive-v9'"):
        a.Workflow(cfg)
    (run / "snapshot_completed.pt").write_bytes(original)
    assert a.Workflow(cfg).datasets.state == "complete"
    # A damaged download is detected the same way.
    version = a.Workflow(cfg).datasets.version
    hub.repos["user/datasets"][f"adaptive-v9/{version}/teacher_completed.pt"] += b"damaged"
    damaged = copy.deepcopy(cfg)
    damaged.output_dir = str(tmp_path / "damaged")
    with pytest.raises(ValueError, match="teacher_completed.pt differs from dataset"):
        a.Workflow(damaged)


def test_a_hub_outage_never_leads_to_a_rebuild_and_missing_logs_need_one_matching_version(
        tmp_path, hub, monkeypatch):
    torch.set_num_threads(1)
    cfg = dataset_config(tmp_path)
    forbid(monkeypatch, *BUILDERS)
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
    built = a.Workflow(cfg).prepare().build_datasets()
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
    built = a.Workflow(cfg).prepare().build_datasets()
    same = copy.deepcopy(cfg)
    same.output_dir, same.snapshot_complete_share = str(tmp_path / "same"), 0
    assert a.Workflow(same).datasets.version == built.datasets.version
    # A complete version is never written to by a session that only loaded it, even by a differing local file.
    before = dict(hub.repos["user/datasets"])
    session = copy.deepcopy(cfg)
    session.output_dir = str(tmp_path / "reader")
    reader = a.Workflow(session)
    (tmp_path / "reader" / "teacher_fold_1.pt").write_bytes(b"legacy copy")
    reader.prepare()
    reader.push("anything")
    assert hub.repos["user/datasets"] == before and "read-only" in reader.datasets.describe()
