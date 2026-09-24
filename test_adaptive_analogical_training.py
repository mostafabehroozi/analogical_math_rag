"""Offline contract and end-to-end tests; no provider/network access."""
import copy
import ast
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

import adaptive_analogical_training as a


def raw_record(index, cfg, prefix="train", all_wrong=False):
    rng = np.random.RandomState(index + 810)
    eids = [str(100+j) for j in range(cfg.k)]
    cids = [f"zs_{i}" for i in range(cfg.zero_shots)] + eids
    labels = rng.randint(0, 2, cfg.n)
    if all_wrong:
        labels[:] = 0
    candidates = {cid: {"source_exemplar_idx": -1 if i < cfg.zero_shots else int(cid),
                        "candidate_text": f"Reasoning {i}", "generation_status": "SUCCESS"}
                  for i, cid in enumerate(cids)}
    state = {
        "target_query_data": {"query_text": f"{prefix} unique mathematical question {index}"},
        "retrieved_set": [{"corpus_index": int(e), "similarity_score": .9-.1*j} for j,e in enumerate(eids)],
        "candidate_set": candidates,
        "ground_truth_labels": {cid: {"is_correct": bool(labels[i]), "evaluation_status": "SUCCESS"}
                                for i,cid in enumerate(cids)},
        "intrinsic_baselines": {e: float(rng.randint(cfg.repeats+1)/cfg.repeats) for e in eids},
        "cross_evaluation_matrix": {cid: {e: float(rng.randint(cfg.repeats+1)/cfg.repeats) for e in eids}
                                    for cid in cids},
    }
    return {"target_query_original_hard_list_idx": index,
            "target_query_text": state["target_query_data"]["query_text"], "layer1_base_execution_state": state}


def record(cfg, index=0):
    return a.parse_record(raw_record(index, cfg), index, "train", cfg)


def fake_predictor(cfg):
    torch.set_num_threads(1)
    a.seed_everything(7)
    model = a.ResNet(cfg.input_dim, 3*cfg.n+2, cfg)
    return a.FrozenPredictor({"weights": a.cpu_state(model), "temperatures": np.ones(5)}, cfg, torch.device("cpu"))


def test_schema_counts_and_all_wrong_labels():
    cfg = a.Config()
    r = a.parse_record(raw_record(0, cfg, all_wrong=True), 0, "train", cfg)
    assert not r.labels.any()
    assert a.teacher_features(r, cfg).shape == (8, 22)
    assert a.observation(r, a.State(), cfg).shape == (223,)
    assert cfg.full_cost == 233
    assert a.state_cost(a.State(), cfg) == 11
    broken = raw_record(0, cfg)
    broken["layer1_base_execution_state"]["ground_truth_labels"]["zs_0"]["is_correct"] = None
    with pytest.raises(ValueError, match="unknown_or_failed_label"):
        a.parse_record(broken, 0, "train", cfg)


def test_safe_max_is_fixed_prefix_not_all_correct_candidates():
    cfg = a.Config()
    scores = np.array([8, 5, 4, 7, 6, 3, 2, 1])
    truth = np.array([1, 0, 1, 1, 1, 0, 0, 0])
    safe, maximum = a.teacher_labels(scores, truth, cfg)
    assert np.flatnonzero(safe).tolist() == [0, 3, 4]
    assert np.flatnonzero(maximum).tolist() == [0]
    truth[0] = 0
    safe, maximum = a.teacher_labels(scores, truth, cfg)
    assert not safe.any() and not maximum.any()


def test_all_coherent_structural_combinations_and_independent_source_roles():
    cfg = a.Config()
    catalog = a.structural_catalog(cfg.zero_shots, cfg.k)
    assert len(catalog) == len(set(catalog)) == 24757
    assert len(a.reachable_states(cfg)) == 186
    # R3 generated a candidate but does not evaluate; R1 evaluates without
    # generating its candidate. Both sources have been retrieved.
    triple = (1 | (1 << (cfg.zero_shots + 2)), (1 << 0) | (1 << 2), 1 << 0)
    assert triple in catalog
    state = a.complete_snapshot_state(*triple, cfg)
    assert state.retrieved == 5 and state.evaluators == 1
    assert a.bit_array(state.ccs_observed, cfg.n * cfg.k).sum() == 2
    assert a.state_cost(state, cfg) == 17
    assert a.observation(record(cfg), state, cfg).shape == (223,)
    # An evaluator or one-shot candidate without retrieval is contradictory.
    with pytest.raises(ValueError, match="without retrieved source"):
        a.complete_snapshot_state(1, 0, 1, cfg)
    with pytest.raises(ValueError, match="without retrieved source"):
        a.complete_snapshot_state(1 | (1 << cfg.zero_shots), 0, 0, cfg)


def test_missing_measurements_do_not_reveal_future_values_or_change_truth():
    cfg = a.Config()
    r = record(cfg)
    state = a.complete_snapshot_state(1 | (1 << cfg.zero_shots), 1, 1, cfg)
    missing = a.SnapshotState(state.candidates, state.retrieved, state.evaluators,
                              0, 1, 0, state.ccs_attempted, 0)
    a.validate_snapshot_state(missing, cfg)
    altered = copy.deepcopy(r)
    altered.similarity[:] = .123
    altered.baseline[:] = .456
    altered.ccs[:] = .789
    assert np.array_equal(a.observation(r, missing, cfg), a.observation(altered, missing, cfg))
    assert a.state_cost(missing, cfg) == a.state_cost(state, cfg)
    safe, maximum = np.zeros(cfg.n), np.zeros(cfg.n)
    safe[cfg.zero_shots], maximum[cfg.zero_shots] = 1, 1
    y = a.snapshot_target(r, safe, maximum, missing, cfg)
    assert y[3*cfg.n] == y[3*cfg.n+1] == 1
    assert np.array_equal(y[:cfg.n], r.labels)
    unattempted = a.SnapshotState(missing.candidates, missing.retrieved, missing.evaluators,
                                   0, 0, 0, 0, 0)
    assert a.state_cost(unattempted, cfg) == 2


def test_snapshot_permutation_preserves_evidence_and_label_alignment():
    cfg = a.Config()
    r = record(cfg)
    state = a.complete_snapshot_state(1 | (1 << (cfg.zero_shots + 3)), 1 << 3, 0, cfg)
    safe = np.arange(cfg.n) % 2
    maximum = np.eye(1, cfg.n, 0).ravel()
    pr, ps, py, pm = a.permute_snapshot(r, state, safe, maximum, cfg, np.random.RandomState(5))
    assert pr.labels.sum() == r.labels.sum()
    assert py.sum() == safe.sum() and pm.sum() == maximum.sum()
    assert a.state_cost(ps, cfg) == a.state_cost(state, cfg)
    assert pr.ccs.shape == r.ccs.shape
    a.validate_snapshot_state(ps, cfg)
    assert a.observation(pr, ps, cfg).shape == (cfg.input_dim,)


def test_augmented_snapshot_dataset_changes_across_epochs_and_preserves_labels():
    cfg = a.Config(snapshot_reachable_fraction=.5, snapshot_missing_fraction=1.,
                   snapshot_permutation_fraction=0.)
    r = record(cfg)
    teacher = {"safe": np.ones((1, cfg.n), np.float32),
               "maximum": np.eye(1, cfg.n, 0).astype(np.float32)}
    x0, y0 = a.build_snapshots([r], [0], teacher, cfg, 8, 31, epoch=0, augmented=True)
    x1, y1 = a.build_snapshots([r], [0], teacher, cfg, 8, 32, epoch=1, augmented=True)
    assert x0.shape == x1.shape == (8, 223)
    assert y0.shape == y1.shape == (8, 3*cfg.n+2+cfg.n)
    assert not np.array_equal(x0, x1)
    for row in y0:
        assert np.array_equal(row[:cfg.n], r.labels)
        assert row[3*cfg.n] == 1
        assert row[3*cfg.n+1] <= row[3*cfg.n]


def test_hidden_future_data_cannot_change_observation_or_action():
    cfg = a.Config(device="cpu", hidden_dim=8, residual_blocks=1)
    r, state = record(cfg), a.State()
    changed = copy.deepcopy(r)
    changed.ccs[1:, :] = 0.123
    changed.ccs[0, 1:] = .456
    changed.baseline[1:] = .789
    changed.similarity[1:] = -.9
    changed.labels[:] = 1 - changed.labels
    assert np.array_equal(a.observation(r, state, cfg), a.observation(changed, state, cfg))
    predictor = fake_predictor(cfg)
    h1, p1 = predictor.predict(r, state)
    h2, p2 = predictor.predict(changed, state)
    assert np.array_equal(h1, h2) and np.array_equal(p1, p2)
    for block in range(3):
        assert not p1[block*cfg.n+1:(block+1)*cfg.n].any()
    head = a.ActionHead(cfg)
    valid = a.valid_actions(state, cfg)
    assert a.greedy_action(head, h1, valid, "cpu") == a.greedy_action(head, h2, valid, "cpu")
    nxt = a.advance(state, 0, cfg)
    assert not np.array_equal(a.observation(r, nxt, cfg), a.observation(changed, nxt, cfg))


@pytest.mark.parametrize("unit,full,initial", [("solver_calls",233,11), ("total_calls",458,21)])
def test_actions_cost_and_full_pool_order_invariance(unit, full, initial):
    cfg = a.Config(cost_unit=unit)
    assert cfg.full_cost == full and a.state_cost(a.State(), cfg) == initial
    assert np.flatnonzero(a.valid_actions(a.State(), cfg)).tolist() == [0,1,2]
    for seed in range(8):
        state, rng, steps = a.State(), np.random.RandomState(seed), 0
        total = initial
        while a.valid_actions(state, cfg).any():
            next_state = a.advance(state, int(rng.choice(np.flatnonzero(a.valid_actions(state, cfg)))), cfg)
            total += a.state_cost(next_state, cfg) - a.state_cost(state, cfg)
            state, steps = next_state, steps+1
        assert total == full and steps == 11
        assert state.candidates == 255 and state.evaluators == 5


def test_absent_candidates_have_no_supervised_gradient():
    cfg = a.Config()
    r = record(cfg)
    safe = np.ones(cfg.n, np.float32)
    maximum = np.zeros(cfg.n, np.float32)
    maximum[0] = 1
    y = torch.tensor(a.snapshot_target(r, safe, maximum, a.State(), cfg)[None])
    z = torch.zeros((1,3*cfg.n+2), requires_grad=True)
    a.snapshot_loss(z, y, cfg).backward()
    for offset in (0, cfg.n, 2*cfg.n):
        assert torch.equal(z.grad[0,offset+1:offset+cfg.n], torch.zeros(cfg.n-1))
    assert z.grad[0,0] != 0


def test_safe_presence_must_match_returned_candidate_when_gated():
    cfg = a.Config(stop_mode="safe", stop_threshold=.9, member_threshold=.8)
    state = a.advance(a.State(), 1, cfg)
    probabilities = np.zeros(3*cfg.n+2, np.float32)
    probabilities[:2] = [.9,.5]
    probabilities[3*cfg.n] = .99
    probabilities[cfg.n+1] = .99   # Some SAFE candidate exists, but it is not selected.
    assert a.stopping_reason(probabilities, state, cfg) is None
    probabilities[cfg.n] = .9
    assert a.stopping_reason(probabilities, state, cfg) == "safe"


def test_bellman_masks_terminals_and_double_dqn():
    reward = torch.tensor([.1, .2])
    done = torch.tensor([False, True])
    valid = torch.tensor([[True,False,True],[False,False,False]])
    target = torch.tensor([[2.,999.,3.],[9.,9.,9.]])
    assert torch.allclose(a.bellman_target(reward, done, valid, target, 1), torch.tensor([3.1,.2]))
    online = torch.tensor([[4.,999.,1.],[2.,2.,2.]])
    assert torch.allclose(a.bellman_target(reward, done, valid, target, 1, online), torch.tensor([2.1,.2]))
    with pytest.raises(ValueError, match="no valid"):
        a.bellman_target(reward, torch.tensor([False,False]), valid, target, 1)


def test_terminal_reward_selects_actual_answer_and_charges_cost():
    cfg = a.Config(device="cpu", hidden_dim=8, residual_blocks=1, max_cost=17,
                   stop_threshold=1, member_threshold=1)
    r, predictor = record(cfg), fake_predictor(cfg)
    env = a.CachedEnvironment(r, predictor, cfg)
    assert np.flatnonzero(a.valid_actions(env.state, cfg)).tolist() == [1,2]
    reward, done = env.step(1)
    selected = a.selected_candidate(env.probabilities, env.state, cfg)
    assert done and env.reason == "budget"
    assert reward == pytest.approx(r.labels[selected] - cfg.cost_weight*6/cfg.full_cost)
    with pytest.raises(ValueError, match="termination"):
        env.step(0)


def test_query_splits_and_streamed_cache_envelope(tmp_path):
    cfg = a.Config(train_file=str(tmp_path/"train.json"), test_files=[], teacher_folds=2)
    records = [record(cfg,i) for i in range(40)]
    splits = a.split_records(records, cfg)
    sets = [set(v) for v in splits.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 40
    path = tmp_path/"cache.json"
    path.write_text(json.dumps({"queries": {"13": raw_record(13,cfg)["layer1_base_execution_state"]}}))
    pairs = list(a.iter_log(path))
    assert pairs[0][0] == "13"
    assert a.parse_record(pairs[0][1], "13", "cache", cfg).uid == "cache::13"


def test_all_wrong_metrics_not_dropped():
    cfg = a.Config()
    r = a.parse_record(raw_record(0,cfg,all_wrong=True),0,"train",cfg)
    metrics = a.ranking_metrics([r],[0],{0:np.arange(cfg.n)},cfg)
    assert metrics["n"] == 1 and metrics["top1"] == 0 and metrics["ap"] == 0
    assert metrics["conditional_ap"] is None


def test_complete_pipeline_checkpoint_resume_and_oof(tmp_path):
    train, test = tmp_path/"train.json", tmp_path/"test.json"
    cfg = a.Config(train_file=str(train), test_files=[str(test)], output_dir=str(tmp_path/"run"),
                   device="cpu", cpu_threads=1, teacher_folds=2, hidden_dim=8, residual_blocks=1,
                   batch_size=16, teacher_epochs=2, snapshot_epochs=2, patience=2,
                   snapshots_per_query=3, dev_snapshots_per_query=3,
                   rl_steps=24, rl_batch_size=4, warmup=4, replay_size=64,
                   target_update=2, eval_every=12, bootstrap_samples=20,
                   stop_threshold=1, member_threshold=1, max_cost=45)
    train.write_text(json.dumps([raw_record(i,cfg) for i in range(40)]))
    test.write_text(json.dumps([raw_record(i,cfg,prefix="external",all_wrong=(i==0)) for i in range(5)]))
    work = a.run_pipeline(cfg)
    assert work.results["test"]["adaptive"]["policies"]["rl"]["n"] == 5
    for fold in work.teacher["folds"]:
        assert not set(fold["heldout_ids"]) & (set(fold["train_ids"]) | set(fold["validation_ids"]))
    assert len(set.union(*(set(f["heldout_ids"]) for f in work.teacher["folds"]))) == len(work.splits["supervised"])
    frozen = a.cpu_state(work.predictor.model)
    assert not any(p.requires_grad for p in work.predictor.model.parameters())
    restored_cfg, predictor, head = a.load_inference_bundle(tmp_path/"run"/"inference_bundle.pt")
    assert all(torch.equal(frozen[k], v.cpu()) for k,v in predictor.model.state_dict().items())
    r = work.records[work.splits["test"][0]]
    assert a.rollout(r,predictor,restored_cfg,head=head)["selected"] == a.rollout(r,work.predictor,cfg,head=work.head)["selected"]
    resumed = a.Workflow(cfg).prepare().train_teacher().train_snapshot().train_rl()
    assert resumed.teacher["scores"].shape == (45,8)
    changed = copy.deepcopy(cfg)
    changed.stop_threshold = .90
    with pytest.raises(ValueError, match="different config"):
        a.Workflow(changed)


def test_missing_labels_and_exact_overlap_are_audited(tmp_path):
    train, test = tmp_path/"train.json", tmp_path/"test.json"
    cfg = a.Config(train_file=str(train), test_files=[str(test)])
    good = raw_record(0,cfg,all_wrong=True)
    bad = raw_record(1,cfg)
    del bad["layer1_base_execution_state"]["ground_truth_labels"]["zs_0"]
    train.write_text(json.dumps([good,bad]))
    test.write_text(json.dumps([good]))
    records, audit = a.load_records(cfg)
    assert len(records) == 1 and not records[0].labels.any()
    assert audit["train"]["unknown_or_failed_label"] == 1
    assert audit["test"]["duplicate_or_cross_file_overlap"] == 1


def test_notebook_is_self_contained_compiles_and_matches_module():
    root = Path(__file__).parent
    nb = json.loads((root/"adaptive_analogical_training_kaggle.ipynb").read_text(encoding="utf-8"))
    embedded = []
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"notebook_cell_{index}", "exec")
        if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
            embedded.append(source)
    original = (root/"adaptive_analogical_training.py").read_text(encoding="utf-8").split("# %% CLI")[0]
    assert ast.dump(ast.parse("\n".join(embedded))) == ast.dump(ast.parse(original))
    assert not any(cell.get("outputs") for cell in nb["cells"])
