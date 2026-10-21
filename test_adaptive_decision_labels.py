"""Offline checks for the Stage 3 acquisition rules, nearest-goal routes, and both label phases.

Each check compares the vectorized code with an independent brute-force statement of the same rule.
"""
import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import adaptive_analogical_training as a

ACTION, UNREACHABLE = (a.DECISION_STATUSES.index(name) for name in ("ACTION", "UNREACHABLE"))


def allowed_moves(state, cfg):
    """The next-step rule, restated by the status of each source."""
    moves = {}
    missing = [z for z in range(cfg.zero_shots) if not state.candidate_mask >> z & 1]
    if missing:
        moves[0] = a.State(state.candidate_mask | 1 << missing[0], state.evaluator_mask)
    status = [(bool(state.candidate_mask >> (cfg.zero_shots + j) & 1), bool(state.evaluator_mask >> j & 1))
              for j in range(cfg.k)]
    opening = next((j for j, (one_shot, evaluator) in enumerate(status) if not one_shot and not evaluator), None)
    for j, (one_shot, evaluator) in enumerate(status):
        used = one_shot or evaluator
        if not evaluator and (used or j == opening):
            moves[1 + j] = a.State(state.candidate_mask, state.evaluator_mask | 1 << j)
        if not one_shot and (used or j == opening):
            moves[1 + cfg.k + j] = a.State(state.candidate_mask | 1 << (cfg.zero_shots + j), state.evaluator_mask)
    return moves


@pytest.mark.parametrize("overrides", [{}, {"max_cost": 70.}])
def test_decision_graph_agrees_with_an_independent_statement_of_the_rule(overrides):
    cfg = a.Config(k=3, zero_shots=2, repeats=2, **overrides)
    graph = a.decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    index = {state: i for i, state in enumerate(graph.states)}
    assert all(a.state_cost(state, cfg) <= cfg.budget for state in graph.states)
    for i, state in enumerate(graph.states):
        expected = {action: nxt for action, nxt in allowed_moves(state, cfg).items()
                    if a.state_cost(nxt, cfg) <= cfg.budget}
        got = {action: graph.states[graph.succ[i, action]]
               for action in range(cfg.action_count) if graph.succ[i, action] >= 0}
        assert expected == got
        assert graph.legal[i].tolist() == a.valid_actions(state, cfg).tolist()
        for action, nxt in got.items():
            assert graph.delta[i, action] == a.state_cost(nxt, cfg) - a.state_cost(state, cfg) > 0
    # Every action adds one item, so a backward pass over the levels sees successors first.
    sizes = [np.unique([bin(graph.states[i].candidate_mask).count("1") + bin(graph.states[i].evaluator_mask).count("1")
                        for i in level]) for level in graph.levels]
    assert all(len(size) == 1 for size in sizes) and [int(s[0]) for s in sizes] == sorted(
        {int(s[0]) for s in sizes}, reverse=True)
    if not overrides:
        # 2 zero-shot prefixes x the ways to use the nearest m of 3 sources.
        assert len(a.reachable_states(cfg)) == cfg.zero_shots * sum(3 ** m for m in range(cfg.k + 1)) == 80
        assert graph.full == index[a.State((1 << cfg.n) - 1, (1 << cfg.k) - 1)]
        assert (~graph.legal.any(1)).sum() == 1 and not graph.legal[graph.full].any()


def brute_force_distance(graph, prefix, cfg):
    @functools.lru_cache(maxsize=None)
    def best(state, rank):
        if prefix[state, rank]:
            return 0.
        options = [graph.delta[state, action] + best(int(graph.succ[state, action]), rank)
                   for action in range(cfg.action_count) if graph.succ[state, action] >= 0]
        return min(options, default=np.inf)
    return best


def test_goal_distances_and_first_steps_match_brute_force_routes():
    cfg = a.Config(k=3, zero_shots=2, repeats=2)
    graph = a.decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    rng = np.random.RandomState(5)
    informative = ties = 0
    for density in (.02, .08, .3, .02, .08, .3, .15, .15):
        goal = rng.rand(len(graph.states), cfg.n) < density
        prefix = np.logical_and.accumulate(goal, axis=1)
        achieved = prefix.sum(1)
        distance = a.goal_distances(graph, prefix)
        best = brute_force_distance(graph, prefix, cfg)
        for rank in range(cfg.n):
            for state in range(len(graph.states)):
                assert distance[rank, state] == best(state, rank)
        options = a.first_steps(graph, distance, achieved)
        for state in range(len(graph.states)):
            rank = achieved[state]
            if rank == cfg.n or not np.isfinite(best(state, rank)):
                assert not options[state].any()
                continue
            expected = {action for action in range(cfg.action_count) if graph.succ[state, action] >= 0
                        and graph.delta[state, action] + best(int(graph.succ[state, action]), rank)
                        == best(state, rank)}
            assert expected and expected == set(np.flatnonzero(options[state]).tolist())
            informative += 1
            ties += len(expected) > 1
    assert informative > 100 and ties > 20  # The comparison is not vacuous.


def test_threads_follow_the_policy_until_the_goal_holds():
    cfg = a.Config(k=3, zero_shots=2, repeats=2)
    graph = a.decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    rng = np.random.RandomState(8)
    reached_some = stuck_some = False
    for density in (.05, .2):
        prefix = np.logical_and.accumulate(rng.rand(len(graph.states), cfg.n) < density, axis=1)[None]
        move = np.full((1, len(graph.states)), -1)
        for state in range(len(graph.states)):
            legal = np.flatnonzero(graph.legal[state])
            if len(legal):
                move[0, state] = rng.choice(legal)
        reached, calls = a.thread_outcomes(graph, move, prefix)
        for rank in range(cfg.n):
            for start in range(len(graph.states)):
                state, spent = start, 0.
                while not prefix[0, state, rank] and move[0, state] >= 0:
                    spent += graph.delta[state, move[0, state]]
                    state = graph.succ[state, move[0, state]]
                expected = (float(prefix[0, state, rank]), spent)
                assert (reached[0, rank, start], calls[0, rank, start]) == expected
                reached_some |= expected[0] == 1 and spent > 0
                stuck_some |= expected[0] == 0
    assert reached_some and stuck_some


def test_visible_leaders_are_the_first_order_with_a_byte_identical_observation():
    digest = np.zeros((3, 2, 4), np.uint64)
    digest[:, 0] = 7                       # State 0 looks the same under every order.
    digest[:, 1] = [[1] * 4, [2] * 4, [2] * 4]  # State 1: orders 1 and 2 coincide.
    leader = a.visible_leaders(digest)
    assert leader.tolist() == [[0, 0], [0, 1], [0, 1]]
    digest[2, 1, 3] = 3                    # One differing word separates it.
    assert a.visible_leaders(digest).tolist() == [[0, 0], [0, 1], [0, 2]]


class Scripted:
    """A predictor whose lights depend only on the acquired items, with constant features."""
    def __init__(self, cfg, lights):
        self.cfg, self.lights, self.device = cfg, lights, torch.device("cpu")

    def predict(self, record, state, cfg_override=None):
        return np.zeros(self.cfg.head_input_dim, np.float32), self.lights(record, state)

    def predict_many(self, record, states):
        values = [self.predict(record, state) for state in states]
        return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])


def tiny_problem(lights, **overrides):
    """k=1, two zero-shots: candidates ZS1, ZS2, OS1; the recorded top candidate is OS1."""
    cfg = a.Config(k=1, zero_shots=2, hidden_dim=4, decision_hidden=8, **overrides)
    rng = np.random.RandomState(3)
    record = a.Record("q", "q", "synthetic", ["zs_0", "zs_1", "os_0"], ["e0"], np.array([.9], np.float32),
                      np.array([.4], np.float32), (rng.randint(6, size=(cfg.n, cfg.k)) / 5).astype(np.float32),
                      np.ones(cfg.n, np.float32))
    scores = np.array([.8, .7, .9], np.float32)
    flags = np.array([0, 0, 1], np.float32)  # Safe and max: OS1.
    return cfg, record, scores, flags, flags, Scripted(cfg, lambda rec, state: lights(cfg, rec, state))


def recognizes(cfg, state, candidate, needs_evaluator=False):
    """Lights for `candidate` once present (and, if asked, once an evaluator is active)."""
    p = np.zeros(3 * cfg.n + 2, np.float32)
    if state.candidate_mask >> candidate & 1 and (state.evaluator_mask or not needs_evaluator):
        p[[cfg.n + candidate, 2 * cfg.n + candidate, 3 * cfg.n, 3 * cfg.n + 1]] = .9
        p[candidate] = .9
    return p


def targets_at(plan, state, order=0):
    i = {s: j for j, s in enumerate(plan["graph"].states)}[state]
    return set(np.flatnonzero(plan["targets"][order, i]).tolist()), plan["status"][order, i], plan["rows"][order, i]


def test_a_one_shot_that_alone_reaches_the_goal_is_the_only_cheapest_first_step():
    # Actions with one source: 0 next zero-shot, 1 add R1, 2 generate OS1.
    cfg, record, scores, safe, maximum, predictor = tiny_problem(lambda c, r, s: recognizes(c, s, 2))
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    # OS1 costs one call; reaching it through R1 first costs 20 more.
    assert targets_at(plan, a.State()) == ({2}, ACTION, True)
    assert targets_at(plan, a.State(1, 1))[0] == {2}
    assert targets_at(plan, a.State(3, 0))[0] == {2}


def test_every_first_step_of_equally_cheap_routes_is_labeled():
    # Recognition needs OS1 and an evaluator: R1 then OS1 and OS1 then R1 both cost 31.
    cfg, record, scores, safe, maximum, predictor = tiny_problem(
        lambda c, r, s: recognizes(c, s, 2, needs_evaluator=True))
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    assert targets_at(plan, a.State()) == ({1, 2}, ACTION, True)  # The zero-shot would overshoot.
    assert targets_at(plan, a.State(1, 1))[0] == {2} and targets_at(plan, a.State(5, 0))[0] == {1}
    targets = a.decision_targets(np.packbits(plan["targets"].reshape(plan["rows"].size, -1), axis=1)[
        np.flatnonzero(plan["rows"].reshape(-1))], "pretrain", cfg)
    assert np.allclose(targets.sum(1), 1) and set(np.unique(targets)) <= {0., .5, 1.}


def test_a_group_without_a_route_defers_to_its_routed_group_mate_and_alone_has_no_preference():
    # Only when slot 0 holds zs_0 does the evaluator light the candidate. Under the swapped order
    # the goal is out of reach, yet both orders look identical until an evaluator is acquired.
    def lights(cfg, record, state):
        return recognizes(cfg, state, 0, needs_evaluator=True) if record.candidate_ids[0] == "zs_0" else (
            np.zeros(3 * cfg.n + 2, np.float32))
    cfg, record, scores, safe, maximum, predictor = tiny_problem(lights)
    safe = maximum = np.array([1, 0, 0], np.float32)
    plan = a.question_plan(record, np.array([.9, .8, .7], np.float32), safe, maximum, predictor, cfg)
    index = {s: j for j, s in enumerate(plan["graph"].states)}
    root = index[a.State()]
    assert plan["leader"][:, root].tolist() == [0, 0]
    assert plan["status"][:, root].tolist() == [ACTION, UNREACHABLE]
    assert plan["rows"][:, root].tolist() == [True, False]           # The routed mate speaks for the group.
    assert targets_at(plan, a.State()) == ({1}, ACTION, True) and not plan["targets"][1, root].any()
    # Where the swapped order's input is its own, it has no route and so no preference: every move.
    own = index[a.State(3, 1)]
    assert plan["leader"][1, own] == 1 and plan["status"][1, own] == UNREACHABLE and plan["rows"][1, own]
    assert plan["targets"][1, own].tolist() == plan["graph"].legal[own].tolist()
    # The complete state has no move and so no row, in either order.
    assert not plan["rows"][:, plan["graph"].full].any()


def test_every_state_but_the_complete_one_has_a_row_when_no_goal_is_reachable():
    cfg, record, scores, safe, maximum, predictor = tiny_problem(lambda c, r, s: np.zeros(3 * c.n + 2, np.float32))
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    graph = plan["graph"]
    assert len(graph.states) == 14
    assert plan["rows"].sum() == 2 * 13 and not plan["rows"][:, graph.full].any()
    assert (plan["targets"][plan["rows"]] == np.broadcast_to(graph.legal, plan["targets"].shape)[plan["rows"]]).all()
    assert (plan["status"][:, graph.full] == a.DECISION_STATUSES.index("EXHAUSTED")).all()
    coverage = a.plan_coverage(plan, cfg)
    assert coverage["state_order_cases"] == 28 and coverage["action_rows"] == 26
    assert coverage["informative_rows"] == 0 and coverage["status_counts"] == {"UNREACHABLE": 26, "EXHAUSTED": 2}


def test_goal_complete_states_keep_a_row_and_a_budget_leaves_unaffordable_cases_out():
    def lights(cfg, record, state):
        p = np.zeros(3 * cfg.n + 2, np.float32)
        for slot in range(cfg.n):
            if state.candidate_mask >> slot & 1:
                p[slot] = {"zs_0": .8, "zs_1": .7, "os_0": .9}[record.candidate_ids[slot]]
        return p
    cfg, record, scores, safe, maximum, predictor = tiny_problem(lights)
    safe = maximum = np.zeros(cfg.n, np.float32)
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    complete = a.DECISION_STATUSES.index("COMPLETE")
    # Every goal holds wherever every candidate is present, evaluators or not.
    graph = plan["graph"]
    all_candidates = [j for j, s in enumerate(graph.states) if s.candidate_mask == 7]
    assert (plan["status"][:, all_candidates] == complete).all() and not (
        plan["status"][:, [j for j in range(len(graph.states)) if j not in all_candidates]] == complete).any()
    # Every other case with a move is a row, and so is a goal-complete state that still has a move:
    # with R1 left to add it has no preference, so it is labeled with every move it has.
    assert plan["rows"][(plan["status"] != complete) & graph.legal.any(1)[None]].all()
    assert targets_at(plan, a.State(7, 0)) == ({1}, complete, True)
    assert not plan["rows"][:, graph.full].any()  # The complete state has no move.
    capped = a.Config(k=1, zero_shots=2, hidden_dim=4, max_cost=25.)
    flat = Scripted(capped, lambda r, s: np.zeros(3 * capped.n + 2, np.float32))
    plan = a.question_plan(record, scores, safe, maximum, flat, capped)
    coverage = a.plan_coverage(plan, capped)
    assert coverage["status_counts"].get("OUT_OF_BUDGET", 0) == 8 and coverage["status_counts"].get("BUDGET", 0) > 0
    assert coverage["state_order_cases"] == 28
    assert not plan["rows"][:, ~plan["graph"].legal.any(1)].any()
    assert plan["graph"].full == -1


def head_with_bias(cfg, bias):
    """A head that ignores its input and ranks the actions by `bias`."""
    head = a.SupervisedActionHead(cfg)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.net[2].bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return head


def finetune(plan, head, cfg):
    chosen, extra = a.finetune_choices(plan, head, torch.device("cpu"), cfg)
    return chosen, extra


def test_the_cheaper_continuation_wins_over_the_head_probability():
    # Recognition needs OS1 and an evaluator. The head likes OS1 > ZS > R1 and so, after R1, goes
    # straight to OS1 (31 calls); after OS1 it detours through a zero-shot (42 calls).
    cfg, record, scores, safe, maximum, predictor = tiny_problem(
        lambda c, r, s: recognizes(c, s, 2, needs_evaluator=True))
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    head = head_with_bias(cfg, [1., 0., 2.])  # Actions: next zero-shot, R1, OS1.
    chosen, extra = finetune(plan, head, cfg)
    root = {s: j for j, s in enumerate(plan["graph"].states)}[a.State()]
    probability = a.head_policy(head, plan["features"], plan["graph"], "cpu")[1]
    assert probability[0, root, 2] > probability[0, root, 1]          # The head favors OS1 at the start ...
    assert chosen[:, root].tolist() == [1, 1]                         # ... but R1 is the cheaper route.
    assert extra["no_thread_reaches_rows"] == 0


@pytest.mark.parametrize("bias,expected", [([0., 2., 1.], 1), ([0., 1., 2.], 2)])
def test_equal_continuations_are_ordered_by_the_pretrained_probability(bias, expected):
    # Recognition needs OS1 and an evaluator; this head reaches it at 31 calls whichever comes first.
    cfg, record, scores, safe, maximum, predictor = tiny_problem(
        lambda c, r, s: recognizes(c, s, 2, needs_evaluator=True))
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    chosen, _ = finetune(plan, head_with_bias(cfg, bias), cfg)
    root = {s: j for j, s in enumerate(plan["graph"].states)}[a.State()]
    assert chosen[:, root].tolist() == [expected, expected]


def test_options_that_never_reach_the_goal_fall_back_to_the_pretrained_probability():
    # Only the state ZS1 + ZS2 + R1 is recognized. A head that always generates OS1 first walks past it.
    def lights(cfg, record, state):
        slot = record.candidate_ids.index("zs_0")
        return recognizes(cfg, state, slot) if (state.candidate_mask, state.evaluator_mask) == (3, 1) else (
            np.zeros(3 * cfg.n + 2, np.float32))
    cfg, record, scores, safe, maximum, predictor = tiny_problem(lights)
    safe = maximum = np.array([1, 0, 0], np.float32)
    scores = np.array([.9, .8, .7], np.float32)
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    root = {s: j for j, s in enumerate(plan["graph"].states)}[a.State()]
    assert targets_at(plan, a.State())[0] == {0, 1}
    for bias, expected in (([2., 1., 3.], 0), ([1., 2., 3.], 1)):
        chosen, extra = finetune(plan, head_with_bias(cfg, bias), cfg)
        assert chosen[:, root].tolist() == [expected, expected]
        assert extra["no_thread_reaches_rows"] >= 2


def explicit_choices(plan, head):
    """The Stage 3b rule restated with the deployment functions: advance, valid_actions' mask, greedy move."""
    graph, prefix = plan["graph"], plan["prefix"]
    orders, states, _ = prefix.shape
    with torch.no_grad():
        logits = head(torch.as_tensor(plan["features"].reshape(orders * states, -1))).numpy().astype(np.float64)
    logits = logits.reshape(orders, states, -1)

    def move(order, state):
        legal = graph.legal[state]
        return int(np.where(legal, logits[order, state], -np.inf).argmax()) if legal.any() else None

    def thread(order, state, rank):
        spent = 0.
        while not prefix[order, state, rank]:
            step = move(order, state)
            if step is None:
                return 0., spent
            spent += graph.delta[state, step]
            state = graph.succ[state, step]
        return 1., spent

    def probability(order, state):
        legal = graph.legal[state]
        weights = np.where(legal, np.exp(logits[order, state] - logits[order, state][legal].max()), 0.)
        return weights / weights.sum()

    chosen = np.full((orders, states), a.NO_CHOICE)
    deciding = plan["informative"] & plan["rows"]  # Routed cases that have a row.
    for state in range(states):
        for lead in range(orders):
            members = [o for o in range(orders) if plan["leader"][o, state] == lead and deciding[o, state]]
            if not members:
                continue
            union = sorted(set().union(*(set(np.flatnonzero(plan["options"][o, state]).tolist()) for o in members)))
            scored = {}
            for action in union:
                results = [thread(o, int(graph.succ[state, action]), plan["achieved"][o, state]) for o in members]
                scored[action] = (np.mean([r for r, _ in results]),
                                  np.mean([graph.delta[state, action] + c for _, c in results]))
            top = max(reach for reach, _ in scored.values())
            if top > 1e-9:
                keep = [x for x in union if abs(scored[x][0] - top) <= 1e-9]
                cheapest = min(scored[x][1] for x in keep)
                keep = [x for x in keep if abs(scored[x][1] - cheapest) <= 1e-9]
            else:
                keep = union
            odds = probability(lead, state)
            pick = max(keep, key=lambda x: (odds[x], -x))
            for o in members:
                chosen[o, state] = pick
        for order in range(orders):
            if plan["rows"][order, state] and not plan["informative"][order, state]:
                chosen[order, state] = move(order, state)
    return chosen


@pytest.mark.parametrize("threshold,seed,reachable_only", [(.5, 5, False), (.45, 1, False), (.45, 8, False),
                                                            (.35, 5, True), (.4, 5, True)])
def test_finetune_choices_match_explicit_continuations_of_the_pretrained_head(threshold, seed, reachable_only):
    torch.set_num_threads(1)
    cfg = a.Config(k=2, zero_shots=2, hidden_dim=8, residual_blocks=1, decision_hidden=8,
                   recognition_threshold=threshold, reachable_only=reachable_only)
    rng = np.random.RandomState(40)
    record = a.Record("q", "q", "synthetic", [f"zs_{i}" for i in range(2)] + ["os_0", "os_1"], ["e0", "e1"],
                      np.array([.9, .8], np.float32), (rng.randint(6, size=2) / 5).astype(np.float32),
                      (rng.randint(6, size=(cfg.n, cfg.k)) / 5).astype(np.float32), np.ones(cfg.n, np.float32))
    scores = np.arange(cfg.n, 0, -1, dtype=np.float32)
    maximum = np.array([1, 0, 0, 0], np.float32)
    torch.manual_seed(seed)
    predictor = a.FrozenPredictor({"weights": a.cpu_state(a.ResNet(cfg.input_dim, 3 * cfg.n + 2, cfg)),
                                   "temperatures": np.ones(5)}, cfg, torch.device("cpu"))
    plan = a.question_plan(record, scores, np.ones(cfg.n, np.float32), maximum, predictor, cfg)
    head = a.SupervisedActionHead(cfg)
    chosen, extra = a.finetune_choices(plan, head, torch.device("cpu"), cfg)
    expected = explicit_choices(plan, head)
    np.testing.assert_array_equal(chosen, expected)
    # The comparison is not vacuous: tied options, merged groups, and moved labels all occur.
    informative = plan["informative"] & plan["rows"]
    assert informative.sum() > 5 and (plan["options"][informative].sum(1) > 1).sum() > 0
    assert ((plan["leader"] != np.arange(2)[:, None]) & informative).sum() > 0 or threshold != .5
    if reachable_only:
        # Routed cases whose first goal holds, where the rollout has stopped, take no part.
        assert (plan["informative"] & ~plan["rows"]).any()
    # A merged group shares one decision.
    for state in range(informative.shape[1]):
        for lead in range(2):
            members = [o for o in range(2) if plan["leader"][o, state] == lead and informative[o, state]]
            assert len({int(chosen[o, state]) for o in members}) <= 1
    # Every labeled case has a legal choice, and unlabeled cases have none.
    rows = plan["rows"]
    assert (chosen[~rows] == a.NO_CHOICE).all()
    assert plan["graph"].legal[np.nonzero(rows)[1], chosen[rows]].all()
    assert extra["changed_rows"] == int((rows & (chosen != a.head_policy(head, plan["features"], plan["graph"],
                                                                         "cpu")[0])).sum())


def test_goals_are_prefixes_so_a_later_goal_alone_does_not_count():
    # Rank one (OS1) needs an evaluator. Rank two (ZS1) holds whenever ZS1 is present, and rank three
    # (ZS2) whenever ZS2 is. A later goal that holds without the earlier ones does not finish them.
    def lights(cfg, record, state):
        p = recognizes(cfg, state, 2, needs_evaluator=True)
        if state.candidate_mask & 1:
            p[0] = .5
        return p
    cfg, record, scores, safe, maximum, predictor = tiny_problem(lights)
    scenario = a.decision_scenarios(record, scores, safe, maximum, predictor, cfg)
    index = {s: j for j, s in enumerate(scenario["graph"].states)}
    assert scenario["goal"][0, index[a.State(1, 0)]].tolist() == [False, True, False]
    assert scenario["goal"][0, index[a.State(6, 1)]].tolist() == [True, False, True]
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    assert plan["achieved"][0, index[a.State(1, 0)]] == 0 and not plan["prefix"][0, index[a.State(1, 0)]].any()
    # The next goal is still rank one: it needs OS1 and an evaluator, whichever comes first.
    assert targets_at(plan, a.State(1, 0)) == ({1, 2}, ACTION, True)
    # ZS2 and OS1 with an evaluator hold ranks one and three; rank two is the first unfinished, so ZS1 is added.
    assert plan["achieved"][0, index[a.State(6, 1)]] == 1
    assert targets_at(plan, a.State(6, 1)) == ({0}, ACTION, True)


def test_safe_flags_follow_the_zero_shot_permutation():
    # Only ZS1 is SAFE/MAX. Under the swapped order ZS2 sits in slot 0 and, not being SAFE, needs no SAFE
    # light there; the unpermuted flag would wrongly demand one.
    def lights(cfg, record, state):
        p = np.zeros(3 * cfg.n + 2, np.float32)
        for slot, name in enumerate(record.candidate_ids):
            if state.candidate_mask >> slot & 1:
                p[slot] = {"zs_0": .9, "zs_1": .8, "os_0": .1}[name]
                if name == "zs_0":
                    p[[cfg.n + slot, 2 * cfg.n + slot, 3 * cfg.n, 3 * cfg.n + 1]] = .9
        return p
    cfg, record, _, _, _, predictor = tiny_problem(lights)
    safe = maximum = np.array([1, 0, 0], np.float32)
    scenario = a.decision_scenarios(record, np.array([.9, .8, .7], np.float32), safe, maximum, predictor, cfg)
    index = {s: j for j, s in enumerate(scenario["graph"].states)}
    for order in (0, 1):
        assert scenario["goal"][order, index[a.State(3, 0)]].tolist() == [True, True, False]


def test_threads_that_reach_the_goal_beat_cheaper_dead_ends():
    # A budget of 40 makes the pool unaffordable. Recognition needs OS1 and an evaluator. This head prefers
    # OS1, then a zero-shot, then R1. After R1 its only affordable moves lead to OS1 and the goal (31 calls).
    # After OS1 it adds a zero-shot and is stuck after 2 calls: cheaper, but it never gets there.
    cfg, record, scores, safe, maximum, predictor = tiny_problem(
        lambda c, r, s: recognizes(c, s, 2, needs_evaluator=True), max_cost=40.)
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    graph = plan["graph"]
    root = {s: j for j, s in enumerate(graph.states)}[a.State()]
    assert graph.full == -1 and targets_at(plan, a.State())[0] == {1, 2}
    head = head_with_bias(cfg, [1., 0., 2.])
    move, probability = a.head_policy(head, plan["features"], graph, "cpu")
    reached, calls = a.thread_outcomes(graph, move, plan["prefix"])
    rank = plan["achieved"][0, root]
    threads = {action: (reached[0, rank, graph.succ[root, action]],
                        graph.delta[root, action] + calls[0, rank, graph.succ[root, action]]) for action in (1, 2)}
    assert threads == {1: (1., 31.), 2: (0., 2.)} and probability[0, root, 2] > probability[0, root, 1]
    chosen, extra = finetune(plan, head, cfg)
    assert chosen[:, root].tolist() == [1, 1] and extra["tied_rows"] > 0


def test_head_policy_breaks_ties_like_the_deployed_greedy_action():
    cfg, record, scores, safe, maximum, predictor = tiny_problem(lambda c, r, s: np.zeros(3 * c.n + 2, np.float32))
    plan = a.question_plan(record, scores, safe, maximum, predictor, cfg)
    graph = plan["graph"]
    for bias in ([0., 0., 0.], [1., 1., 0.], [0., 2., 2.]):
        head = head_with_bias(cfg, bias)
        move, probability = a.head_policy(head, plan["features"], graph, "cpu")
        for order in range(move.shape[0]):
            for state in range(len(graph.states)):
                legal = graph.legal[state]
                if not legal.any():
                    assert move[order, state] == -1 and not probability[order, state].any()
                    continue
                assert move[order, state] == a.greedy_action(head, plan["features"][order, state], legal, "cpu")
                assert probability[order, state].sum() == pytest.approx(1) and not probability[order, state][~legal].any()


def clear_rule_caches():
    for cached in (a.state_graph, a.decision_structure, a.budget_state_index, a.decision_graph,
                   a.decision_valid_table):
        cached.cache_clear()


def test_label_identity_includes_the_acquisition_rules(monkeypatch):
    cfg = a.Config(k=2, zero_shots=2, hidden_dim=4)
    records = [a.Record(str(i), str(i), "synthetic", list(range(cfg.n)), list(range(cfg.k)),
                        np.ones(cfg.k, np.float32), np.ones(cfg.k, np.float32),
                        np.ones((cfg.n, cfg.k), np.float32), np.ones(cfg.n, np.float32)) for i in range(2)]
    teacher = {name: np.ones((2, cfg.n), np.float32) for name in ("scores", "safe", "maximum")}
    predictor = SimpleNamespace(model=torch.nn.Linear(4, 4), temperatures=np.ones(5))
    splits = {"policy": [0], "dev": [1]}
    before = a.rules_fingerprint(cfg)
    identity = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    original = a.raw_next_state

    def any_source_at_any_time(state, action, config):
        """Revision 8's looser rule: every absent one-shot candidate is available."""
        slot = config.zero_shots + action - config.k - 1
        if action > config.k and not state.candidate_mask >> slot & 1:
            return a.State(state.candidate_mask | 1 << slot, state.evaluator_mask)
        return original(state, action, config)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(a, "raw_next_state", any_source_at_any_time)
            clear_rule_caches()
            loosened = a.rules_fingerprint(cfg)
            changed_identity = a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg)
    finally:
        clear_rule_caches()
    assert loosened != before and changed_identity != identity
    assert a.rules_fingerprint(cfg) == before
    assert a.decision_dataset_fingerprint(records, splits, teacher, predictor, cfg) == identity
    # Costs are part of the rules too.
    assert a.rules_fingerprint(a.Config(k=2, zero_shots=2, repeats=3)) != before


def test_reachable_only_is_off_by_default_and_enters_the_identity_only_when_on():
    off, on = a.Config(k=2, zero_shots=2, hidden_dim=4), a.Config(k=2, zero_shots=2, hidden_dim=4, reachable_only=True)
    assert off.reachable_only is False
    # Off, the contract and the Stage 3 identity hold exactly what they held before the setting existed,
    # so runs started earlier still resume.
    assert a.contract_settings(off) == {key: value for key, value in a.asdict(off).items()
                                        if key not in a.NON_CONTRACT_SETTINGS and key not in a.GROUPING_SETTINGS
                                        and key != "reachable_only"}
    assert a.contract_settings(on) == {**a.contract_settings(off), "reachable_only": True}
    records = [a.Record(str(i), str(i), "synthetic", list(range(off.n)), list(range(off.k)),
                        np.ones(off.k, np.float32), np.ones(off.k, np.float32),
                        np.ones((off.n, off.k), np.float32), np.ones(off.n, np.float32)) for i in range(2)]
    teacher = {name: np.ones((2, off.n), np.float32) for name in ("scores", "safe", "maximum")}
    predictor = SimpleNamespace(model=torch.nn.Linear(4, 4), temperatures=np.ones(5))
    splits = {"policy": [0], "dev": [1]}
    assert (a.decision_dataset_fingerprint(records, splits, teacher, predictor, off) !=
            a.decision_dataset_fingerprint(records, splits, teacher, predictor, on))
    assert a.rules_fingerprint(off) != a.rules_fingerprint(on)
    with pytest.raises(AssertionError):
        a.Config(reachable_only=1).validate()


@pytest.mark.parametrize("max_cost", [None, 70.])
def test_reachable_only_graph_is_the_runtime_part_of_the_full_graph(max_cost):
    cfg = a.Config(k=3, zero_shots=2, repeats=2, max_cost=max_cost, reachable_only=True)
    full = a.decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost)
    graph = a.decision_graph(cfg.zero_shots, cfg.k, cfg.repeats, cfg.cost_unit, cfg.max_cost, True)
    assert set(graph.states) == set(a.reachable_states(cfg)) and len(graph.states) < len(full.states)
    index = {state: i for i, state in enumerate(full.states)}
    assert [index[state] for state in graph.states] == sorted(index[state] for state in graph.states)  # Catalog order.
    for i, state in enumerate(graph.states):
        j = index[state]
        assert graph.legal[i].tolist() == full.legal[j].tolist() and np.array_equal(graph.delta[i], full.delta[j])
        assert [graph.states[s] if s >= 0 else None for s in graph.succ[i]] == [
            full.states[s] if s >= 0 else None for s in full.succ[j]]
    assert a.decision_case_count(cfg) == 2 * len(graph.states)
    assert a.decision_valid(np.arange(a.decision_case_count(cfg)), cfg).shape == (2 * len(graph.states), cfg.action_count)
    if max_cost is None:
        assert len(graph.states) == 80 and graph.states[graph.full] == a.State((1 << cfg.n) - 1, (1 << cfg.k) - 1)


def prefix_lights(cfg, record, state):
    """Rank one (OS1) needs an evaluator; ZS1 and ZS2 then hold ranks two and three once present.

    Under the swapped zero-shot order nothing lights, so no goal can be reached there.
    """
    if record.candidate_ids[0] != "zs_0":
        return np.zeros(3 * cfg.n + 2, np.float32)
    p = recognizes(cfg, state, 2, needs_evaluator=True)
    if state.candidate_mask & 1:
        p[0] = .5
    return p


@pytest.mark.parametrize("has_max", [True, False])
def test_reachable_only_rows_are_the_routed_cases_the_runtime_asks_about(has_max):
    plans = {}
    for reachable_only in (False, True):
        cfg, record, scores, safe, maximum, predictor = tiny_problem(prefix_lights, reachable_only=reachable_only)
        if not has_max:
            safe = maximum = np.zeros(cfg.n, np.float32)
        plans[reachable_only] = (cfg, a.question_plan(record, scores, safe, maximum, predictor, cfg))
    (_, every), (cfg, plan) = plans[False], plans[True]
    states = plan["graph"].states
    assert set(states) == set(a.reachable_states(cfg)) and len(states) == 8 < len(every["graph"].states) == 14
    # On the reachable states, goals, routes, and statuses are those of the full catalog.
    index = [{s: j for j, s in enumerate(every["graph"].states)}[s] for s in states]
    for key in ("achieved", "status", "informative", "options", "leader"):
        assert np.array_equal(plan[key], every[key][:, index]), key
    # Rows: routed cases only, and with a MAX candidate only before its first goal, the runtime's stop test.
    expected = plan["informative"] & ~((plan["achieved"] >= 1) & has_max)
    assert np.array_equal(plan["rows"], expected) and plan["rows"].any()
    assert (plan["targets"][plan["rows"]] == plan["options"][plan["rows"]]).all() and not plan["targets"][~plan["rows"]].any()
    # The comparison is not vacuous: the full catalog labels cases this drops.
    assert (every["rows"][:, index] & ~every["informative"][:, index]).any()  # No preference.
    assert (plan["informative"] & (plan["achieved"] >= 1)).any()              # Past the first goal.
    coverage = a.plan_coverage(plan, cfg)
    assert coverage["state_order_cases"] == 28 and coverage["status_counts"]["UNVISITED"] == 12
    assert coverage["action_rows"] == coverage["informative_rows"] == plan["rows"].sum()
    assert set(coverage["trained_status_counts"]) == {"ACTION"}


def test_reachable_only_question_without_rows_round_trips_its_labels():
    cfg, record, scores, safe, maximum, predictor = tiny_problem(
        lambda c, r, s: np.zeros(3 * c.n + 2, np.float32), reachable_only=True)
    labels = a.build_route_labels(record, scores, safe, maximum, predictor, cfg)
    assert labels["options"].shape == (a.decision_case_count(cfg), 1) == (16, 1) and not labels["options"].any()
    assert labels["coverage"]["action_rows"] == 0
    assert labels["coverage"]["status_counts"] == {"UNREACHABLE": 14, "EXHAUSTED": 2, "UNVISITED": 12}
    assert len(a.check_route_labels(record, labels, cfg)) == 0
    rows = a.decision_training_rows(record, np.zeros(0, np.int64), np.zeros((0, 1), np.uint8), "pretrain",
                                    a.FrozenPredictor({"weights": a.cpu_state(a.ResNet(cfg.input_dim, 3 * cfg.n + 2, cfg)),
                                                       "temperatures": np.ones(5)}, cfg, torch.device("cpu")), cfg)
    assert rows["h"].shape == (0, cfg.head_input_dim) and rows["valid"].shape == (0, cfg.action_count)
    assert a.masked_action_loss(a.SupervisedActionHead(cfg), rows, [], cfg, "cpu") == (0., 0)
