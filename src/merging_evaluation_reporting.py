"""Compute merging evaluation reports from explicit saved runs; no model calls."""

import numpy as np

from src.merging_finetuning import summarize_evaluated_runs


def diagnostic_summary(runs, eval_mode):
    result = summarize_evaluated_runs(runs)
    if eval_mode == 'minimal':
        result.pop('parent_to_child_transitions', None)
        return result  # only final answers were judged
    candidate_values, layer_values, candidate_nodes_count = [], {}, 0
    strata = {name: [] for name in (
        'both_correct', 'one_correct', 'neither_correct',
        'candidate_correct', 'candidate_incorrect',
        'all_correct', 'some_correct', 'none_correct', 'unknown',
    )}
    for run in runs:
        correctness = run['evaluation'].get('node_correctness', {})
        candidate_nodes = [
            node for node in run['tree'].get('trace', [])
            if node.get('kind', '').endswith('_candidate')
        ]
        values = [correctness.get(node['node_id']) for node in candidate_nodes]
        candidate_nodes_count += len(candidate_nodes)
        candidate_values.extend(value for value in values if value is not None)
        if values:
            if any(value is None for value in values):
                stratum = 'unknown'
            elif len(values) == 1:
                stratum = 'candidate_correct' if values[0] else 'candidate_incorrect'
            elif len(values) == 2:
                stratum = 'both_correct' if all(values) else (
                    'one_correct' if any(values) else 'neither_correct'
                )
            else:
                stratum = 'all_correct' if all(values) else (
                    'some_correct' if any(values) else 'none_correct'
                )
            root = run['evaluation'].get('root_correct')
            if root is not None:
                strata[stratum].append(root)
        for node in run['tree'].get('trace', []):
            if node.get('kind') == 'fusion':
                layer = int(node['node_id'].split('-')[1])
                value = correctness.get(node['node_id'])
                if value is not None:
                    layer_values.setdefault(layer, []).append(value)
    result['candidate_accuracy'] = (
        sum(candidate_values) / len(candidate_values) if candidate_values else None
    )
    result['candidate_nodes'] = candidate_nodes_count
    result['candidate_judged'] = len(candidate_values)
    result['root_accuracy_by_candidate_stratum'] = {
        name: {'runs': len(values), 'accuracy': sum(values) / len(values) if values else None}
        for name, values in strata.items()
    }
    result['fusion_accuracy_by_layer'] = {
        str(layer): {'nodes': len(values), 'accuracy': sum(values) / len(values)}
        for layer, values in sorted(layer_values.items())
    }
    return result

def paired_arm_effect(runs, seed=42, bootstrap_samples=10000):
    by_question = {}
    for run in runs:
        value = run.get('evaluation', {}).get('root_correct')
        if value is not None:
            by_question.setdefault(run['benchmark_index'], {})[run['arm']] = int(value)
    pairs = [arms for arms in by_question.values() if {'base', 'adapted'} <= arms.keys()]
    if not pairs:
        return {'paired_questions': 0, 'accuracy_delta': None, 'bootstrap_95_ci': None}
    differences = np.asarray([pair['adapted'] - pair['base'] for pair in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))].mean(axis=1)
    return {
        'paired_questions': len(pairs),
        'base_accuracy': float(np.mean([pair['base'] for pair in pairs])),
        'adapted_accuracy': float(np.mean([pair['adapted'] for pair in pairs])),
        'accuracy_delta': float(differences.mean()),
        'base_incorrect_adapted_correct': sum(pair['adapted'] > pair['base'] for pair in pairs),
        'base_correct_adapted_incorrect': sum(pair['adapted'] < pair['base'] for pair in pairs),
        'bootstrap_95_ci': [float(value) for value in np.quantile(boot, [0.025, 0.975])],
    }

def benchmark_metrics(benchmark_name, p1, p2, audit, *, eval_mode, question_limit, seed=42):
    runs = p1 + p2
    summaries, paired_effects = {}, {}
    fields = ('phase', 'candidate_source', 'candidate_count', 'mode')
    for key in sorted({tuple(run[field] for field in fields) for run in runs}):
        grouped = [run for run in runs if tuple(run[field] for field in fields) == key]
        group_name = '/'.join(map(str, key))
        for arm in ('base', 'adapted'):
            arm_runs = [run for run in grouped if run['arm'] == arm]
            if arm_runs:
                summaries[f'{group_name}/{arm}'] = diagnostic_summary(arm_runs, eval_mode)
        paired_effects[group_name] = paired_arm_effect(grouped, seed)
    result = {
        'benchmark': benchmark_name, 'population_audit': audit,
        'evaluation_mode': eval_mode,
        'question_limit': question_limit,
        'selected_questions': min(audit['eligible_external_questions'], question_limit)
            if question_limit is not None else audit['eligible_external_questions'],
        'evaluated_questions': len({run['benchmark_index'] for run in runs}),
        'phase_1_runs': len(p1), 'phase_2_runs': len(p2),
        'summaries': summaries, 'paired_adapted_minus_base': paired_effects,
    }
    if eval_mode == 'minimal':
        result['minimal_comparison'] = {
            'base': diagnostic_summary([run for run in p1 if run['arm'] == 'base'], eval_mode),
            'adapted': diagnostic_summary([run for run in p1 if run['arm'] == 'adapted'], eval_mode),
            'paired': paired_arm_effect(p1, seed),
        }
    return result

