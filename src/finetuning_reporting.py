"""Readable benchmark comparisons; rendering never invokes models or judges."""

import json


def percent(value):
    return 'n/a' if value is None else f'{value:.2%}'


def delta(value):
    return 'n/a' if value is None else f'{100 * value:+.2f} pp'


def table(headers, rows):
    rows = [[str(value) for value in row] for row in rows]
    widths = [max([len(str(header)), *(len(row[i]) for row in rows)]) for i, header in enumerate(headers)]
    def line(row):
        return ' | '.join(str(value).ljust(width) for value, width in zip(row, widths))
    return '\n'.join([line(headers), '-+-'.join('-' * width for width in widths), *map(line, rows)])


def paired_row(name, paired):
    ci = paired.get('bootstrap_95_ci')
    return [name, paired.get('paired_questions', 0), percent(paired.get('base_accuracy')),
            percent(paired.get('adapted_accuracy')), delta(paired.get('accuracy_delta')),
            f'[{delta(ci[0])}, {delta(ci[1])}]' if ci else 'n/a',
            paired.get('base_incorrect_adapted_correct', 0),
            paired.get('base_correct_adapted_incorrect', 0)]


def population_lines(report, count):
    audit = report.get('population_audit', {})
    limit = report.get('question_limit')
    return [f"Benchmark: {report['benchmark']} | reported questions: {count} | limit: {'all' if limit is None else limit}",
            'Population audit: ' + json.dumps(audit, sort_keys=True),
            'Accuracy uses successful known judgments; unknown outcomes are excluded, with coverage shown.']


def format_merging_report(report):
    lines = population_lines(report, report['evaluated_questions'])
    lines.append(f"Protocol: {report['evaluation_mode']} | phase 1 runs: {report['phase_1_runs']} | phase 2 runs: {report['phase_2_runs']}")
    lines.append(f"Selected questions: {report.get('selected_questions', 'n/a')}; questions represented by saved runs: {report['evaluated_questions']}.")
    rows, resources, failures = [], [], []
    for name, metric in report['summaries'].items():
        rows.append([name, metric['runs'], metric['evaluated_roots'], metric.get('correct_roots', 'n/a'),
                     metric.get('unknown_roots', metric['runs'] - metric['evaluated_roots']),
                     percent(metric['accuracy_on_evaluated']), percent(metric['evaluation_coverage']), metric['incomplete_trees']])
        resources.append([name, metric['generated_nodes'], metric['input_tokens'], metric['output_tokens'], f"{metric['elapsed_seconds']:.2f}"])
        failures.append(f"{name}: root judgment statuses {json.dumps(metric.get('root_judge_status_counts', {}), sort_keys=True)}")
    lines += ['\nFinal-answer comparison', table(['Mode / arm', 'Runs', 'Judged', 'Correct', 'Unknown', 'Accuracy', 'Coverage', 'Incomplete'], rows),
              '\nPaired comparisons (same questions in both arms)',
              table(['Mode', 'Pairs', 'Base', 'Adapted', 'Delta', 'Bootstrap 95% CI', 'Corrections', 'Regressions'],
                    [paired_row(name, paired) for name, paired in report['paired_adapted_minus_base'].items()]),
              'Delta is adapted minus base. CI is a paired question bootstrap; small samples have limited precision.',
              '\nRecorded trace resources', table(['Mode / arm', 'Nodes', 'Input tokens', 'Output tokens', 'Seconds'], resources),
              'Trace totals include shared candidate leaves and reused tree nodes; do not sum rows as actual execution cost. Times are recorded generation times, not end-to-end wall time.',
              '\nJudgment status counts', *failures]
    if report['evaluation_mode'] == 'full':
        diagnostics = []
        for name, metric in report['summaries'].items():
            lines.append(f"{name}: candidate accuracy={percent(metric.get('candidate_accuracy'))} on {metric.get('candidate_judged', 0)}/{metric.get('candidate_nodes', 0)} nodes; parent-child transitions={json.dumps(metric.get('parent_to_child_transitions', {}), sort_keys=True)}")
            for kind, values in metric.get('root_accuracy_by_candidate_stratum', {}).items():
                if values['runs']:
                    diagnostics.append([name, 'root given ' + kind, values['runs'], percent(values['accuracy'])])
            for layer, values in metric.get('fusion_accuracy_by_layer', {}).items():
                diagnostics.append([name, 'fusion layer ' + layer, values['nodes'], percent(values['accuracy'])])
        lines += ['\nCandidate and fusion diagnostics (counts are judged roots or nodes)', table(['Mode / arm', 'Diagnostic', 'Count', 'Accuracy'], diagnostics)]
    else:
        lines.append('Minimal protocol judges final answers only; candidate/layer correctness is unavailable.')
    return '\n'.join(lines)


def format_simplification_report(report):
    lines = population_lines(report, report['questions'])
    solver = report['solver']
    rows = []
    for arm, metric in solver['arms'].items():
        rows.append([arm, metric['questions'], metric['judged'], metric['correct'], metric['unknown'], percent(metric['accuracy']), percent(metric['coverage'])])
    lines += ['Fixed base solver: direct = original question; base/adapted = corresponding simplifier then assisted original-question solving.',
              '\nSolver final-answer comparison', table(['Arm', 'Questions', 'Judged', 'Correct', 'Unknown', 'Accuracy', 'Coverage'], rows),
              f"Common three-arm cohort: {solver['paired_judged']} questions; direct={percent(solver['accuracy']['direct'])}; base={percent(solver['accuracy']['base'])}; adapted={percent(solver['accuracy']['adapted'])}.",
              '\nPaired comparisons (matched known judgments for each contrast)',
              table(['Contrast', 'Pairs', 'Baseline', 'Adapted', 'Delta', 'Bootstrap 95% CI', 'Corrections', 'Regressions'],
                    [paired_row(name, value) for name, value in solver['pairwise'].items()]),
              'Deltas are percentage points; each contrast can have a different paired cohort. CI is a paired question bootstrap; small samples have limited precision.']
    behavior = []
    for kind, values in report['behavior'].items():
        if values['questions']:
            for arm in ('base', 'adapted'):
                m = values[arm]
                behavior.append([kind, arm, values['questions'], m['generated'], values['questions'] - m['generated'], percent(m['exact_copy_rate']), percent(m['normalized_copy_rate']), percent(m['change_rate'])])
    lines += ['\nSimplifier behavior', table(['Label', 'Arm', 'Questions', 'Generated', 'Unknown', 'Exact copy', 'Normalized copy', 'Changed'], behavior),
              'Copy/change rates use successful simplifications. External questions are unlabeled: changing a question is not evidence of preserving its meaning.',
              '\nRecorded generation resources', table(['Arm', 'Calls', 'Calls with metadata', 'Input tokens', 'Output tokens', 'Seconds'],
                    [[arm, m['recorded_calls'], m['resource_metadata_calls'], m['input_tokens'], m['output_tokens'], f"{m['elapsed_seconds']:.2f}"] for arm, m in report['generation'].items()]),
              'Calls are recorded outputs, excluding evaluator calls and retries. Missing resource metadata contributes zero; these totals are not a full provider bill. Reused direct solves are counted once.',
              '\nFailures and reuse', 'Question solver statuses: ' + json.dumps(report['solver_status_counts'], sort_keys=True)]
    for arm, metric in report['generation'].items():
        lines.append(f"{arm}: generation statuses={json.dumps(metric['status_counts'], sort_keys=True)}; solver statuses={json.dumps(metric['solver_status_counts'], sort_keys=True)}; judge statuses={json.dumps(solver['arms'][arm]['judge_status_counts'], sort_keys=True)}")
    return '\n'.join(lines)


def format_benchmark_overview(reports, kind):
    rows = []
    for report in reports:
        if kind == 'merging':
            for mode, paired in report['paired_adapted_minus_base'].items():
                rows.append([report['benchmark'], mode, *paired_row('', paired)[1:]])
        else:
            for contrast, paired in report['solver']['pairwise'].items():
                rows.append([report['benchmark'], contrast, *paired_row('', paired)[1:]])
    return 'Benchmark comparison (separate populations; no pooled average)\n' + table(
        ['Benchmark', 'Mode / contrast', 'Pairs', 'Baseline', 'Adapted', 'Delta', 'Bootstrap 95% CI', 'Corrections', 'Regressions'], rows)
