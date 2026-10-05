"""Reporting distinguishes unknowns, matched cohorts, failures, and recorded costs."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.finetuning_reporting import (
    format_benchmark_overview, format_merging_report, format_simplification_report,
)
from src.merging_evaluation_reporting import benchmark_metrics, rebuild_legacy_two_phase_summary
from src.simplification_finetuning import summarize_evaluation


def judgment(correct, status='SUCCESS'):
    return {'status': status, 'is_correct': correct}


def case(index, direct, base, adapted):
    return {
        'record_id': str(index), 'label_kind': 'unlabeled', 'solver_status': 'PARTIAL',
        'ground_truth': '42', 'direct': {'evaluation': judgment(direct),
            'solution': {'status': 'SUCCESS', 'input_tokens': 10, 'output_tokens': 5}},
        'arms': {arm: {'evaluation': judgment(value), 'exact_copy': True,
            'normalized_copy': True, 'solver_status': 'REUSED_DIRECT',
            'simplification': {'status': 'SUCCESS', 'input_tokens': 2, 'output_tokens': 1}}
            for arm, value in (('base', base), ('adapted', adapted))},
    }


class ReportTests(TestCase):
    def test_legacy_summary_cell_recovers_saved_runs_without_notebook_globals(self):
        runs = []
        for population, base, adapted in (('heldout_accepted', False, True), ('remaining_benchmark', True, False)):
            for arm, correct in (('base', base), ('adapted', adapted)):
                runs.append({'benchmark': 'numina_hard', 'benchmark_index': 0,
                             'population': population, 'phase': 'phase_1',
                             'candidate_source': 'one_shot', 'candidate_count': 2,
                             'mode': 'pair_fusion', 'arm': arm,
                             'tree': {'status': 'SUCCESS', 'root_node_id': 'root', 'trace': []},
                             'evaluation': {'root_correct': correct}})
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            for population in ('heldout_accepted', 'remaining_benchmark'):
                (root / f'phase_1_{population}_results.json').write_text(json.dumps(
                    [run for run in runs if run['population'] == population]))
            # Aggregate copies must not double the sample size.
            (root / 'phase_1_results.json').write_text(json.dumps(runs))
            notebook = json.loads(Path('merging_finetuning.ipynb').read_text(encoding='utf-8'))
            cell = next(''.join(c['source']) for c in notebook['cells']
                        if c['cell_type'] == 'code' and 'report_index =' in ''.join(c['source']))
            namespace = {'WORK_DIR': root, 'SEED': 42}
            exec(cell, namespace)
            self.assertNotIn('all_runs', namespace)
            summary = json.loads((root / 'two_phase_evaluation_summary.json').read_text())
            self.assertEqual(summary['phase_1_runs'], 4)
            self.assertEqual(summary['phase_2_runs'], 0)
            pairs = summary['paired_adapted_minus_base'].values()
            self.assertEqual(sorted(pair['accuracy_delta'] for pair in pairs), [-1, 1])
            self.assertTrue(all(pair['paired_questions'] == 1 for pair in pairs))
            self.assertIn('Corrections', (root / 'evaluation_report.txt').read_text())

    def test_conflicting_legacy_copies_are_rejected(self):
        run = {'benchmark': 'numina_hard', 'population': 'heldout_accepted',
               'phase': 'phase_1', 'benchmark_index': 0, 'candidate_source': 'one_shot',
               'candidate_count': 2, 'mode': 'pair_fusion', 'arm': 'base'}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'phase_1_heldout_accepted_results.json').write_text(json.dumps([run]))
            (root / 'phase_1_results.json').write_text(json.dumps([{**run, 'changed': True}]))
            with self.assertRaisesRegex(ValueError, 'Conflicting saved legacy runs'):
                rebuild_legacy_two_phase_summary(root)

    def test_simplifier_pairwise_cohorts_and_unknowns_are_not_counted_incorrect(self):
        rows = [case(0, True, False, True), case(1, None, True, False),
                case(2, True, None, True), case(3, None, None, None)]
        rows[3]['arms']['adapted']['evaluation'] = judgment(True, 'API_FAILURE')
        report = {'benchmark': 'math500', 'population_audit': {}, 'question_limit': None,
                  **summarize_evaluation(rows)}
        solver = report['solver']
        self.assertEqual(solver['paired_judged'], 1)
        self.assertEqual(solver['arms']['adapted']['judged'], 3)
        self.assertEqual(solver['arms']['adapted']['unknown'], 1)
        self.assertEqual(solver['arms']['adapted']['accuracy'], 2 / 3)
        paired = solver['pairwise']['adapted_minus_base']
        self.assertEqual(paired['paired_questions'], 2)
        self.assertEqual(paired['base_incorrect_adapted_correct'], 1)
        self.assertEqual(paired['base_correct_adapted_incorrect'], 1)
        self.assertEqual(paired['accuracy_delta'], 0)
        self.assertEqual(solver['pairwise']['adapted_minus_direct']['paired_questions'], 2)
        self.assertEqual(report['generation']['direct']['recorded_calls'], 4)
        self.assertEqual(report['generation']['adapted']['recorded_calls'], 4,
                         'Reused direct solutions must not be recounted as calls.')
        text = format_simplification_report(report)
        for label in ('66.67%', 'Unknown', 'Coverage', 'Bootstrap 95% CI', 'Corrections',
                      'Regressions', 'Simplifier behavior', 'Failures and reuse', 'API_FAILURE',
                      'Calls with metadata', 'REUSED_DIRECT'):
            self.assertIn(label, text)

    def test_merging_partial_report_uses_actual_question_count_and_paired_delta(self):
        runs = []
        for index, base, adapted in ((0, False, True), (1, None, True)):
            for arm, correct in (('base', base), ('adapted', adapted)):
                runs.append({'benchmark_index': index, 'phase': 'phase_1', 'candidate_source': 'one_shot',
                             'candidate_count': 2, 'mode': 'pair_fusion', 'arm': arm,
                             'tree': {'status': 'SUCCESS', 'root_node_id': 'root', 'trace': []},
                             'evaluation': {'root_correct': correct, 'judge_status': {'root': 'API_FAILURE' if correct is None else 'SUCCESS'}}})
        report = benchmark_metrics('math500', runs, [], {'eligible_external_questions': 100},
                                   eval_mode='minimal', question_limit=10)
        self.assertEqual(report['selected_questions'], 10)
        self.assertEqual(report['evaluated_questions'], 2)
        pair = next(iter(report['paired_adapted_minus_base'].values()))
        self.assertEqual(pair['paired_questions'], 1)
        self.assertEqual(pair['accuracy_delta'], 1)
        text = format_merging_report(report)
        for label in ('+100.00 pp', '50.00%', 'Unknown', 'API_FAILURE', 'shared candidate', 'final answers only'):
            self.assertIn(label, text)
        self.assertIn('math500', format_benchmark_overview([report], 'merging'))

    def test_empty_reports_print_unknown_accuracy_without_crashing(self):
        report = benchmark_metrics('aime26', [], [], {'eligible_external_questions': 0},
                                   eval_mode='full', question_limit=None)
        self.assertIn('aime26', format_merging_report(report))
        report = {'benchmark': 'aime26', 'question_limit': None, **summarize_evaluation([])}
        self.assertIn('n/a', format_simplification_report(report))
        self.assertIn('no pooled average', format_benchmark_overview([], 'merging'))
