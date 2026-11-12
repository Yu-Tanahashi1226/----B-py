"""Regression cases for legal resets, held evidence, and paired evaluation."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json, sha256, write_jsonl
from gating import signature, summarize, make_gate
from scoreboard_rules import RuleFilter, filter_observations
from scoreboard_rule_report import csv_out, evaluate, run_validation


def observation(index, count=(2, 2, 1), **changes):
    row = dict(frame=index * 15, time_sec=index * .5, board={'box': [700, 750, 950, 870]},
               board_layout_status='supported_corner_layout',
               count=copy.deepcopy(count) if isinstance(count, dict) or count is None else dict(zip(('balls', 'strikes', 'outs'), count)),
               runners=dict(first=False, second=False, third=False),
               scoreboard=dict(inning=1, half='top', upper_score=1, lower_score=0,
                               upper_team='関学', lower_team='京大'), ocr_status='inferred')
    for key, value in changes.items():
        if key in row['scoreboard']:
            row['scoreboard'][key] = value
        else:
            row[key] = value
    return row


class RuleTests(unittest.TestCase):
    def warm(self, count=(2, 2, 1), **changes):
        validator = RuleFilter()
        for i in range(4):
            last = validator.apply(observation(i, count, **changes))
        self.assertEqual(signature(last), count)
        return validator

    def test_ball_or_strike_decrease_is_held(self):
        for changed in ((1, 2, 1), (2, 1, 1)):
            v = self.warm()
            row = v.apply(observation(4, changed))
            self.assertEqual(row['count'], dict(balls=2, strikes=2, outs=1))
            self.assertIsNone(signature(row))
            self.assertEqual(row['rule_validation']['fields']['balls']['reason'],
                             'count_decreased_without_batter_change_evidence')

    def test_impossible_changes_hold_last_confirmed_value(self):
        for changes, field, expected in (({'count': dict(balls=2, strikes=2, outs=0)}, 'outs', 1),
                                         ({'upper_score': 0}, 'upper_score', 1),
                                         ({'upper_team': '阪神'}, 'upper_team', '関学'),
                                         ({'inning': 10}, 'inning', 1)):
            v = self.warm()
            for i in range(4, 8):
                row = v.apply(observation(i, **changes))
            location = row['count'] if field == 'outs' else row['scoreboard']
            self.assertEqual(location[field], expected)
            self.assertEqual(row['rule_validation']['fields'][field]['status'], 'held')

    def test_out_after_batter_ends_allows_count_reset(self):
        v = self.warm()
        for i in range(4, 8):
            row = v.apply(observation(i, (0, 0, 2)))
        self.assertEqual(signature(row), (0, 0, 2))

    def test_new_runner_or_run_allows_new_batter(self):
        for changes in ({'runners': dict(first=True, second=False, third=False)}, {'upper_score': 2}):
            v = self.warm()
            for i in range(4, 9):
                row = v.apply(observation(i, (0, 0, 1), **changes))
            self.assertEqual(signature(row), (0, 0, 1))

    def test_old_batter_change_evidence_does_not_authorize_future_decrease(self):
        v = self.warm((2, 0, 0))
        for i in range(4, 8):
            v.apply(observation(i, (2, 0, 1)))
        row = v.apply(observation(30, (1, 0, 1)))
        self.assertEqual(row['count']['balls'], 2)

    def test_next_half_inning_allows_outs_reset_and_backwards_does_not(self):
        v = self.warm((3, 2, 2))
        for i in range(4, 9):
            row = v.apply(observation(i, (0, 0, 0), half='bottom'))
        self.assertEqual(signature(row), (0, 0, 0))
        self.assertEqual(row['scoreboard']['half'], 'bottom')
        row = v.apply(observation(9, (0, 0, 0), half='top'))
        self.assertEqual(row['scoreboard']['half'], 'bottom')
        self.assertIsNone(signature(row))

    def test_sparse_sampling_gap_can_resynchronize_confirmed_inning(self):
        v = self.warm()
        for i in (100, 101, 102, 103, 104):
            row = v.apply(observation(i, (0, 0, 0), inning=5))
        self.assertEqual(row['scoreboard']['inning'], 5)
        self.assertEqual(signature(row), (0, 0, 0))

    def test_count_reset_may_follow_inning_text_update_after_a_short_delay(self):
        v = self.warm((3, 2, 2))
        for i in range(4, 7):
            v.apply(observation(i, (3, 2, 2), half='bottom'))
        for i in range(7, 12):
            row = v.apply(observation(i, (0, 0, 0), half='bottom'))
        self.assertEqual(signature(row), (0, 0, 0))

    def test_all_lamps_unreadable_after_inning_change_does_not_lose_boundary(self):
        v = self.warm((3, 2, 2))
        for i in range(4, 7):
            v.apply(observation(i, (3, 2, 2), half='bottom'))
        for i in range(7, 50):
            held = v.apply(observation(i, count=None, half='bottom'))
            self.assertIsNone(signature(held))
        for i in range(50, 55):
            row = v.apply(observation(i, (1, 0, 0), half='bottom'))
        self.assertEqual(signature(row), (1, 0, 0))

    def test_missing_count_is_held_but_never_fresh_gate_evidence(self):
        v = self.warm()
        row = v.apply(observation(4, count=None))
        self.assertEqual(row['count']['balls'], 2)
        self.assertIsNone(signature(row))

    def test_held_score_context_is_not_summarized_as_a_fresh_reading(self):
        v = self.warm()
        row = v.apply(observation(4, upper_team='誤読'))
        self.assertEqual(signature(row), (2, 2, 1))
        row['rule_validation']['fields']['upper_score']['status'] = 'held'
        self.assertIsNone(summarize([row])['upper_score'])

    def test_cached_ocr_cannot_confirm_initial_score_or_team(self):
        v = RuleFilter()
        v.apply(observation(0))
        for i in range(1, 5):
            row = v.apply(observation(i, ocr_status='same_visible_block_cached'))
        self.assertIsNone(row['scoreboard']['upper_team'])
        self.assertIsNone(row['scoreboard']['upper_score'])
        row = v.apply(observation(5))
        self.assertEqual(row['scoreboard']['upper_team'], '関学')

    def test_nonconsecutive_changed_value_does_not_meet_confirmation(self):
        v = self.warm((1, 2, 1))
        v.apply(observation(4, (2, 2, 1)))
        v.apply(observation(5, (1, 2, 1)))
        row = v.apply(observation(6, (2, 2, 1)))
        self.assertEqual(row['count']['balls'], 1)
        self.assertIsNone(signature(row))

    def test_runner_drop_without_out_or_score_is_held(self):
        v = self.warm(runners=dict(first=True, second=False, third=False))
        for i in range(4, 8):
            row = v.apply(observation(i))
        self.assertTrue(row['runners']['first'])
        self.assertIsNone(signature(row))

    def test_unchanged_two_strike_foul_still_selected_and_held_only_is_not(self):
        rows = []
        for i in range(35):
            row = observation(i)
            if 5 <= row['time_sec'] < 7:
                row.update(board=None, count=None, runners=None, scoreboard=None)
            rows.append(row)
        filtered, _ = filter_observations(rows)
        self.assertEqual(make_gate(filtered, 30, 525)['selected_episodes'], 1)
        for row in rows:
            if row['time_sec'] >= 7:
                row['count'] = None
        filtered, _ = filter_observations(rows)
        self.assertEqual(make_gate(filtered, 30, 525)['selected_episodes'], 0)

    def test_raw_input_is_unchanged(self):
        rows = [observation(i) for i in range(6)]
        original = copy.deepcopy(rows)
        filter_observations(rows)
        self.assertEqual(rows, original)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = dict(video='test.mp4', sha256='abc')

    def truth(self, rows):
        path = self.root / 'truth.csv'
        csv_out(path, rows, ['video', 'source_sha256', 'frame', 'true_balls'])
        return path

    def test_same_ground_truth_and_missing_output_counted_in_both_variants(self):
        path = self.truth([dict(video='test.mp4', source_sha256='abc', frame=0, true_balls=1),
                           dict(video='test.mp4', source_sha256='abc', frame=15, true_balls=2)])
        a = [observation(0, (0, 2, 1))]
        b = [observation(0, (1, 2, 1))]
        results = evaluate(a, b, path, self.source)
        before, after = [next(r for r in results if r['field'] == 'balls' and r['variant'] == v) for v in ('before', 'after')]
        self.assertEqual((before['tp'], before['fp'], before['fn']), (0, 1, 2))
        self.assertEqual((after['tp'], after['fp'], after['fn']), (1, 0, 1))
        self.assertEqual(after['precision'], 1)
        self.assertEqual(after['recall'], .5)
        self.assertEqual(before['labeled_values'], after['labeled_values'])

    def test_no_ground_truth_does_not_fabricate_metrics(self):
        for row in evaluate([observation(0)], [observation(0)], None, self.source):
            self.assertIsNone(row['precision'])
            self.assertIsNone(row['recall'])

    def test_ground_truth_for_different_video_or_same_name_wrong_hash_rejected(self):
        for identity in ({'video': 'other.mp4'}, {'video': 'test.mp4', 'source_sha256': 'wrong'}):
            path = self.truth([dict(identity, frame=0, true_balls=1)])
            with self.assertRaises(ValueError):
                evaluate([], [], path, self.source)

    def test_conflicting_duplicate_ground_truth_rejected(self):
        path = self.truth([dict(video='test.mp4', frame=0, true_balls=v) for v in (1, 2)])
        with self.assertRaises(ValueError):
            evaluate([], [], path, self.source)

    def test_offline_cli_preserves_raw_and_existing_human_annotations(self):
        input_dir, output = self.root / 'input', self.root / 'review'
        input_dir.mkdir()
        source = input_dir / 'observations.jsonl'
        write_jsonl(source, [observation(i) for i in range(8)])
        save_json(input_dir / 'job.json', self.source)
        original_hash = sha256(source)
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'validate_scoreboard.py'),
                   '--observations', str(source), '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertIn('review.html', result.stdout)
        with (output / 'ground_truth_template.csv').open(encoding='utf-8-sig', newline='') as stream:
            annotations = list(csv.DictReader(stream))
        self.assertTrue(all(not row['true_balls'] for row in annotations))
        annotations[0]['true_balls'] = '2'
        csv_out(output / 'ground_truth_template.csv', annotations, list(annotations[0]))
        annotation_hash = sha256(output / 'ground_truth_template.csv')
        subprocess.run(command + ['--ground-truth', str(output / 'ground_truth_template.csv')],
                       capture_output=True, text=True, check=True)
        self.assertEqual(sha256(source), original_hash)
        self.assertEqual(sha256(output / 'ground_truth_template.csv'), annotation_hash)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['evaluation_status'], 'evaluated')
        write_jsonl(source, [observation(i) for i in range(9)])
        with self.assertRaises(ValueError):
            run_validation(dict(self.source, output=str(input_dir)), output)


if __name__ == '__main__':
    unittest.main()
