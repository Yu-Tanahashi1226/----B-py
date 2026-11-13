import csv
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json, sha256, write_jsonl
from make_manual_review import generate_review
from scoreboard_rule_report import evaluate, csv_out
from test_scoreboard_rules import observation


def page_data(out):
    text = (out / 'index.html').read_text(encoding='utf-8')
    return json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', text, re.S).group(1))


class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.board = self.root / 'analysis' / 'scoreboard'
        self.board.mkdir(parents=True)
        self.raw = [observation(i) for i in range(10)]
        write_jsonl(self.board / 'observations.jsonl', self.raw)
        save_json(self.board / 'job.json', dict(video=r'C:\input\test.mp4', sha256='abc', config={}))
        self.events = [dict(event_id=1, reference='example.mp4', pitch_index=1, start_sec=2, end_sec=3,
                            count_gate=dict(before=dict(last_sec=1.5), after=dict(last_sec=4.5)),
                            pitch_model_score=.9, foul_model_score=.7, foul_model_decision=True,
                            model_start_sec=2.1, model_end_sec=2.9, status='model_supported_foul_candidate'),
                       dict(event_id=2, reference='example.mp4', pitch_index=2, start_sec=30, end_sec=33,
                            count_gate=dict(before=None, after=None), pitch_model_score=None,
                            foul_model_score=None, status='after_bso_unknown')]
        save_json(self.root / 'analysis' / 'events.json', self.events)

    def read_csv(self, path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_paired_predictions_and_blank_truth_without_changing_source(self):
        before = sha256(self.board / 'observations.jsonl')
        out = generate_review(self.root)
        states, pitches = [self.read_csv(out / n) for n in ('scoreboard_review.csv', 'pitch_review.csv')]
        self.assertEqual(states[4]['pred_balls'], '2')
        self.assertEqual(states[4]['recorded_balls'], '2')
        self.assertEqual(pitches[0]['before_frame'], '45')
        self.assertEqual(pitches[0]['after_frame'], '135')
        self.assertEqual(pitches[0]['pitch_model_score'], '0.9')
        self.assertTrue(all(v == '' for rows in (states, pitches) for row in rows for k, v in row.items() if k.startswith('true_')))
        self.assertEqual(sha256(self.board / 'observations.jsonl'), before)

    def test_no_nearby_sample_or_missing_model_prediction_is_invented(self):
        self.events[0]['count_gate']['before'] = dict(last_sec=1.8)
        save_json(self.root / 'analysis' / 'events.json', self.events)
        out = generate_review(self.root)
        data = page_data(out)
        self.assertIsNone(data['events'][0]['snapshots']['before'])
        self.assertIsNone(data['events'][1]['pitch_score'])
        self.assertIsNone(data['events'][1]['predictions']['pitch_present'])
        self.assertIsNone(data['events'][1]['predictions']['foul'])

    def test_zero_score_is_preserved_as_a_real_output(self):
        self.events[0]['pitch_model_score'] = 0
        save_json(self.root / 'analysis' / 'events.json', self.events)
        out = generate_review(self.root)
        data = page_data(out)
        self.assertEqual(data['events'][0]['pitch_score'], 0)
        self.assertIs(data['events'][0]['predictions']['pitch_present'], False)

    def test_existing_human_csv_is_preserved_and_loaded_in_page(self):
        out = generate_review(self.root)
        path = out / 'scoreboard_review.csv'
        rows = self.read_csv(path)
        rows[3].update(true_balls='1', true_runner_first='true', note='実際は1B,確認済み')
        csv_out(path, rows, list(rows[0]))
        before = sha256(path)
        generate_review(self.root)
        self.assertEqual(sha256(path), before)
        data = page_data(out)
        self.assertEqual(data['state_labels']['45']['true_balls'], '1')
        self.assertEqual(data['state_labels']['45']['true_runner_first'], '1')
        self.assertEqual(data['state_labels']['45']['note'], '実際は1B,確認済み')

    def test_pitch_csv_snapshot_truth_is_shared_with_frame_truth(self):
        out = generate_review(self.root)
        path = out / 'pitch_review.csv'
        rows = self.read_csv(path)
        rows[0].update(true_before_balls='1', true_foul='0', note='ファウルではない')
        csv_out(path, rows, list(rows[0]))
        before = sha256(path)
        generate_review(self.root)
        data = page_data(out)
        self.assertEqual(data['state_labels']['45']['true_balls'], '1')
        self.assertEqual(data['event_labels']['1']['true_foul'], '0')
        self.assertEqual(sha256(path), before)

    def test_conflicting_frame_and_pitch_labels_are_rejected(self):
        out = generate_review(self.root)
        for name, key, value in [('scoreboard_review.csv', 'true_balls', '1'), ('pitch_review.csv', 'true_before_balls', '2')]:
            path = out / name
            rows = self.read_csv(path)
            rows[3 if name.startswith('scoreboard') else 0][key] = value
            csv_out(path, rows, list(rows[0]))
        with self.assertRaises(ValueError):
            generate_review(self.root)

    def test_embedded_ocr_is_not_html_or_executable_script(self):
        attack = '</script><img src=x onerror=alert(1)>'
        self.raw[0]['scoreboard']['raw'] = {'upper_team': attack}
        write_jsonl(self.board / 'observations.jsonl', self.raw)
        out = generate_review(self.root)
        self.assertNotIn(attack, (out / 'index.html').read_text())
        self.assertEqual(page_data(out)['observations'][0][5]['upper_team'], attack)

    def test_different_source_cannot_reuse_human_annotations(self):
        out = generate_review(self.root)
        save_json(self.board / 'job.json', dict(video='test.mp4', sha256='other', config={}))
        with self.assertRaises(ValueError):
            generate_review(self.root, out)

    def test_new_state_csv_is_compatible_with_existing_evaluator(self):
        out = generate_review(self.root)
        path = out / 'scoreboard_review.csv'
        rows = self.read_csv(path)
        rows[4]['true_balls'] = '2'
        csv_out(path, rows, list(rows[0]))
        metrics = evaluate(self.raw, self.raw, path, {'video':'test.mp4', 'sha256':'abc'})
        result = next(m for m in metrics if m['variant'] == 'after' and m['field'] == 'balls')
        self.assertEqual(result['precision'], 1)
        self.assertEqual(result['labeled_values'], 1)

    def test_no_completed_pitch_results_still_exports_yolo_and_ocr(self):
        (self.root / 'analysis' / 'events.json').unlink()
        out = generate_review(self.root)
        self.assertEqual(page_data(out)['events'], [])
        self.assertEqual(len(self.read_csv(out / 'scoreboard_review.csv')), 10)


if __name__ == '__main__':
    unittest.main()
