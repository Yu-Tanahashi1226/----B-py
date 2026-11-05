import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gating import make_gate
from pitch_worker import clip_starts, validate_gate, run as run_pitch
from run_foul_pipeline import resolve_config
from report import build_events
from board_adapter import inner_state


def observations(before=(0,2,0), after=(0,2,0), gap=(5,7)):
    rows=[]
    for index in range(25):
        time=index*.5
        absent=gap[0] <= time < gap[1]
        state=before if time < gap[0] else after
        rows.append(dict(frame=index*15,time_sec=time,board=None if absent else {'box':[700,750,950,870]},
                         board_layout_status='no_detection' if absent else 'supported_corner_layout',
                         count=None if absent else dict(zip(['balls','strikes','outs'],state)),
                         runners=None if absent else dict(first=False,second=False,third=False),
                         scoreboard=None if absent else dict(inning=1,half='bottom',upper_score=0,lower_score=0)))
    return rows


class GateTests(unittest.TestCase):
    def gate(self, rows, **settings):
        return make_gate(rows,30,375,.5,settings)

    def test_only_equal_two_strike_gate(self):
        gate=self.gate(observations())
        self.assertEqual(gate['selected_episodes'],1)
        self.assertEqual(len(gate['windows']),1)
        window=gate['windows'][0]
        self.assertLess(window['start_sec'],5)
        self.assertGreaterEqual(window['end_sec'],7)
        self.assertLess(window['end_frame_exclusive'],375)

    def test_not_two_strikes(self):
        gate=self.gate(observations((0,1,0),(0,1,0)))
        self.assertEqual(gate['windows'],[])
        self.assertEqual(gate['episodes'][0]['reason'],'before_not_two_strikes')

    def test_ball_changed(self):
        self.assertEqual(self.gate(observations(after=(1,2,0)))['windows'],[])

    def test_out_and_bs_reset(self):
        self.assertEqual(self.gate(observations((2,2,0),(0,0,1)))['windows'],[])

    def test_no_disappearance(self):
        self.assertEqual(self.gate(observations(gap=(30,32)))['episodes'],[])

    def test_single_missing_sample_is_not_disappearance(self):
        self.assertEqual(self.gate(observations(gap=(5,5.5)))['windows'],[])

    def test_unsupported_layout_is_not_absence(self):
        rows=observations()
        for row in rows:
            if row['board'] is None:
                row['board']={'box':[0,0,200,100]}
                row['board_layout_status']='unsupported_layout'
        self.assertEqual(self.gate(rows)['episodes'],[])

    def test_visible_but_unreadable_is_not_absence(self):
        rows=observations(gap=(30,32))
        for row in rows[10:14]:row['count']=None
        self.assertEqual(self.gate(rows)['episodes'],[])

    def test_insufficient_pre_count_remains_unknown(self):
        rows=observations()
        for row in rows[:10]:row['count']=None
        gate=self.gate(rows)
        self.assertEqual(gate['windows'],[])
        self.assertEqual(gate['episodes'][0]['reason'],'before_bso_unreadable')

    def test_delayed_update_does_not_select_early_unchanged_display(self):
        rows=observations()
        for row in rows:
            if row['time_sec'] >= 10:row['count']['balls']=1
        gate=self.gate(rows)
        self.assertEqual(gate['windows'],[])
        self.assertEqual(gate['episodes'][0]['reason'],'bso_changed_or_delayed_update')

    def test_runner_change_excludes_candidate(self):
        rows=observations()
        for row in rows:
            if row['time_sec']>=7:row['runners']['first']=True
        self.assertEqual(self.gate(rows)['episodes'][0]['reason'],'context_changed')

    def test_missing_ocr_does_not_fabricate_context(self):
        rows=observations()
        for row in rows:row['scoreboard']=None
        gate=self.gate(rows)
        self.assertEqual(gate['selected_episodes'],1)
        self.assertIn('inning',gate['episodes'][0]['missing_context'])

    def test_oversized_sampling_gap_is_not_real_absence(self):
        rows=observations()
        rows=[row for row in rows if row['time_sec'] not in [4,4.5]]
        self.assertEqual(self.gate(rows)['episodes'],[])

    def test_following_pitch_is_explicit_and_separate(self):
        rows=observations()
        # Extend with a second display disappearance and a ball-count change.
        for index in range(25,49):
            row=copy.deepcopy(rows[-1]);row.update(frame=index*15,time_sec=index*.5)
            rows.append(row)
        for row in rows:
            if 15<=row['time_sec']<17:
                row.update(board=None,count=None,runners=None,scoreboard=None,board_layout_status='no_detection')
            elif row['time_sec']>=17:
                row['count']={'balls':1,'strikes':2,'outs':0}
        gate=self.gate(rows)
        self.assertEqual(len(gate['windows']),1)
        with_follow=self.gate(rows,include_following_pitch=True)
        self.assertEqual(len(with_follow['windows']),2)
        self.assertEqual(with_follow['requests'][1]['role'],'following_pitch_comparison')
        self.assertLess(with_follow['windows'][0]['end_sec'],with_follow['windows'][1]['end_sec'])

    def test_zero_detections_is_unknown_not_zero_count(self):
        self.assertIsNone(inner_state([],400,180)[0])

    def test_short_post_period_is_not_reported_as_proven_annotation_failure(self):
        rows=observations()[:17]  # Return at7s, clip ends at8s, after1s settling.
        gate=self.gate(rows)
        self.assertEqual(gate['windows'],[])
        self.assertEqual(gate['episodes'][0]['after_failure_detail'],'visible_period_too_short_after_settle')

    def test_clip_range_never_crosses_selected_window(self):
        gate=self.gate(observations())
        for window in gate['windows']:
            for start in clip_starts(window):
                self.assertGreaterEqual(start,window['start_frame'])
                self.assertLessEqual(start+16,window['end_frame_exclusive'])

    def test_full_video_mode_is_rejected(self):
        gate=self.gate(observations());gate['scope']='whole_video'
        with self.assertRaises(ValueError):validate_gate(gate,{'frames':375})

    def test_unselected_parent_is_rejected(self):
        gate=self.gate(observations());gate['episodes'][0]['selected']=False
        with self.assertRaises(ValueError):validate_gate(gate,{'frames':375})

    def test_empty_gate_does_not_import_models(self):
        with tempfile.TemporaryDirectory() as directory:
            gate=self.gate(observations((0,1,0),(0,1,0)))
            Path(directory,'gate.json').write_text(json.dumps(gate))
            original=__import__
            def guarded(name,*args,**kwargs):
                if name in ['torch','transformers','pitch_backend']:raise AssertionError('Model import on empty gate')
                return original(name,*args,**kwargs)
            with mock.patch('builtins.__import__',side_effect=guarded):
                run_pitch(dict(output=directory,frames=375,config={'pitch':{}}))
            report=json.loads(Path(directory,'pitch_report.json').read_text())
            self.assertEqual(report['encoded_clips'],0)
            self.assertFalse(report['whole_video_encoder_used'])

    def test_foul_score_is_not_fabricated_from_gate(self):
        gate=self.gate(observations())
        events=build_events({'config':{'pitch':{'pitch_confirmation_score':.5}}},gate,[])
        self.assertEqual(events[0]['status'],'pitch_not_supported')
        self.assertIsNone(events[0]['model_score'])
        self.assertIsNone(events[0]['confirmed_foul'])

    def test_venv_python_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            executable=path/'venv_python';executable.symlink_to(sys.executable)
            cfg={'python':{'yolo':str(executable)},'yolo':{'weights':'a','inner_weights':'b'},
                 'ocr':{'detector_dir':'d','recognizer_dir':'r'},'pitch':{'checkpoint':'c'}}
            config=path/'config.json';config.write_text(json.dumps(cfg))
            self.assertEqual(resolve_config(config)['python']['yolo'],str(executable))


if __name__=='__main__':unittest.main()
