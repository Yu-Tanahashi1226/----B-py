import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from event_count_gate import count_gate
from outcome_features import predict_outcome
from reference_matcher import index_video,locate
from common import sha256


def counts():
    return [dict(time_sec=i*.5,board={'box':[0,0,20,10]},board_layout_status='supported_corner_layout',
                 count={'balls':1,'strikes':2,'outs':0},runners=None,scoreboard=None) for i in range(30)]


class CountGateTests(unittest.TestCase):
    def test_constant_on_screen_two_strike_count_is_eligible(self):
        result=count_gate(counts(),3.,5.)
        self.assertTrue(result['selected'])
        self.assertFalse(result['disappearance_required'])

    def test_late_ball_update_excludes_earlier_unchanged_value(self):
        rows=counts()
        for r in rows:
            if r['time_sec']>=8:r['count']['balls']=2
        self.assertEqual(count_gate(rows,3.,5.)['reason'],'bso_changed_or_delayed_update')

    def test_missing_post_count_remains_unknown(self):
        rows=counts()
        for r in rows:
            if r['time_sec']>5:r['count']=None
        result=count_gate(rows,3.,5.)
        self.assertFalse(result['selected']);self.assertEqual(result['reason'],'after_bso_unknown')

    def test_following_pitch_does_not_change_this_pitch_post_comparison(self):
        rows=counts()
        for r in rows:
            if r['time_sec']>=10:r['count']['balls']=2
        self.assertTrue(count_gate(rows,3.,5.,next_start_sec=10.)['selected'])

    def test_not_two_strikes(self):
        rows=counts()
        for r in rows:r['count']['strikes']=1
        self.assertEqual(count_gate(rows,3.,5.)['reason'],'not_two_strikes')

    def test_temporary_out_drop_requires_known_half_and_returned_count(self):
        rows=counts()
        for r in rows:
            r['count']['outs']=1;r['scoreboard']={'inning':5,'half':'bottom'}
            if 6<=r['time_sec']<=8:r['count']['outs']=0
        result=count_gate(rows,3.,5.)
        self.assertTrue(result['selected']);self.assertEqual(len(result['ignored_inconsistent_out_drops']),1)
        self.assertEqual(rows[12]['count']['outs'],0)  # Raw prediction was not rewritten.

    def test_persistent_out_drop_is_not_silently_corrected(self):
        rows=counts()
        for r in rows:
            r['count']['outs']=1 if r['time_sec']<6 else 0
            r['scoreboard']={'inning':5,'half':'bottom'}
        self.assertFalse(count_gate(rows,3.,5.)['selected'])

    def test_unknown_inning_prevents_out_drop_filter(self):
        rows=counts()
        for r in rows:
            r['count']['outs']=0 if 6<=r['time_sec']<=8 else 1
        self.assertFalse(count_gate(rows,3.,5.)['selected'])

    def test_outcome_head_rejects_target_reference_leakage(self):
        payload={'leave_one_reference_out_heads':{'a':{'training_references':['a'],'coefficient':[1.],'intercept':0.}}}
        with self.assertRaises(ValueError):predict_outcome(payload,'a',np.asarray([1.]))

    def test_unknown_count_never_starts_new_videomae_worker_models(self):
        from matched_videomae import verify
        root=Path(__file__).resolve().parents[1]
        payload=json.loads((root/'reference_outcome_heads.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            job=dict(output=directory,head_sha256=sha256(root/'reference_outcome_heads.json'),
                     config={'pitch':{'encoder':payload['encoder'],'revision':payload['revision']}},
                     events=[dict(event_id=1,count_gate={'selected':False,'reason':'after_bso_unknown'})])
            original=__import__
            def guarded(name,*args,**kwargs):
                if name in ['torch','transformers','pitch_backend']:raise AssertionError('Model started for an unknown count')
                return original(name,*args,**kwargs)
            with mock.patch('builtins.__import__',side_effect=guarded):verify(job)
            report=json.loads(Path(directory,'verification_report.json').read_text())
            self.assertEqual(report['encoded_clips'],0)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
        rng=np.random.default_rng(4)
        coarse=rng.integers(0,255,(6,10,3),dtype=np.uint8)
        base=cv2.resize(coarse,(160,96),interpolation=cv2.INTER_CUBIC)
        self.scene=[]
        for i in range(80):
            frame=base.copy();cv2.rectangle(frame,(10+i,20),(40+i,70),(10,240,40),-1)
            cv2.putText(frame,str(i//10),(5,90),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            self.scene.append(frame)
        self.ref=self.write('reference.mp4',self.scene)

    def tearDown(self):self.directory.cleanup()

    def write(self,name,frames):
        path=self.root/name;writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,96))
        self.assertTrue(writer.isOpened())
        for frame in frames:writer.write(frame)
        writer.release();return path

    def match(self,frames):
        source=self.write('source.mp4',frames)
        a=index_video(source,self.root/'source-index');b=index_video(self.ref,self.root/'ref-index')
        return locate(source,self.ref,a,b)

    def test_unknown_offset_and_reencoding(self):
        blank=np.full((96,160,3),80,dtype=np.uint8)
        result=self.match([blank]*12+self.scene+[blank]*30)
        self.assertEqual(result['status'],'matched');self.assertAlmostEqual(result['offset_sec'],1.2,delta=.11)

    def test_replayed_duplicate_is_ambiguous(self):
        blank=np.full((96,160,3),80,dtype=np.uint8)
        result=self.match([blank]*12+self.scene+[blank]*30+self.scene+[blank]*30)
        self.assertEqual(result['status'],'ambiguous')

    def test_absent_scene_is_not_fabricated(self):
        blank=np.full((96,160,3),80,dtype=np.uint8)
        self.assertEqual(self.match([blank]*150)['status'],'not_found')


if __name__=='__main__':unittest.main()
