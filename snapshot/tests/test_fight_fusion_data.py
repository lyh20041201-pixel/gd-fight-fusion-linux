import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.download_fight_fusion_data import check_content_range, safe_extract, download_ubi_parallel, UBI_URL
from scripts.prepare_fight_fusion_data import (
    DSU, allocate_groups, deduplicate_and_split, intervals_from_labels,
    parse_scfd_sources, source_id, screen_overlap,
    sampled_frames,
    inspect_video, LabelConflict,
)


def row(sid, group, label=0, old=False, split='unassigned', dataset='scfd', sha=None, original_split='unspecified'):
    return dict(sample_id=sid, group=group, label=label, frozen_legacy=old, split=split,
                dataset=dataset, sha256=sha or sid, source_id=None, path=sid+'.mp4', original_split=original_split)


class FightFusionDataTests(unittest.TestCase):
    def test_directory_frame_label_conflict_is_not_relabelled(self):
        from unittest.mock import patch
        from scripts.download_fight_fusion_data import sha256
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            video=Path(folder)/'fight.avi';annotation=Path(folder)/'fight.csv'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(64,48))
            for _ in range(8):writer.write(np.zeros((48,64,3),np.uint8))
            writer.release();annotation.write_text('0\n'*8)
            r=dict(sample_id='conflict',dataset='ubi',path=str(video),sha256=sha256(video),label=1,
                   label_kind='frame',annotation_path=str(annotation),annotation_sha256=sha256(annotation),original_split='test')
            with patch('scripts.prepare_fight_fusion_data.OUT',Path(folder)/'out'):
                with self.assertRaises(LabelConflict) as error:inspect_video(r)
            self.assertEqual(error.exception.evidence['original_video_label'],1)
            self.assertEqual(error.exception.evidence['positive_frame_count'],0)
            self.assertEqual(error.exception.evidence['original_split'],'test')
            self.assertEqual(r['label'],1)
            self.assertEqual(annotation.read_text(),'0\n'*8)

    def test_sequential_sampling_matches_timestamp_seeks(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            for fps in (25.,29.97,30.):
                path=Path(folder)/(str(fps)+'.avi')
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),fps,(64,48))
                self.assertTrue(writer.isOpened())
                for i in range(150):
                    frame=np.full((48,64,3),(i%255,(i*3)%255,(i*7)%255),np.uint8)
                    cv2.putText(frame,str(i),(2,25),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                    writer.write(frame)
                writer.release()
                times=[0,.5,1,1.5,2,4.8]
                cap=cv2.VideoCapture(str(path));actual_fps=cap.get(cv2.CAP_PROP_FPS);reference=[]
                for t in times:
                    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read();self.assertTrue(ok);reference.append(frame)
                cap.release();cap=cv2.VideoCapture(str(path))
                observed=list(sampled_frames(cap,times,actual_fps,150));cap.release()
                for expected,(_,ok,actual) in zip(reference,observed):
                    self.assertTrue(ok)
                    np.testing.assert_array_equal(expected,actual)

    def test_ubi_resume_preserves_prefix_and_limits_tls_exception(self):
        from unittest.mock import patch
        payload=bytes(range(100))
        class Response:
            status_code=206
            headers={'Content-Range':'bytes 37-99/100'}
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def iter_content(self,chunk_size):yield payload[37:]
        def get(url,**kw):
            self.assertEqual(url,UBI_URL)
            self.assertFalse(kw['verify'])
            self.assertFalse(kw['allow_redirects'])
            self.assertEqual(kw['headers']['Range'],'bytes=37-99')
            return Response()
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'UBI_FIGHTS.zip'
            archive.with_suffix('.zip.part').write_bytes(payload[:37])
            with patch('scripts.download_fight_fusion_data.UBI_BYTES',100), \
                 patch('scripts.download_fight_fusion_data.check_disk'), \
                 patch('scripts.download_fight_fusion_data.requests.get',side_effect=get):
                download_ubi_parallel(archive,lambda *args,**kw:None,workers=1)
            self.assertEqual(archive.read_bytes(),payload)

    def test_range_requires_exact_total_and_offset(self):
        check_content_range('bytes 10-19/100',10,19,100)
        for actual in [None,'bytes 0-19/100','bytes 10-19/101']:
            with self.assertRaises(RuntimeError):check_content_range(actual,10,19,100)

    def test_zip_traversal_refused(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside.txt','unsafe')
            with patch('scripts.download_fight_fusion_data.check_disk'):
                with self.assertRaises(ValueError):safe_extract(archive,Path(folder)/'out')
            self.assertFalse((Path(folder)/'outside.txt').exists())

    def test_frame_labels_are_half_open_seconds(self):
        self.assertEqual(intervals_from_labels([0,1,1,0,1],2),[[.5,1.5],[2,2.5]])
        self.assertEqual(intervals_from_labels([0,0],30),[])
        with self.assertRaises(ValueError):intervals_from_labels([0,2],30)

    def test_youtube_identity_preserves_case(self):
        self.assertEqual(source_id('https://youtu.be/-Cjvg5ECIrE?t=1'),'youtube:-Cjvg5ECIrE')
        self.assertEqual(source_id('https://www.youtube.com/watch?v=-Cjvg5ECIrE'),'youtube:-Cjvg5ECIrE')

    def test_scfd_source_context(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'videos.txt'
            p.write_text('https://youtu.be/-Cjvg5ECIrE\nfi001: 1 - 2\nnofi002: 2 - 3\n')
            self.assertEqual(parse_scfd_sources(p),{'fi001':['youtube:-Cjvg5ECIrE'],'nofi002':['youtube:-Cjvg5ECIrE']})

    def test_scfd_ambiguous_author_sources_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'videos.txt'
            p.write_text('https://youtu.be/-Cjvg5ECIrE\nnofi022: 1 - 2\nhttps://youtu.be/MTBxPiwxQCw\nnofi022: 3 - 4\nnofi062 - nofi076 : CamNet Dataset\n')
            m=parse_scfd_sources(p)
            self.assertEqual(len(m['nofi022']),2)
            self.assertEqual(m['nofi062'],['scfd-author-collection:camnet dataset'])

    def test_group_assignment_is_order_independent(self):
        groups={str(i):[row(str(i),str(i),i%2)] for i in range(30)}
        a=allocate_groups(groups,{'train':50,'branch_val':10,'fusion_fit':20,'calibration':10})
        b=allocate_groups(dict(reversed(list(groups.items()))),{'train':50,'branch_val':10,'fusion_fit':20,'calibration':10})
        self.assertEqual(a,b)
        self.assertEqual(set(a.values()),{'train','branch_val','fusion_fit','calibration'})

    def test_new_duplicate_never_enters_old_test_or_training(self):
        rows=[row('old','old-g',old=True,split='legacy_test',dataset='vfd'),row('new','new-g')]
        d=DSU(2);d.union(0,1)
        kept,excluded,_=deduplicate_and_split(rows,d)
        self.assertEqual([r['sample_id'] for r in kept],['old'])
        self.assertEqual(excluded[0]['reason'],'overlap_with_frozen_legacy_source')
        self.assertEqual(kept[0]['split'],'legacy_test')

    def test_ubi_official_test_group_kept_whole(self):
        rows=[row('u','u',dataset='ubi',original_split='test'),row('s','s')]
        d=DSU(2);d.union(0,1)
        kept,excluded,_=deduplicate_and_split(rows,d)
        self.assertEqual({r['split'] for r in kept},{'test'})
        self.assertEqual(len({r['group'] for r in kept}),1)
        self.assertEqual(excluded,[])

    def test_conflicting_exact_duplicate_labels_are_quarantined(self):
        rows=[row('a','a',label=0,sha='same'),row('b','b',label=1,sha='same')]
        d=DSU(2);d.union(0,1)
        kept,excluded,_=deduplicate_and_split(rows,d)
        self.assertEqual(kept,[])
        self.assertEqual(len(excluded),2)
        self.assertEqual({r['reason'] for r in excluded},{'conflicting_new_labels_for_identical_bytes'})

    def test_deduplication_cannot_erase_official_test_membership(self):
        rows=[row('a','a',dataset='ubi',sha='same'),row('b','b',dataset='ubi',sha='same',original_split='test')]
        d=DSU(2);d.union(0,1)
        kept,excluded,_=deduplicate_and_split(rows,d)
        self.assertEqual(len(kept),1)
        self.assertEqual(kept[0]['split'],'test')

    def test_single_visual_match_cannot_merge_sources(self):
        rows=[row('a','a',old=True,split='train',dataset='vfd'),row('b','b')]
        ins={'a':{'fingerprints':[[0,12345,12345]]},'b':{'fingerprints':[[0,12345,12345]]}}
        d,e=screen_overlap(rows,ins)
        self.assertNotEqual(d.find(0),d.find(1))

    def test_repeated_static_background_cannot_merge_sources(self):
        rows=[row('a','a',old=True,split='train',dataset='vfd'),row('b','b')]
        frames=[[t,12345,12345] for t in [0,.5,1,1.5]]
        d,e=screen_overlap(rows,{'a':{'fingerprints':frames},'b':{'fingerprints':frames}})
        self.assertNotEqual(d.find(0),d.find(1))

    def test_three_temporally_aligned_matches_are_grouped(self):
        rows=[row('a','a',old=True,split='train',dataset='vfd'),row('b','b')]
        frames=[[0,0xFFFF0000FFFF,0x123456],[.5,0xEEEE0000EEEE,0x765432],[1,0xDDDD0000DDDD,0xABCDE]]
        ins={'a':{'fingerprints':frames},'b':{'fingerprints':[[t+2,p,d] for t,p,d in frames]}}
        d,e=screen_overlap(rows,ins)
        self.assertEqual(d.find(0),d.find(1))
        self.assertTrue(any(x['method']=='perceptual_temporal_candidate' for x in e))


if __name__=='__main__':unittest.main()
