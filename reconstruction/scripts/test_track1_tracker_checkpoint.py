"""Recovery must preserve trusted poses and reject stale/corrupt checkpoints."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modules/v2d_cari4d/lib/cari4d'))
from prep.tracker_checkpoint import load_checkpoint, save_checkpoint


class TrackerCheckpointTests(unittest.TestCase):
    def test_interrupted_trajectory_restores_trusted_state_and_missing_prefix(self):
        names = ['000000','000001','000002']
        pose = np.eye(4); pose[2,3] = 3
        records = [{'frame':names[0],'reliable':False}, {'frame':names[1],'reliable':True}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pkl'
            save_checkpoint(path, {'source':'original'}, names, [None,pose], records, pose)
            poses, restored, previous = load_checkpoint(path, {'source':'original'}, names)
            self.assertIsNone(poses[0])
            np.testing.assert_array_equal(poses[1], pose)
            np.testing.assert_array_equal(previous, pose)
            self.assertEqual(restored, records)
            self.assertEqual(names[len(restored):], ['000002'])

    def test_changed_inputs_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pkl'
            save_checkpoint(path, {'mesh':'original'}, ['000000'], [], [], None)
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_checkpoint(path, {'mesh':'replaced'}, ['000000'])

    def test_noncontiguous_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                save_checkpoint(Path(tmp)/'checkpoint.pkl', {}, ['000000','000001'], [None], [{'frame':'000001','reliable':False}], None)

    def test_invalid_update_preserves_previous_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pkl'
            save_checkpoint(path, {}, ['000000'], [], [], None)
            bad = np.eye(4); bad[2,3] = np.nan
            with self.assertRaisesRegex(ValueError, 'invalid pose'):
                save_checkpoint(path, {}, ['000000'], [bad], [{'frame':'000000','reliable':True}], bad)
            self.assertEqual(load_checkpoint(path, {}, ['000000']), ([],[],None))

    def test_accepted_records_require_trusted_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'trusted state'):
                save_checkpoint(Path(tmp)/'checkpoint.pkl', {}, ['000000'], [np.eye(4)], [{'frame':'000000','reliable':True}], None)


if __name__ == '__main__':
    unittest.main()
