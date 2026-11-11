from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from configure_windows import find_pitch_checkpoint
from extract_reference_scenes import load_config


class ConfigurationTests(unittest.TestCase):
    def test_bundled_weight_is_found_when_project_has_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)/'for_labelimg';program=project/'two_strike_foul'
            checkpoint=program/'models/compact_actionformer.pth';checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fixture')
            self.assertEqual(find_pitch_checkpoint(project,program_root=program),str(checkpoint.resolve()))

    def test_explicit_weight_takes_precedence_over_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);program=project/'two_strike_foul'
            bundled=program/'models/compact_actionformer.pth';bundled.parent.mkdir(parents=True);bundled.write_bytes(b'bundle')
            custom=project/'custom.pth';custom.write_bytes(b'custom')
            self.assertEqual(find_pitch_checkpoint(project,custom,program),str(custom.resolve()))
            with self.assertRaises(FileNotFoundError):find_pitch_checkpoint(project,project/'missing.pth',program)

    def test_zip_extracted_repository_name_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);project=root/'for_labelimg';project.mkdir()
            checkpoint=root/'VideoMAE-Baseball-main/results/videomae_actionformer/compact_actionformer.pth'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'fixture')
            self.assertEqual(find_pitch_checkpoint(project,program_root=project/'two_strike_foul'),str(checkpoint.resolve()))

    def test_missing_supplied_config_explains_setup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.local.json'
            with self.assertRaises(FileNotFoundError) as caught:load_config(path,required=True)
            self.assertIn(str(path.resolve()),str(caught.exception))
            self.assertIn('configure_windows.py',str(caught.exception))
            self.assertNotIn('--analyze には --config',str(caught.exception))
            self.assertIsNone(load_config(path,required=False))


if __name__=='__main__':unittest.main()
