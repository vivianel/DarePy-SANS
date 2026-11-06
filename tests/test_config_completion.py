"""Run with: python -m unittest discover -s . -p test_update_config.py -v"""
import tempfile
from pathlib import Path
import unittest
from darepy.config_compat import complete_config, parse, dump, assert_preserved, save_plan


class CompletionTests(unittest.TestCase):
    def test_saved_values_and_comments_survive(self):
        old = parse('# Keep comment\na: false\nb: 0\nc: ""\nd: null\ne: [1, 2]\ncustom: yes\n')
        plan = complete_config(old, parse('a: true\nb: 5\nc: changed\nd: 9\ne: [3]\nf: 8\n'))
        assert_preserved(old, parse(dump(plan.data)))
        self.assertIn('# Keep comment', dump(plan.data))
        self.assertEqual([x.path for x in plan.additions], ['f'])

    def test_renamed_values_are_copied(self):
        old = parse('analysis_flags:\n  force_reintegrate: false\ntransmission_setup:\n  transmission_dist: 18\n')
        ref = parse('pipeline_control:\n  force_reintegrate: true\ntransmission_setup:\n  dist_trans_measurements: 17.5\n')
        plan = complete_config(old, ref)
        self.assertIs(plan.data['pipeline_control']['force_reintegrate'], False)
        self.assertEqual(plan.data['transmission_setup']['dist_trans_measurements'], 18)
        assert_preserved(old, plan.data)

    def test_dynamic_distances_and_samples_are_not_imported(self):
        old = parse('detector_geometry:\n  beam_center_guess:\n    6.0: [1, 2]\ncalibration_samples:\n  thickness:\n    default: 0.3\n')
        ref = parse('detector_geometry:\n  beam_center_guess:\n    8: [3, 4]\ncalibration_samples:\n  thickness:\n    EB: 0.1\n    default: 2\n')
        plan = complete_config(old, ref)
        self.assertEqual(plan.data, old)
        self.assertEqual(plan.additions, [])

    def test_missing_dynamic_mapping_is_empty(self):
        plan = complete_config(parse('{}'), parse('beam_center_mask:\n  scans:\n    8: 123\n'))
        self.assertEqual(plan.data['beam_center_mask']['scans'], {})

    def test_scalar_section_conflict_blocks_save(self):
        plan = complete_config(parse('section: null'), parse('section:\n  x: 1\n'))
        self.assertTrue(plan.errors)
        self.assertIsNone(plan.data['section'])

    def test_second_run_is_unchanged(self):
        ref = parse('section:\n  a: 1\n  b: [2, 3]\n')
        first = complete_config(parse('section:\n  a: 7'), ref)
        second = complete_config(parse(dump(first.data)), ref)
        self.assertEqual(second.additions, [])
        self.assertEqual(second.data, first.data)

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(Exception):
            parse('x: 1\nx: 2\n')

    def test_save_backup_and_concurrent_change(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'old.yaml'
            source.write_text('x: 0\n', encoding='utf-8')
            text = source.read_text(encoding='utf-8')
            plan = complete_config(parse(text), parse('x: 1\ny: 2'))
            save_plan(plan, text, source, source)
            backups = list(Path(folder).glob('*.bak'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding='utf-8'), text)
            with self.assertRaises(ValueError):
                save_plan(plan, text, source, Path(folder) / 'new.yaml')

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'old.yaml', Path(folder) / 'new.yaml'
            source.write_text('x: 0\n', encoding='utf-8')
            output.write_text('keep me', encoding='utf-8')
            text = source.read_text(encoding='utf-8')
            plan = complete_config(parse(text), parse('x: 1\ny: 2'))
            with self.assertRaises(ValueError):
                save_plan(plan, text, source, output)
            self.assertEqual(output.read_text(encoding='utf-8'), 'keep me')


if __name__ == '__main__':
    unittest.main()
