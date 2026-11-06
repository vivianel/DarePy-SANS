"""Configuration completion through the actual pipeline and GUI loading paths."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from darepy.config_compat import (
    REFERENCE, assert_preserved, complete_config, dump, load_config_file,
    load_reference, parse,
)
from darepy.codes import utils
from darepy.gui_manager import DarePyGUI


class PipelineCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / 'config_experiment.yaml'
        # Model a pre-feature file while keeping its real experiment data.
        self.original = load_reference()
        del self.original['sample_environment']
        del self.original['resolution_settings']
        self.original['physics_corrections'].pop('calculate_resolution_smearing')
        self.original['analysis_flags']['force_reintegrate'] = False
        del self.original['pipeline_control']['force_reintegrate']
        self.original['analysis_paths']['project_base'] = 'my experiment'
        self.original['calibration_samples']['thickness'] = {'default': 0.3, 'CTAB': 0.1}
        self.original['beam_center_mask']['scans'] = {6.0: 123}
        self.path.write_text(dump(self.original), encoding='utf-8')
        self.before = self.path.read_bytes()

    def test_shared_loader_completes_without_writing(self):
        messages = []
        plan = load_config_file(self.path, reporter=messages.append)
        assert_preserved(self.original, plan.data)
        self.assertIn('resolution_settings', plan.data)
        self.assertFalse(plan.data['pipeline_control']['force_reintegrate'])
        self.assertEqual(plan.data['beam_center_mask']['scans'], {6.0: 123})
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertTrue(any(x.startswith('[CONFIG ADDED]') for x in messages))

    def test_pipeline_utils_uses_shared_loader(self):
        with redirect_stdout(StringIO()) as output:
            data = utils.load_config(str(self.path))
        assert_preserved(self.original, data)
        self.assertIn('resolution_settings', data)
        self.assertIn('[CONFIG ADDED]', output.getvalue())
        self.assertEqual(self.path.read_bytes(), self.before)

    def test_argument_and_experiment_pointer_paths(self):
        with patch.object(utils.sys, 'argv', ['caller.py', str(self.path)]), redirect_stdout(StringIO()):
            self.assertIn('resolution_settings', utils.load_config())
        (self.folder / '.active_experiment.txt').write_text(str(self.folder), encoding='utf-8')
        with patch.object(utils.sys, 'argv', ['caller.py']), patch.object(utils, 'CURRENT_DIR', str(self.folder)), redirect_stdout(StringIO()):
            self.assertIn('resolution_settings', utils.load_config())

    def test_new_reference_setting_appears_automatically(self):
        ref = load_reference()
        ref['future_feature'] = {'enabled': False, 'threshold': 12}
        reference = self.folder / 'reference.yaml'
        reference.write_text(dump(ref), encoding='utf-8')
        plan = load_config_file(self.path, reference, reporter=None)
        self.assertEqual(plan.data['future_feature'], {'enabled': False, 'threshold': 12})
        assert_preserved(self.original, plan.data)

    def test_current_reference_requires_no_completion(self):
        plan = load_config_file(REFERENCE, reporter=None)
        self.assertEqual(plan.additions, [])
        self.assertEqual(plan.errors, [])

    def test_reload_is_idempotent(self):
        first = load_config_file(self.path, reporter=None)
        second = complete_config(parse(dump(first.data)), load_reference())
        self.assertEqual(second.additions, [])

    def make_gui(self):
        gui = object.__new__(DarePyGUI)  # Exercise actual methods without a display.
        gui.config_file = str(self.path)
        gui.entries = {}
        gui.log_to_console = MagicMock()
        gui.archive_config = MagicMock()
        return gui

    def test_gui_load_uses_same_values_and_marks_additions(self):
        gui = self.make_gui()
        self.assertTrue(gui.load_data())
        assert_preserved(self.original, gui.config_dict)
        self.assertIn('pipeline_control.force_reintegrate', gui.config_added_paths)
        self.assertTrue(gui.config_completion_pending)
        self.assertEqual(self.path.read_bytes(), self.before)

    def test_gui_first_save_keeps_backup(self):
        gui = self.make_gui()
        self.assertTrue(gui.load_data())
        self.assertTrue(gui.save_data())
        backups = list(self.folder.glob('*.bak'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), self.before)
        assert_preserved(self.original, parse(self.path.read_text(encoding='utf-8')))
        self.assertTrue(gui.save_data())
        self.assertEqual(len(list(self.folder.glob('*.bak'))), 1)

    def test_malformed_section_blocks_gui_save_and_pipeline_load(self):
        self.path.write_text('resolution_settings: null\n', encoding='utf-8')
        gui = self.make_gui()
        self.assertFalse(gui.load_data())
        self.assertFalse(gui.save_data())
        with self.assertRaisesRegex(ValueError, 'resolution_settings'):
            utils.load_config(self.path)

    def test_console_marks_new_values_red(self):
        gui = self.make_gui()
        gui.console, gui.root = MagicMock(), MagicMock()
        DarePyGUI.log_to_console(gui, '[CONFIG ADDED] x = 1')
        self.assertEqual(gui.console.insert.call_args.args[2], 'config_added')

    def test_review_tab_includes_custom_editor_settings(self):
        gui = self.make_gui()
        gui.notebook = MagicMock()
        self.assertTrue(gui.load_data())
        with patch('darepy.gui_manager.ttk') as ttk, patch('darepy.gui_manager.tk.Text'):
            gui._build_config_review()
        rows = ttk.Treeview.return_value
        self.assertEqual(rows.insert.call_count, len(gui.config_completion.additions))
        self.assertEqual(rows.tag_configure.call_args.kwargs['foreground'], '#b00020')


if __name__ == '__main__':
    unittest.main()
