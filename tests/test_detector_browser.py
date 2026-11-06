"""Read-only detector access and the real Tk preview/save workflow."""
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch, MagicMock

import h5py
import numpy as np

from darepy.config_compat import dump, load_reference, parse
from darepy.detector_data import DetectorReader, display_image, find_files, scan_number, validate_limits, viewer_limits, SampleNameReader
from darepy.detector_browser import DetectorBrowser
from darepy.gui_manager import DarePyGUI


def make_hdf(path, data, instrument='SANS-LLB'):
    dataset = '/entry0/central_data/data' if instrument == 'SANS-LLB' else '/entry1/SANS/detector/counts'
    with h5py.File(path, 'w') as source:
        source.create_dataset(dataset, data=data)
        source.create_dataset('/entry0/sample/name', data=np.bytes_('CTAB'))
        source.create_dataset('/entry0/sample/temperature', data=[25.0])
        source.create_dataset('/entry0/sample/long_log', data=np.arange(1000))


class DetectorDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / 'sans2026n000001.hdf'
        self.data = np.arange(60).reshape(3, 4, 5)
        make_hdf(self.path, self.data)

    def test_only_requested_frame_and_small_metadata_are_read(self):
        original = h5py.Dataset.__getitem__
        selections = []

        def track(dataset, selection):
            selections.append((dataset.name, selection))
            return original(dataset, selection)

        with patch.object(h5py.Dataset, '__getitem__', track):
            result = DetectorReader().read(self.path, 'SANS-LLB', 1)
        np.testing.assert_equal(result.image, self.data[1])
        self.assertEqual(result.frames, 3)
        self.assertEqual([key for name, key in selections if name == result.dataset], [(1, slice(None), slice(None))])
        self.assertFalse(any(name.endswith('long_log') for name, _ in selections))
        self.assertIn('array (1000,)', dict(result.metadata)['/entry0/sample/long_log'])
        self.assertEqual(dict(result.metadata)['/entry0/sample/name'], 'CTAB')

    def test_cache_reuses_closed_files_and_is_bounded(self):
        reader = DetectorReader(cache_size=2)
        first = reader.read(self.path, 'SANS-LLB', 0)
        with patch('darepy.detector_data.h5py.File', side_effect=AssertionError('must not reopen')):
            self.assertIs(reader.read(self.path, 'SANS-LLB', 0), first)
        reader.read(self.path, 'SANS-LLB', 1)
        reader.read(self.path, 'SANS-LLB', 2)
        self.assertEqual(len(reader.frames), 2)
        # No persistent HDF handle: file remains writable after every read.
        with h5py.File(self.path, 'r+') as source:
            source['/entry0/central_data/data'][0, 0, 0] = 999
        self.assertEqual(reader.read(self.path, 'SANS-LLB', 0).image[0, 0], 999)

    def test_supports_sans_i_and_rejects_bad_frames(self):
        path = self.folder / 'sans2022n083900.hdf'
        make_hdf(path, np.ones((4, 5)), 'SANS-I')
        self.assertEqual(DetectorReader().read(path, 'SANS-I').frames, 1)
        with self.assertRaisesRegex(ValueError, 'Frame'):
            DetectorReader().read(path, 'SANS-I', 1)
        with self.assertRaisesRegex(ValueError, 'missing'):
            DetectorReader().read(path, 'SANS-LLB')

    def test_recursive_discovery_and_scan_numbers(self):
        child = self.folder / 'subfolder'
        child.mkdir()
        (child / 'test000002.H5').touch()
        (child / 'ignore.txt').touch()
        self.assertEqual(len(find_files(self.folder)), 2)
        self.assertEqual(scan_number(self.path), 1)
        self.assertEqual(scan_number('sans2022n083907_mod.hdf'), 83907)

    def test_display_transform_preserves_raw_counts_and_log_convention(self):
        raw = np.array([[0.0, 10.0], [20.0, 30.0]])
        original = raw.copy()
        background = np.ones((2, 2))
        image = display_image(raw, 'log', background)
        self.assertAlmostEqual(float(image[0, 1]), np.log10(9), places=5)
        self.assertTrue(np.isfinite(image).all())
        np.testing.assert_equal(raw, original)
        with self.assertRaisesRegex(ValueError, 'shape'):
            display_image(raw, background=np.ones((3, 3)))

    def test_sample_names_only_read_small_name_field_and_use_cache(self):
        original = h5py.Dataset.__getitem__
        fields = []

        def track(dataset, selection):
            fields.append(dataset.name)
            return original(dataset, selection)

        reader = SampleNameReader()
        with patch.object(h5py.Dataset, '__getitem__', track):
            self.assertEqual(reader.read(self.path, 'SANS-LLB'), 'CTAB')
        self.assertEqual(fields, ['/entry0/sample/name'])
        with patch('darepy.detector_data.h5py.File', side_effect=AssertionError('must not reopen')):
            self.assertEqual(reader.read(self.path, 'SANS-LLB'), 'CTAB')

    def test_sans_i_sample_name_and_missing_name(self):
        path = self.folder / 'sans2022n000003.hdf'
        with h5py.File(path, 'w') as source:
            source.create_dataset('/entry1/sample/name', data=[np.bytes_('Water reference')])
        self.assertEqual(SampleNameReader().read(path, 'SANS-I'), 'Water reference')
        with h5py.File(path, 'w') as source:
            source.create_dataset('/entry1/sample/other', data=1)
        self.assertEqual(SampleNameReader().read(path, 'SANS-I'), '(unnamed)')

    def test_legacy_log_limits_convert_once_without_changing_config(self):
        cfg = {'plot_scale': 'log', 'clim': [0, float(np.log(1000))]}
        before = deepcopy(cfg)
        low, high = viewer_limits(cfg)
        self.assertAlmostEqual(high, 3)
        self.assertEqual(cfg, before)
        self.assertEqual(viewer_limits({'plot_scale': 'log', 'clim': [0, 3], 'log_base': '10'}), (0, 3))
        image = np.array([[100.0]])
        self.assertAlmostEqual(float(display_image(image, 'log')[0, 0]), 2)
        self.assertAlmostEqual(float(display_image(image, 'log', log_base='e')[0, 0]), np.log(100), places=5)

    def test_bad_limits_are_rejected(self):
        for low, high in [(1, 1), (2, 1), ('nan', 5), (1, 'inf')]:
            with self.assertRaises(ValueError):
                validate_limits(low, high)


class DetectorGuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.raw = self.folder / 'raw_data'
        self.raw.mkdir()
        self.path = self.raw / 'sans2026n000001.hdf'
        make_hdf(self.path, np.arange(60).reshape(3, 4, 5))
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)

        def close():
            self.root.destroy()
            self.assertEqual(self.callback_errors, [], 'Tk callback errors during preview or cleanup')
        self.addCleanup(close)
        host = object.__new__(DarePyGUI)
        host.root = self.root
        host.config_file = str(self.folder / 'config_experiment.yaml')
        host.config_dict = load_reference()
        host.config_dict['analysis_paths']['raw_data'] = str(self.raw)
        host.config_dict['analysis_paths']['project_base'] = str(self.folder)
        host.config_dict['instrument_setup']['which_instrument'] = 'SANS-LLB'
        host.config_dict['plot_2d']['enable_bg_subtraction'] = False
        host.entries = {}
        host._config_load_ok = True
        host.config_completion_pending = False
        host.log_to_console = MagicMock()
        host.archive_config = MagicMock()
        Path(host.config_file).write_text(dump(host.config_dict), encoding='utf-8')
        self.original_text = Path(host.config_file).read_text(encoding='utf-8')
        self.host = host
        self.viewer = DetectorBrowser(self.root, host)
        host.detector_browser = self.viewer
        self.viewer.pack(fill='both', expand=True)
        self.wait(lambda: len(self.viewer.files) == 1)
        self.wait(lambda: self.viewer.sample_names.get(self.path) == 'CTAB')
        self.viewer.file_tree.selection_set('0')
        self.viewer._selected()
        self.wait(lambda: self.viewer.current is not None)

    def wait(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(0.01)
        self.fail('Viewer did not finish: ' + self.viewer.status.get())

    def test_click_loads_frame_and_metadata(self):
        self.assertEqual(self.viewer.current.frames, 3)
        self.assertTrue(self.viewer.meta_tree.get_children())
        np.testing.assert_equal(self.viewer.raw, np.arange(60).reshape(3, 4, 5)[0])

    def test_scale_and_clim_changes_do_not_read_hdf_again(self):
        with patch.object(self.viewer.reader, 'read', side_effect=AssertionError('unexpected HDF read')):
            self.viewer.scale.set('lin')
            self.viewer.low.set('1')
            self.viewer.high.set('20')
            self.wait(lambda: self.viewer.artist.get_clim() == (1, 20))
            self.assertEqual(self.viewer.colorbar.ax.get_ylabel(), 'Counts')

    def test_frame_control_loads_requested_frame(self):
        self.viewer.frame.set('2')
        self.wait(lambda: self.viewer.current.frame == 1)
        np.testing.assert_equal(self.viewer.raw, np.arange(60).reshape(3, 4, 5)[1])

    def test_last_settings_only_reach_disk_when_saved(self):
        self.viewer.scale.set('lin')
        self.viewer.low.set('2')
        self.viewer.high.set('15')
        self.viewer.render()
        self.assertEqual(Path(self.host.config_file).read_text(encoding='utf-8'), self.original_text)
        self.assertTrue(self.host.save_data())
        saved = parse(Path(self.host.config_file).read_text(encoding='utf-8'))
        self.assertEqual(saved['plot_2d']['clim'], [2, 15])
        self.assertEqual(saved['plot_2d']['plot_scale'], 'lin')
        self.assertEqual(saved['analysis_paths'], parse(self.original_text)['analysis_paths'])

    def test_save_displayed_image_and_invalid_settings(self):
        destination = self.folder / 'preview.png'
        with patch('darepy.detector_browser.filedialog.asksaveasfilename', return_value=str(destination)):
            self.viewer.save_image()
        self.assertGreater(destination.stat().st_size, 1000)
        self.viewer.low.set('nan')
        before = Path(self.host.config_file).read_bytes()
        self.assertFalse(self.host.save_data())
        self.assertEqual(Path(self.host.config_file).read_bytes(), before)

    def test_background_subtraction_and_toggle(self):
        make_hdf(self.raw / 'sans2026n000002.hdf', np.ones((4, 5)) * 2)
        self.viewer.refresh_files()
        self.wait(lambda: len(self.viewer.files) == 2)
        self.viewer.bg_scan.set('2')
        self.viewer.bg_enabled.set(True)
        self.viewer.request_frame()
        self.wait(lambda: self.viewer.background is not None)
        self.viewer.scale.set('lin')
        self.viewer.render()
        self.assertAlmostEqual(float(self.viewer.artist.get_array()[1, 0]), 3.0)
        self.viewer.bg_enabled.set(False)
        self.viewer.request_frame()
        self.wait(lambda: self.viewer.background is None)
        self.assertEqual(float(self.viewer.raw[1, 0]), 5.0)

    def test_rapid_frame_changes_display_latest_selection(self):
        original = self.viewer.reader.read

        def slow(path, instrument, frame):
            if frame == 1:
                time.sleep(0.08)
            return original(path, instrument, frame)

        with patch.object(self.viewer.reader, 'read', side_effect=slow):
            self.viewer.frame.set('2')
            self.viewer.request_frame()
            self.viewer.frame.set('3')
            self.viewer.request_frame()
            self.wait(lambda: self.viewer.current.frame == 2)
            self.assertEqual(float(self.viewer.raw[0, 0]), 40.0)


    def test_different_file_gets_auto_limits_and_current_file_keeps_manual_limits(self):
        make_hdf(self.raw / 'sans2026n000002.hdf', np.arange(20).reshape(4, 5) * 1000 + 100)
        self.viewer.refresh_files()
        self.wait(lambda: len(self.viewer.files) == 2)
        self.wait(lambda: len(self.viewer.sample_names) == 2)
        self.viewer.low.set('-10')
        self.viewer.high.set('-5')
        self.viewer.file_tree.selection_set('1')
        self.viewer._selected()
        self.wait(lambda: self.viewer.current is not None and self.viewer.current.path.name.endswith('000002.hdf'))
        expected = np.percentile(display_image(self.viewer.raw, 'log').compressed(), [1, 99.7])
        self.assertAlmostEqual(float(self.viewer.low.get()), expected[0], places=4)
        self.assertAlmostEqual(float(self.viewer.high.get()), expected[1], places=4)
        self.viewer.low.set('2')
        self.viewer.high.set('4')
        self.viewer._selected()  # A label/filter refresh must not reset the image.
        self.assertEqual(self.viewer.low.get(), '2')
        self.assertEqual(self.viewer.high.get(), '4')

    def test_file_list_shows_and_filters_sample_name(self):
        self.assertEqual(self.viewer.file_tree.set('0', 'sample'), 'CTAB')
        self.viewer.filter.set('ctab')
        self.root.update()
        self.assertEqual(self.viewer.file_tree.get_children(), ('0',))
        self.viewer.filter.set('not this sample')
        self.assertEqual(self.viewer.file_tree.get_children(), ())

    def test_action_buttons_are_blue_and_saved_log_base_is_explicit(self):
        button = self.viewer._button(self.viewer, text='test')
        self.assertEqual(button.cget('background'), '#2196F3')
        self.assertEqual(button.cget('foreground'), 'white')
        button.destroy()
        self.assertTrue(self.host.save_data())
        saved = parse(Path(self.host.config_file).read_text(encoding='utf-8'))
        self.assertEqual(saved['plot_2d']['log_base'], 10)
        self.assertAlmostEqual(saved['plot_2d']['clim'][1], float(self.viewer.high.get()))


if __name__ == '__main__':
    unittest.main()
