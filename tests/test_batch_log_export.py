"""The actual command-line exporter honors log10 and GIF playback options."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image
from darepy.config_compat import dump, load_reference


class BatchLogExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[1]
        raw = self.folder / 'raw_data'
        raw.mkdir()
        with h5py.File(raw / 'sans2026n000001.hdf', 'w') as source:
            source.create_dataset('/entry0/central_data/data', data=np.stack([np.ones((4, 5)) * 10, np.ones((4, 5)) * 100]))
            source.create_dataset('/entry0/sample/name', data=[np.bytes_('CTAB')])
        shutil.copyfile(self.repo / 'experiments/default_exp/instrument_registry.yaml', self.folder / 'instrument_registry.yaml')
        self.config = load_reference()
        self.config['analysis_paths']['raw_data'] = str(raw)
        self.config['analysis_paths']['project_base'] = str(self.folder)
        self.config['instrument_setup']['which_instrument'] = 'SANS-LLB'
        self.config['plot_2d'].update(list_scan=[1], clim=[0, 3], plot_scale='log', log_base=10,
                                      enable_bg_subtraction=False, gif_fps=2)

    def run_export(self, mode, once=False):
        self.config['plot_2d'].update(output_mode=mode, play_once=once)
        config = self.folder / 'config_experiment.yaml'
        config.write_text(dump(self.config), encoding='utf-8')
        result = subprocess.run([sys.executable, '-B', str(self.repo / 'darepy/caller_plot_2Dpattern.py'), str(config)],
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        return self.folder / 'extra_results'

    def test_log10_png_export_saves_all_frames(self):
        output = self.run_export('individual_frames')
        files = list(output.glob('*.png'))
        self.assertEqual(len(files), 2)
        for file in files:
            with Image.open(file) as image:
                image.verify()

    def test_log10_gif_export_respects_play_once(self):
        output = self.run_export('gif_animation', once=True)
        with Image.open(output / 'scan_1_1.gif') as image:
            self.assertEqual(image.n_frames, 2)
            self.assertNotIn('loop', image.info)
            self.assertEqual(image.info['duration'], 500)

    def test_log10_gif_export_can_repeat(self):
        output = self.run_export('gif_animation', once=False)
        with Image.open(output / 'scan_1_1.gif') as image:
            self.assertEqual(image.n_frames, 2)
            self.assertEqual(image.info['loop'], 0)


if __name__ == '__main__':
    unittest.main()
