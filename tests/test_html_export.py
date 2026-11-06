"""Single-file beamtime summaries with escaped metadata and no Excel CSV writes."""
from contextlib import redirect_stdout
from copy import deepcopy
from html.parser import HTMLParser
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'darepy' / 'codes'))
import prepare_input as preparation
from metadata_html import render_metadata_html, write_metadata_html
from test_csv_export import windows_reader_lock


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.headers, self.rows, self.tags = [], [], []
        self.cell, self.row, self.in_button = None, None, False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == 'tr':
            self.row = []
        if tag in ('td', 'th'):
            self.cell = ''

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag == 'th':
            self.headers.append(self.cell)
            self.cell = None
        elif tag == 'td':
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.row:
            self.rows.append(self.row)


class HtmlExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def export(self, table):
        with redirect_stdout(StringIO()) as output:
            actual = preparation.save_list_files(str(self.folder), str(self.folder), table, 'all_files')
        return Path(actual), output.getvalue()

    def test_export_reuses_one_html_file_and_preserves_metadata(self):
        table = {'name_hdf': ['one.hdf', 'two.hdf'], 'scan': [10, 2], 'sample_name': ['A', 'B']}
        original = deepcopy(table)
        first, _ = self.export(table)
        second, output = self.export(table)
        self.assertEqual(first, self.folder / 'all_files.html')
        self.assertEqual(first, second)
        self.assertEqual(table, original)
        self.assertEqual(list(self.folder.iterdir()), [first])
        self.assertIn('all_files.html', output)

    def test_page_contains_every_row_even_when_console_is_truncated(self):
        table = {'name_hdf': [f'{i}.hdf' for i in range(25)], 'scan': list(range(25))}
        path, output = self.export(table)
        parser = TableParser()
        parser.feed(path.read_text(encoding='utf-8'))
        self.assertEqual(len(parser.rows), 25)
        self.assertEqual(parser.rows[-1], ['24.hdf', '24'])
        self.assertIn('Showing first 20', output)

    def test_metadata_and_headers_are_escaped_without_changing_displayed_values(self):
        page = render_metadata_html(['name_hdf', '<header> @@COUNT@@'],
                                    [['α.hdf', '<script>alert("x")</script> & CTAB']])
        parser = TableParser()
        parser.feed(page)
        self.assertEqual(parser.rows, [['α.hdf', '<script>alert("x")</script> & CTAB']])
        self.assertEqual(parser.tags.count('script'), 1)  # Only our own sorting/search script.
        self.assertIn('<header> @@COUNT@@', parser.headers[1])
        self.assertIn('&lt;script&gt;', page)

    def test_empty_summary_is_valid_and_has_no_external_dependencies(self):
        path, _ = self.export({'name_hdf': [], 'scan': []})
        page = path.read_text(encoding='utf-8')
        parser = TableParser()
        parser.feed(page)
        self.assertEqual(parser.rows, [])
        self.assertEqual(len(parser.headers), 2)
        self.assertIn('0 measurements', page)
        self.assertNotIn('src="http', page)
        self.assertNotIn('href="http', page)

    def test_replace_error_leaves_prior_summary_and_no_temporary_copies(self):
        path = self.folder / 'all_files.html'
        path.write_text('previous complete summary', encoding='utf-8')
        with patch.object(preparation.os, 'replace', side_effect=PermissionError('locked HTML')):
            with self.assertRaises(PermissionError):
                write_metadata_html(str(path), ['name_hdf'], [['new.hdf']])
        self.assertEqual(path.read_text(encoding='utf-8'), 'previous complete summary')
        self.assertEqual(list(self.folder.iterdir()), [path])

    @unittest.skipUnless(os.name == 'nt', 'Requires a real Windows file-sharing lock')
    def test_locked_legacy_csv_does_not_block_pipeline_or_create_extra_csvs(self):
        raw = self.folder / 'raw_data'
        raw.mkdir()
        (raw / 'sans2022n000001.hdf').write_bytes(b'0' * 2048)
        analysis = self.folder / 'analysis'
        analysis.mkdir()
        legacy = analysis / 'all_files.csv'
        legacy.write_bytes(b'old Excel snapshot')
        configuration = {
            'experiment': {'sample_environment': {'electromagnet': False,
                            'water_bath': False, 'syringe_pumps': False}},
            'analysis': {'path_dir': str(self.folder), 'path_hdf_raw': str(raw), 'add_id': ''},
        }

        def fake_hdf(folder, filename, property):
            return {'counts': np.ones((2, 2)), 'sample_name': 'my sample',
                    'wl': 6.0}.get(property, 1)

        result = {'overview': {}}
        with windows_reader_lock(legacy), patch.object(preparation, 'load_hdf', side_effect=fake_hdf), \
                patch.object(preparation, 'load_thickness', return_value=0.1), redirect_stdout(StringIO()):
            table = preparation.list_files(configuration, result)
        self.assertEqual(table['scan'], [1])
        self.assertEqual(result['overview']['all_files']['scan'], [1])
        self.assertTrue((analysis / 'all_files.html').exists())
        self.assertTrue((analysis / 'config.npy').exists())
        self.assertTrue((analysis / 'result.npy').exists())
        self.assertEqual(legacy.read_bytes(), b'old Excel snapshot')
        self.assertEqual(list(analysis.glob('*.csv')), [legacy])


if __name__ == '__main__':
    unittest.main()
