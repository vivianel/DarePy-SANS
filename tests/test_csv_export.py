"""Metadata exports stay usable while Excel holds a Windows sharing lock."""
from contextlib import contextmanager, redirect_stdout
import csv
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Pipeline callers import these modules directly from darepy/codes as well.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'darepy' / 'codes'))
import prepare_input as preparation


@contextmanager
def windows_reader_lock(path):
    """Deny write/delete sharing, as an open Excel CSV can do."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel.CloseHandle(handle)


class CsvExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.target = self.folder / 'det_files_6p0.csv'
        self.table = {'name_hdf': ['sans2022n000001.hdf'], 'scan': [1],
                      'sample_name': ['sample, with comma']}

    def export(self, name='det_files_6p0', table=None):
        with redirect_stdout(StringIO()) as output:
            actual = preparation.save_list_files(str(self.folder), str(self.folder), table or self.table, name)
        return Path(actual), output.getvalue()

    def read_table(self, path):
        with path.open(newline='') as stream:
            return list(csv.reader(stream))

    def test_normal_export_replaces_usual_file_without_extra_copies(self):
        self.target.write_text('old table', encoding='utf-8')
        actual, output = self.export()
        self.assertEqual(actual, self.target)
        self.assertEqual(self.read_table(actual)[1], ['sans2022n000001.hdf', '1', 'sample, with comma'])
        self.assertEqual(list(self.folder.glob('*.csv')), [self.target])
        self.assertIn(str(actual), output)

    def test_permission_error_keeps_original_and_publishes_complete_copy(self):
        self.target.write_bytes(b'original table\r\n')
        with patch.object(preparation.os, 'replace', side_effect=PermissionError('locked')) as replace:
            actual, output = self.export()
        self.assertEqual(replace.call_count, 1)  # No waiting or retry loop.
        self.assertNotEqual(actual, self.target)
        self.assertTrue(actual.name.startswith('det_files_6p0_'))
        self.assertEqual(self.target.read_bytes(), b'original table\r\n')
        self.assertEqual(self.read_table(actual)[0], list(self.table))
        self.assertEqual(self.read_table(actual)[1][-1], 'sample, with comma')
        self.assertIn(str(actual), output)
        self.assertIn('Processing continues', output)

    def test_repeated_locked_exports_keep_distinct_complete_copies(self):
        with patch.object(preparation.os, 'replace', side_effect=PermissionError('locked')):
            first, _ = self.export()
            second, _ = self.export()
        self.assertNotEqual(first, second)
        self.assertEqual(self.read_table(first), self.read_table(second))

    def test_detector_csv_uses_same_fallback(self):
        with patch.object(preparation.os, 'replace', side_effect=PermissionError('locked')):
            actual, _ = self.export(name='det_files_6p0')
        self.assertTrue(actual.name.startswith('det_files_6p0_'))

    def test_real_io_errors_propagate_and_remove_incomplete_output(self):
        self.target.write_bytes(b'keep original')
        with patch.object(preparation.os, 'replace', side_effect=OSError('disk error')):
            with self.assertRaisesRegex(OSError, 'disk error'):
                self.export()
        self.assertEqual(self.target.read_bytes(), b'keep original')
        self.assertEqual(list(self.folder.glob('*.csv')), [self.target])

    def test_error_while_writing_preserves_original(self):
        self.target.write_bytes(b'keep original')

        def bad_rows():
            yield ['first row']
            raise OSError('disk full')

        with self.assertRaisesRegex(OSError, 'disk full'):
            preparation._write_metadata_csv(str(self.target), ['header'], bad_rows())
        self.assertEqual(self.target.read_bytes(), b'keep original')
        self.assertEqual(list(self.folder.glob('*.csv')), [self.target])

    @unittest.skipUnless(os.name == 'nt', 'Requires a real Windows file-sharing lock')
    def test_actual_windows_lock_then_normal_export_after_closing(self):
        self.target.write_bytes(b'Excel has this open')
        with windows_reader_lock(self.target):
            actual, _ = self.export()
            self.assertNotEqual(actual, self.target)
            self.assertEqual(self.target.read_bytes(), b'Excel has this open')
            self.assertEqual(self.read_table(actual)[0], list(self.table))
        normal, _ = self.export()
        self.assertEqual(normal, self.target)
        self.assertEqual(self.read_table(normal), self.read_table(actual))


if __name__ == '__main__':
    unittest.main()
