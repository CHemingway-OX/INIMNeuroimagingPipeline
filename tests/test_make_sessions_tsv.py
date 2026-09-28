"""sessions.tsv generation from an Excel table, with a synthetic workbook."""
import contextlib
from datetime import date
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hpc'))
import make_sessions_tsv as tool

M = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def serial(d):
    return (d - date(1899, 12, 30)).days


def column(i):
    return chr(ord('A') + i)


def write_xlsx(path, rows):
    """Minimal workbook: strings as shared strings, numbers as numeric cells."""
    strings = []
    sheet_rows = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(row):
            ref = f'{column(c)}{r}'
            if value is None:
                continue
            if isinstance(value, str):
                strings.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>')
            else:
                cells.append(f'<c r="{ref}" s="1"><v>{value}</v></c>')
        sheet_rows.append(f'<row r="{r}">{"".join(cells)}</row>')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{M}" xmlns:r="{R}"><sheets>'
                   f'<sheet name="Tabelle1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels',
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{M}"><sheetData>{"".join(sheet_rows)}</sheetData></worksheet>')
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{M}">' + ''.join(f'<si><t>{s}</t></si>' for s in strings) + '</sst>')


class SessionsTsvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.bids = self.base / 'bids'
        for sub, sessions in (('sub-001', 3), ('sub-005', 2), ('sub-007', 2), ('sub-009', 1)):
            for s in range(1, sessions + 1):
                (self.bids / sub / f'ses-{s}' / 'anat').mkdir(parents=True)
        self.excel = self.base / 'sessions.xlsx'
        write_xlsx(self.excel, [
            ['study-id', 'ses_1', 'ses_2', 'ses_3', 'ses_4'],
            ['sub-1', serial(date(2020, 1, 2)), serial(date(2021, 2, 3)), serial(date(2022, 3, 4)), None],
            ['sub-5', '05.10.2024', '2025-10-06', None, None],
            ['sub-7', serial(date(2023, 1, 1)), serial(date(2022, 1, 1)), None, None],  # out of order
            ['sub-42', serial(date(2020, 1, 1)), None, None, None],                      # no BIDS dir
        ])

    def run_tool(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = tool.main(['--excel', str(self.excel), '--bids', str(self.bids), *extra])
        return code, out.getvalue()

    def test_dry_run_writes_nothing_and_reports_problems(self):
        code, out = self.run_tool()
        self.assertFalse(list(self.bids.glob('sub-*/*_sessions.tsv')))
        self.assertEqual(code, 1)  # sub-007 skipped
        self.assertIn('sub-007: ses-2 is not after ses-1', out)
        self.assertIn('sub-42: in Excel, but no BIDS directory', out)
        self.assertIn('sub-009: no row in Excel', out)
        self.assertIn('Dry run', out)
        self.assertNotIn('2020-01-02', out)  # dates only with --show-dates

    def test_write_maps_ids_and_converts_dates(self):
        self.run_tool('--write')
        self.assertEqual((self.bids / 'sub-001/sub-001_sessions.tsv').read_bytes(),
                         b'session_id\tacq_time\nses-1\t2020-01-02\nses-2\t2021-02-03\nses-3\t2022-03-04\n')
        self.assertEqual((self.bids / 'sub-005/sub-005_sessions.tsv').read_text(),
                         'session_id\tacq_time\nses-1\t2024-10-05\nses-2\t2025-10-06\n')
        self.assertFalse((self.bids / 'sub-007/sub-007_sessions.tsv').exists())
        _, out = self.run_tool('--write')
        self.assertIn('unchanged: 2', out)

    def test_existing_file_keeps_columns_and_replaces_placeholders(self):
        target = self.bids / 'sub-005/sub-005_sessions.tsv'
        target.write_text('session_id\tacq_time\tage\r\nses-1\t1999-01-01\t40\r\nses-2\t1999-06-01\t41')
        _, out = self.run_tool('--write')
        self.assertIn('sub-005: changed', out)
        self.assertEqual(target.read_text(),
                         'session_id\tacq_time\tage\nses-1\t2024-10-05\t40\nses-2\t2025-10-06\t41\n')

    def test_missing_date_for_existing_session_is_reported(self):
        (self.bids / 'sub-005/ses-3/anat').mkdir(parents=True)
        _, out = self.run_tool()
        self.assertIn('sub-005: ses-3: no date in Excel', out)

    def test_ids_and_dates(self):
        self.assertEqual([tool.subject_number(v) for v in ('sub-1', 'sub-001', '12', 12.0, 'x-1')], [1, 1, 12, 12, None])
        self.assertEqual(tool.to_date(float(serial(date(2024, 10, 5))) + .5), date(2024, 10, 5))
        with self.assertRaises(ValueError):
            tool.to_date('10/05/2024')  # ambiguous US/European order is refused


if __name__ == '__main__':
    unittest.main()
