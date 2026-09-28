#!/usr/bin/env python3
"""Write BIDS sub-<ID>/sub-<ID>_sessions.tsv files (session_id, acq_time) from an Excel table.

The table has one row per subject: an ID column (study-id / study_id, values like
sub-1 or 1) and date columns ses_1, ses_2, ... (Excel dates, ISO or DD.MM.YYYY
text). IDs are matched to the existing sub-* directories by number, so sub-1
matches sub-001. Only sessions that exist as ses-<N> directories are written.

Without --write this is a dry run: it reports what would change and every
problem, but writes nothing. Dates are not printed unless --show-dates is given.
Existing sessions.tsv files keep their other columns; only acq_time is set.

Needs only Python 3.8+ (no openpyxl/pandas):
  python3 make_sessions_tsv.py --excel Sessions.xlsx --bids /mnt/e/COHORT/BIDS
  python3 make_sessions_tsv.py --excel Sessions.xlsx --bids /mnt/e/COHORT/BIDS --write
"""
import argparse
import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
REL = '{http://schemas.openxmlformats.org/package/2006/relationships}'
ID_HEADERS = ('study-id', 'study_id', 'studyid', 'id', 'subject', 'participant_id')
SESSION_HEADER = re.compile(r'^ses[-_ ]?(\d+)$', re.IGNORECASE)
TEXT_FORMATS = ('%Y-%m-%d', '%d.%m.%Y', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M')
EARLIEST = date(1990, 1, 1)


def column_index(ref):
    letters = re.match(r'[A-Z]+', ref).group()
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - 64
    return index - 1


def read_xlsx(path, sheet_name=None):
    """Rows of cell values (str, float or None) of one worksheet."""
    with zipfile.ZipFile(path) as z:
        workbook = ET.fromstring(z.read('xl/workbook.xml'))
        pr = workbook.find('m:workbookPr', NS)
        date1904 = pr is not None and pr.get('date1904') in ('1', 'true')
        sheets = workbook.find('m:sheets', NS)
        sheet = next((s for s in sheets if sheet_name in (None, s.get('name'))), None)
        if sheet is None:
            raise SystemExit(f'Sheet {sheet_name!r} not found: {[s.get("name") for s in sheets]}')
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target = next(r.get('Target') for r in rels.iter(REL + 'Relationship')
                      if r.get('Id') == sheet.get('{%s}id' % NS['r']))
        target = target.lstrip('/')
        target = target if target.startswith('xl/') else 'xl/' + target
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            for item in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
                shared.append(''.join(t.text or '' for t in item.iter('{%s}t' % NS['m'])))
        rows = []
        for row in ET.fromstring(z.read(target)).find('m:sheetData', NS).findall('m:row', NS):
            values = {}
            for cell in row.findall('m:c', NS):
                kind = cell.get('t')
                raw = cell.find('m:v', NS)
                raw = raw.text if raw is not None else None
                if kind == 's':
                    value = shared[int(raw)]
                elif kind == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.iter('{%s}t' % NS['m']))
                elif kind in ('str', 'e', 'b'):
                    value = raw
                else:
                    value = float(raw) if raw not in (None, '') else None
                values[column_index(cell.get('r'))] = value
            rows.append([values.get(i) for i in range(max(values, default=-1) + 1)])
    return rows, date1904


def to_date(value, date1904=False):
    """ISO date from an Excel serial number or a date text; None for empty cells."""
    if value is None or (isinstance(value, str) and value.strip() in ('', 'n/a', 'NA', '-')):
        return None
    if isinstance(value, float):
        base = date(1904, 1, 1) if date1904 else date(1899, 12, 30)
        return base + timedelta(days=int(value))
    text = value.strip()
    for fmt in TEXT_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f'unreadable date {text!r} (expected an Excel date, YYYY-MM-DD or DD.MM.YYYY)')


def subject_number(value):
    """1 for 'sub-1', 'sub-001', '1' or the Excel number 1.0; None otherwise."""
    text = re.sub(r'\.0$', '', str(value).strip())
    match = re.fullmatch(r'(?:sub-)?0*(\d+)', text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_table(path, sheet=None):
    rows, date1904 = read_xlsx(path, sheet)
    if not rows:
        raise SystemExit('The sheet is empty')
    header = [str(h).strip() if h is not None else '' for h in rows[0]]
    id_col = next((i for i, h in enumerate(header) if h.lower() in ID_HEADERS), None)
    if id_col is None:
        raise SystemExit(f'No ID column ({", ".join(ID_HEADERS)}) in header {header}')
    session_cols = {i: int(m.group(1)) for i, h in enumerate(header) if (m := SESSION_HEADER.match(h))}
    if not session_cols:
        raise SystemExit(f'No session columns (ses_1, ses_2, ...) in header {header}')
    table, problems = {}, []
    for line, row in enumerate(rows[1:], start=2):
        row = row + [None] * (len(header) - len(row))
        if row[id_col] in (None, ''):
            if any(row[i] not in (None, '') for i in session_cols):
                problems.append(f'row {line}: dates without an ID')
            continue
        number = subject_number(row[id_col])
        if number is None:
            problems.append(f'row {line}: unreadable ID {row[id_col]!r}')
            continue
        if number in table:
            problems.append(f'row {line}: ID {row[id_col]} appears more than once')
            continue
        dates = {}
        for col, session in session_cols.items():
            try:
                parsed = to_date(row[col], date1904)
            except ValueError as error:
                problems.append(f'row {line} ({row[id_col]}), ses_{session}: {error}')
                continue
            if parsed is not None:
                dates[session] = parsed
        table[number] = {'id': str(row[id_col]).strip(), 'dates': dates}
    return table, problems


def check_dates(dates, today):
    issues = [f'ses-{s}: implausible date' for s, d in dates.items() if not EARLIEST <= d <= today]
    ordered = sorted(dates.items())
    issues += [f'ses-{b[0]} is not after ses-{a[0]}' for a, b in zip(ordered, ordered[1:]) if b[1] <= a[1]]
    return issues


def read_sessions_tsv(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields = [f.strip() for f in reader.fieldnames or []]
        rows = [{k.strip(): (v or '').strip() for k, v in row.items() if k} for row in reader]
    return fields, rows


def planned_tsv(path, sessions):
    """Header and rows after setting acq_time; keeps other columns and sessions of an existing file."""
    fields, rows = (read_sessions_tsv(path) if path.exists() else (['session_id', 'acq_time'], []))
    fields = ['session_id', 'acq_time'] + [f for f in fields if f not in ('session_id', 'acq_time')]
    by_id = {r.get('session_id'): r for r in rows}
    for session_id, acq_time in sessions.items():
        by_id.setdefault(session_id, {'session_id': session_id})['acq_time'] = acq_time
    number = lambda s: int(m.group(1)) if (m := re.match(r'ses-(\d+)$', s or '')) else 10 ** 9
    ordered = sorted(by_id.values(), key=lambda r: (number(r.get('session_id')), r.get('session_id')))
    return fields, [{f: r.get(f) or 'n/a' for f in fields} for r in ordered]


def render(fields, rows):
    lines = ['\t'.join(fields)] + ['\t'.join(r[f] for f in fields) for r in rows]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--excel', type=Path, required=True)
    p.add_argument('--bids', type=Path, required=True, help='BIDS root with sub-*/ses-* directories.')
    p.add_argument('--sheet', help='Worksheet name. Default: the first sheet.')
    p.add_argument('--write', action='store_true', help='Write the files. Default: dry run.')
    p.add_argument('--show-dates', action='store_true', help='Print the dates (personal data).')
    args = p.parse_args(argv)

    table, problems = read_table(args.excel, args.sheet)
    subjects = {}
    for d in sorted(args.bids.glob('sub-*')):
        number = subject_number(d.name) if d.is_dir() else None
        if number is not None:
            if number in subjects:
                raise SystemExit(f'{d.name} and {subjects[number].name} have the same number')
            subjects[number] = d

    today = date.today()
    counts = {'new': 0, 'changed': 0, 'unchanged': 0, 'skipped': 0}
    for number in sorted(set(table) - set(subjects)):
        problems.append(f'{table[number]["id"]}: in Excel, but no BIDS directory')
    for number, directory in sorted(subjects.items()):
        sub = directory.name
        sessions_on_disk = {int(m.group(1)): s.name for s in directory.glob('ses-*')
                            if s.is_dir() and (m := re.fullmatch(r'ses-(\d+)', s.name))}
        if number not in table:
            problems.append(f'{sub}: no row in Excel')
            continue
        dates = table[number]['dates']
        issues = check_dates(dates, today)
        issues += [f'{name}: no date in Excel' for n, name in sorted(sessions_on_disk.items()) if n not in dates]
        extra = sorted(set(dates) - set(sessions_on_disk))
        if extra:
            issues.append('dates without session directory: ' + ', '.join(f'ses_{n}' for n in extra))
        blocking = [i for i in issues if 'implausible' in i or 'not after' in i]
        for issue in issues:
            problems.append(f'{sub}: {issue}')
        if blocking:
            counts['skipped'] += 1
            continue
        sessions = {name: dates[n].isoformat() for n, name in sessions_on_disk.items() if n in dates}
        if not sessions:
            counts['skipped'] += 1
            continue
        target = directory / f'{sub}_sessions.tsv'
        content = render(*planned_tsv(target, sessions))
        old = target.read_text() if target.exists() else None
        state = 'new' if old is None else 'unchanged' if old.replace('\r\n', '\n') == content else 'changed'
        counts[state] += 1
        if args.show_dates or state != 'unchanged':
            line = f'{sub}: {state}, {len(sessions)} session(s)'
            print(line + (': ' + ', '.join(f'{k}={v}' for k, v in sorted(sessions.items())) if args.show_dates else ''))
        if args.write and state != 'unchanged':
            tmp = target.with_suffix('.tsv.tmp')
            tmp.write_text(content, newline='\n')
            tmp.replace(target)

    print()
    for problem in problems:
        print('WARNING', problem)
    print(f'\nExcel rows: {len(table)}, BIDS subjects: {len(subjects)}; files new: {counts["new"]}, '
          f'changed: {counts["changed"]}, unchanged: {counts["unchanged"]}, skipped: {counts["skipped"]}')
    if not args.write:
        print('Dry run: nothing written. Add --write to write the files.')
    return 1 if counts['skipped'] else 0


if __name__ == '__main__':
    sys.exit(main())
