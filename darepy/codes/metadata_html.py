"""Standalone, browser-friendly metadata summaries; no third-party assets."""
from datetime import datetime
from html import escape
import json
import os
import re
import tempfile


def render_metadata_html(keys, rows, configured_exclusions=None):
    """Escape every metadata value before embedding it into the HTML table."""
    keys = list(keys)
    rows = [list(row) for row in rows]
    excluded_index = keys.index('excluded') if 'excluded' in keys else None
    scan_index = keys.index('scan') if 'scan' in keys else None
    shown_indices = [index for index, key in enumerate(keys) if key != 'excluded']
    headings = ''.join(
        '<th scope="col" aria-sort="none"><button type="button" data-column="'
        + str(display_index + 1) + '">' + escape(str(keys[index])) + '<span aria-hidden="true"> ↕</span></button></th>'
        for display_index, index in enumerate(shown_indices)
    )
    body = []
    for row in rows:
        if scan_index is not None:
            scan = escape(str(row[scan_index]))
            checked = ' checked' if excluded_index is not None and row[excluded_index] else ''
            checkbox = (
                '<td class="exclude-cell"><input class="exclude-check" type="checkbox" '
                'data-scan="' + scan + '" aria-label="Exclude scan ' + scan + ' from radial integration"'
                + checked + '></td>'
            )
        else:
            checkbox = '<td></td>'
        cells = ''.join(
            '<td>' + escape('' if row[index] is None else str(row[index])) + '</td>'
            for index in shown_indices
        )
        body.append('<tr>' + checkbox + cells + '</tr>')
    updated = escape(datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
    if configured_exclusions is None:
        configured_exclusions = []
        if excluded_index is not None and scan_index is not None:
            configured_exclusions = [
                str(row[scan_index]) for row in rows if row[excluded_index]
            ]
    replacements = {'HEADINGS': headings, 'UPDATED': updated,
                    'COUNT': str(len(body)), 'ROWS': '\n'.join(body),
                    'CONFIG_EXCLUSIONS': json.dumps(configured_exclusions)}
    return re.sub(r'@@(HEADINGS|UPDATED|COUNT|ROWS|CONFIG_EXCLUSIONS)@@',
                  lambda match: replacements[match.group(1)], _TEMPLATE)


def write_metadata_html(save_file, keys, rows, configured_exclusions=None):
    """Replace the single summary only after the complete page is written."""
    page = render_metadata_html(keys, rows, configured_exclusions=configured_exclusions)
    directory = os.path.dirname(os.path.abspath(save_file))
    descriptor, temporary = tempfile.mkstemp(prefix='.all_files_', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
            stream.write(page)
        os.replace(temporary, save_file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return save_file


_TEMPLATE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DarePy · Beamtime summary</title>
<style>
:root {color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; color: #1a2c3f; background: #f3f6fa;}
body {margin: 0; padding: 30px;}
main {max-width: 1800px; margin: auto;}
.eyebrow {font-size: 12px; letter-spacing: .16em; color: #36679d; font-weight: 700;}
h1 {font-size: 30px; margin: 8px 0;}
.description {color: #52657a; line-height: 1.6; margin: 0 0 22px;}
.toolbar {display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 18px 0;}
label {font-weight: 600;}
input {padding: 10px 12px; border: 1px solid #a8b7c7; border-radius: 6px; font: inherit; width: min(380px, 70vw);}
button {font: inherit; cursor: pointer;}
#refresh {background: #1d588c; color: white; border: 0; border-radius: 6px; padding: 11px 16px;}
#save-exclusions {background: #2196F3; color: white; border: 0; border-radius: 6px; padding: 11px 16px;}
#count {color: #52657a; font-size: 14px;}
.table-wrap {overflow: auto; max-height: 72vh; border: 1px solid #d3dde7; border-radius: 8px; background: white;}
table {border-collapse: separate; border-spacing: 0; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums;}
caption {text-align: left; padding: 12px 16px; color: #52657a; font-size: 12px;}
th {position: sticky; top: 0; background: #e8eff7; text-align: left; border-bottom: 1px solid #c2d1df; z-index: 1;}
th button {border: 0; background: transparent; color: #244766; padding: 13px 14px; white-space: nowrap; font-weight: 600; width: 100%; text-align: left;}
th[aria-sort="ascending"] span::after {content: " ↑";}
th[aria-sort="descending"] span::after {content: " ↓";}
th[aria-sort="ascending"] span, th[aria-sort="descending"] span {font-size: 0;}
th span::after {font-size: 13px;}
td {padding: 10px 14px; border-bottom: 1px solid #edf1f5; white-space: nowrap;}
.exclude-cell {text-align: center; width: 76px;}
.exclude-check {width: 18px; height: 18px; padding: 0; border: 0; accent-color: #1976D2; cursor: pointer;}
tbody tr.excluded td {background: #fff1f0; color: #8b2323;}
tbody tr:nth-child(even) {background: #f8fafc;}
tbody tr:hover {background: #edf5ff;}
tr[hidden] {display: none;}
#empty {padding: 20px; color: #52657a;}
footer {font-size: 12px; color: #61758a; margin-top: 14px;}
:focus-visible {outline: 3px solid #dc8b22; outline-offset: 2px;}
@media (max-width: 650px) {body {padding: 16px;} h1 {font-size: 25px;}}
</style>
</head>
<body><main>
<div class="eyebrow">DAREPY / EXPERIMENT METADATA</div>
<h1>Beamtime summary</h1>
<p class="description">All indexed measurements in one place. Keep this page open while DarePy runs.<br>
Refresh after processing to view the latest saved summary.</p>
<div class="toolbar">
<label for="search">Find measurements</label>
<input id="search" type="search" placeholder="Sample name, scan, distance…" autocomplete="off">
<label><input id="hide-excluded" type="checkbox" style="width:auto"> Hide excluded</label>
<button id="save-exclusions" type="button">Make Exclude Array</button>
<button id="refresh" type="button">Refresh summary</button>
<span id="count" role="status" aria-live="polite">@@COUNT@@ measurements</span>
</div>
<p id="save-status" class="description" role="status">Checked scans are excluded from radial integration. Make the list, then paste it into <code>exclude_files</code> in DarePy’s Pipeline Control tab or config YAML and save the settings. Consecutive scans are shown as ranges, for example <code>1:10, 14</code>.</p>
<textarea id="exclusion-array" aria-label="Paste-ready excluded scan numbers" hidden rows="3" style="width:100%;box-sizing:border-box;font:14px Consolas,monospace;margin-bottom:14px"></textarea>
<div class="table-wrap">
<table id="metadata-table">
<caption>Click a column heading to sort. Updated @@UPDATED@@.</caption>
<thead><tr><th scope="col" class="exclude-heading">Exclude<br>
<button id="mark-all" type="button">Mark all</button>
<button id="unmark-all" type="button">Unmark all</button></th>@@HEADINGS@@</tr></thead>
<tbody>@@ROWS@@</tbody>
</table>
<p id="empty" hidden>No measurements match your search.</p>
</div>
<footer>Local summary · No internet connection required · Values and units follow the DarePy metadata columns.</footer>
<noscript><p>Enable JavaScript for search and sorting. All measurements are displayed below the headings.</p></noscript>
</main>
<script>
'use strict';
const table = document.getElementById('metadata-table');
const tbody = table.tBodies[0];
const rows = Array.from(tbody.rows);
const search = document.getElementById('search');
const count = document.getElementById('count');
const empty = document.getElementById('empty');
const hideExcluded = document.getElementById('hide-excluded');
const status = document.getElementById('save-status');
const configuredExclusions = new Set(@@CONFIG_EXCLUSIONS@@.map(String));
const searchable = new Map(rows.map(row => [row, Array.from(row.cells, cell => cell.textContent).join(' ').toLocaleLowerCase()]));
function filterRows() {
  const query = search.value.trim().toLocaleLowerCase();
  let visible = 0;
  rows.forEach(row => {
    const checked = row.querySelector('.exclude-check').checked;
    row.classList.toggle('excluded', checked);
    row.hidden = !searchable.get(row).includes(query) || (hideExcluded.checked && checked);
    if (!row.hidden) visible++;
  });
  count.textContent = visible + ' of ' + rows.length + ' measurements';
  empty.hidden = visible !== 0;
}
search.addEventListener('input', filterRows);
hideExcluded.addEventListener('change', filterRows);
table.querySelectorAll('.exclude-check').forEach(box => box.addEventListener('change', filterRows));
document.getElementById('mark-all').addEventListener('click', () => {
  table.querySelectorAll('.exclude-check').forEach(box => { box.checked = true; });
  filterRows();
});
document.getElementById('unmark-all').addEventListener('click', () => {
  table.querySelectorAll('.exclude-check').forEach(box => { box.checked = false; });
  filterRows();
});
document.getElementById('save-exclusions').addEventListener('click', () => {
  const known = new Set(rows.map(row => row.querySelector('.exclude-check').dataset.scan));
  const selected = new Set([...configuredExclusions].filter(scan => !known.has(scan)));
  rows.forEach(row => {
    const checkbox = row.querySelector('.exclude-check');
    if (checkbox.checked) selected.add(checkbox.dataset.scan);
  });
  const scans = Array.from(selected, Number).sort((a, b) => a - b);
  const ranges = [];
  for (let index = 0; index < scans.length;) {
    const first = scans[index];
    let last = first;
    while (index + 1 < scans.length && scans[index + 1] === last + 1) {
      last = scans[++index];
    }
    ranges.push(first === last ? String(first) : first + ':' + last);
    index++;
  }
  const arrayText = ranges.join(', ');
  const output = document.getElementById('exclusion-array');
  output.value = arrayText;
  output.hidden = false;
  output.focus();
  output.select();
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(arrayText).then(() => {
      status.textContent = 'Exclude list copied. Paste it into exclude_files in Pipeline Control or your config YAML, then save. Run “List All Measurements” afterward to refresh the marks.';
    }).catch(() => {
      status.textContent = 'Exclude list is selected below. Press Ctrl+C, then paste it into exclude_files in Pipeline Control or your config YAML and save.';
    });
  } else {
    status.textContent = 'Exclude list is selected below. Press Ctrl+C, then paste it into exclude_files in Pipeline Control or your config YAML and save.';
  }
});
document.getElementById('refresh').addEventListener('click', () => window.location.reload());
table.querySelectorAll('th button').forEach(button => {
  button.addEventListener('click', () => {
    const header = button.parentElement;
    const column = Number(button.dataset.column);
    const ascending = header.getAttribute('aria-sort') !== 'ascending';
    table.querySelectorAll('th').forEach(th => th.setAttribute('aria-sort', 'none'));
    header.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
    const ordered = Array.from(tbody.rows);
    ordered.sort((a, b) => {
      const left = a.cells[column].textContent.trim();
      const right = b.cells[column].textContent.trim();
      if (!left || !right) return !left && !right ? 0 : !left ? 1 : -1;
      const numeric = Number.isFinite(Number(left)) && Number.isFinite(Number(right));
      const comparison = numeric ? Number(left) - Number(right) : left.localeCompare(right, undefined, {numeric: true, sensitivity: 'base'});
      return ascending ? comparison : -comparison;
    });
    const fragment = document.createDocumentFragment();
    ordered.forEach(row => fragment.appendChild(row));
    tbody.appendChild(fragment);
  });
});
filterRows();
</script>
</body>
</html>
'''
