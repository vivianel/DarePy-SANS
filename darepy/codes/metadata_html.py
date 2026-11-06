"""Standalone, browser-friendly metadata summaries; no third-party assets."""
from datetime import datetime
from html import escape
import os
import re
import tempfile


def render_metadata_html(keys, rows):
    """Escape every metadata value before embedding it into the HTML table."""
    keys = list(keys)
    headings = ''.join(
        '<th scope="col" aria-sort="none"><button type="button" data-column="'
        + str(index) + '">' + escape(str(key)) + '<span aria-hidden="true"> ↕</span></button></th>'
        for index, key in enumerate(keys)
    )
    body = []
    for row in rows:
        cells = ''.join('<td>' + escape('' if value is None else str(value)) + '</td>' for value in row)
        body.append('<tr>' + cells + '</tr>')
    updated = escape(datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
    replacements = {'HEADINGS': headings, 'UPDATED': updated,
                    'COUNT': str(len(body)), 'ROWS': '\n'.join(body)}
    return re.sub(r'@@(HEADINGS|UPDATED|COUNT|ROWS)@@',
                  lambda match: replacements[match.group(1)], _TEMPLATE)


def write_metadata_html(save_file, keys, rows):
    """Replace the single summary only after the complete page is written."""
    page = render_metadata_html(keys, rows)
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
<button id="refresh" type="button">Refresh summary</button>
<span id="count" role="status" aria-live="polite">@@COUNT@@ measurements</span>
</div>
<div class="table-wrap">
<table id="metadata-table">
<caption>Click a column heading to sort. Updated @@UPDATED@@.</caption>
<thead><tr>@@HEADINGS@@</tr></thead>
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
const searchable = new Map(rows.map(row => [row, Array.from(row.cells, cell => cell.textContent).join(' ').toLocaleLowerCase()]));
function filterRows() {
  const query = search.value.trim().toLocaleLowerCase();
  let visible = 0;
  rows.forEach(row => {
    row.hidden = !searchable.get(row).includes(query);
    if (!row.hidden) visible++;
  });
  count.textContent = visible + ' of ' + rows.length + ' measurements';
  empty.hidden = visible !== 0;
}
search.addEventListener('input', filterRows);
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
