"""A self-contained HTML report of a results folder (offline: Plotly and the figures inlined).

``paoflow-studio report <folder> -o report.html`` (or :func:`write_report`) discovers the datasets,
builds every figure with its default options and writes one HTML file with the Plotly bundle of
the installed ``plotly`` package, the figure specifications, a light/dark switch and a table view
per figure.  The file needs no server and no network, so it can be mailed, archived or mirrored.
"""

from __future__ import annotations

import html
import json
import os
import time

from .discovery import discover
from .figures import figure_spec
from .loaders import load
from .palette import PALETTE

#: Datasets of these families are listed but not drawn (text, arrays, large 3D meshes).
NOT_DRAWN = ('text', 'npz')

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --text:#1c2330; --muted:#5d6778; --border:#dde1e7; }}
:root[data-theme=dark] {{ --bg:#0f1218; --card:#171b23; --text:#e4e8ef; --muted:#a3acbb; --border:#2c3442; }}
body {{ margin:0; font:14px/1.45 system-ui,-apple-system,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); }}
header {{ padding:18px 28px; border-bottom:1px solid var(--border); background:var(--card); display:flex; gap:16px; align-items:baseline; }}
header h1 {{ margin:0; font-size:20px; }} header .meta {{ color:var(--muted); flex:1; }}
main {{ padding:20px 28px; display:grid; grid-template-columns:repeat(auto-fill,minmax(560px,1fr)); gap:18px; }}
section {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px 12px; }}
section h2 {{ font-size:14px; margin:2px 0 6px; }} .where {{ color:var(--muted); font-size:12px; font-family:ui-monospace,monospace; }}
.plot {{ height:380px; }} pre {{ white-space:pre-wrap; font-size:12px; max-height:340px; overflow:auto; }}
table {{ border-collapse:collapse; font-size:12px; }} td,th {{ padding:2px 8px; border-bottom:1px solid var(--border); text-align:left; }}
details summary {{ cursor:pointer; color:var(--muted); font-size:12.5px; }} button {{ font:inherit; }}
.cat {{ grid-column:1/-1; font-size:12px; letter-spacing:.6px; text-transform:uppercase; color:var(--muted); margin-top:8px; }}
</style>
<script>{plotly}</script>
</head><body>
<header><h1>{title}</h1><div class="meta">{meta}</div><button id="theme">light / dark</button></header>
<main id="main"></main>
<script>
const PALETTE = {palette};
const FIGURES = {figures};
function resolve(spec, theme) {{
  const t = PALETTE[theme];
  const role = (v) => v.startsWith('series-') ? t.series[(+v.split('-')[1] - 1) % 8]
    : v.startsWith('ord-') ? t.ordinal[(+v.split('-')[1] - 1) % 5] : (typeof t[v] === 'string' ? t[v] : v);
  const walk = (n) => {{
    if (Array.isArray(n)) {{ n.forEach(walk); return; }}
    if (n && typeof n === 'object') for (const k of Object.keys(n)) {{
      const v = n[k];
      if (k === 'color' && typeof v === 'string') n[k] = role(v);
      else if (k === 'colorscale' && typeof v === 'string' && v.startsWith('role:')) n[k] = t.scales[v.slice(5)];
      else walk(v);
    }}
  }};
  const copy = JSON.parse(JSON.stringify(spec));
  walk(copy);
  const L = copy.layout;
  Object.assign(L, {{ paper_bgcolor: t.surface, plot_bgcolor: t.surface, font: {{ color: t.text, family: 'system-ui, sans-serif', size: 12 }} }});
  for (const a of ['xaxis', 'yaxis']) if (L[a]) Object.assign(L[a], {{ gridcolor: t.grid, linecolor: t.axis, tickcolor: t.axis }});
  return copy;
}}
function table(spec) {{
  const box = document.createElement('table');
  const rows = [];
  for (const tr of spec.data.slice(0, 8)) if (tr.x && tr.y && tr.type === 'scatter')
    for (let i = 0; i < Math.min(tr.x.length, 60); i++) if (tr.x[i] !== null) rows.push([tr.name, tr.x[i], tr.y[i]]);
  box.innerHTML = '<tr><th>series</th><th>x</th><th>y</th></tr>';
  for (const r of rows) {{ const tr = document.createElement('tr'); for (const c of r) {{ const td = document.createElement('td'); td.textContent = typeof c === 'number' ? c.toPrecision(5) : c; tr.append(td); }} box.append(tr); }}
  return box;
}}
let theme = (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
function render() {{
  document.documentElement.dataset.theme = theme;
  const main = document.getElementById('main');
  main.textContent = '';
  let category = null;
  for (const f of FIGURES) {{
    if (f.category !== category) {{ category = f.category; const c = document.createElement('div'); c.className = 'cat'; c.textContent = category; main.append(c); }}
    const s = document.createElement('section');
    const h = document.createElement('h2'); h.textContent = f.title; s.append(h);
    const w = document.createElement('div'); w.className = 'where'; w.textContent = f.files.join(', '); s.append(w);
    if (f.spec) {{
      const p = document.createElement('div'); p.className = 'plot'; s.append(p);
      Plotly.newPlot(p, resolve(f.spec, theme).data, resolve(f.spec, theme).layout, {{ responsive: true, displaylogo: false }});
      if (f.spec.data.some((t) => t.type === 'scatter')) {{ const d = document.createElement('details'); const sm = document.createElement('summary'); sm.textContent = 'table view (first points)'; d.append(sm, table(f.spec)); s.append(d); }}
    }} else if (f.text) {{ const pre = document.createElement('pre'); pre.textContent = f.text; s.append(pre); }}
    for (const n of f.notes) {{ const d = document.createElement('div'); d.className = 'where'; d.textContent = n; s.append(d); }}
    main.append(s);
  }}
}}
document.getElementById('theme').addEventListener('click', () => {{ theme = theme === 'dark' ? 'light' : 'dark'; render(); }});
render();
</script></body></html>
"""


def build_report(folder: str, title: str | None = None, kinds: list[str] | None = None) -> str:
    """HTML of the report of ``folder``.

    Parameters
    ----------
    folder : str
        Results folder.
    title : str, optional
        Report title (default: the folder name).
    kinds : list of str, optional
        Only these dataset kinds.

    Returns
    -------
    str

    Raises
    ------
    ImportError
        When ``plotly`` (its JavaScript bundle) is not installed.
    """
    from ..gui.server import plotly_bundle

    bundle = plotly_bundle()
    if bundle is None:
        raise ImportError('the report embeds Plotly: pip install "PAOFLOW[gui]"')
    folder = os.path.abspath(folder)
    figures = []
    for ref in discover(folder):
        if kinds and ref.kind not in kinds:
            continue
        entry = {
            'title': ref.title,
            'category': ref.category,
            'files': ref.files,
            'notes': [],
            'spec': None,
            'text': None,
        }
        if ref.family in NOT_DRAWN:
            entry['notes'].append('listed only (%s)' % ref.family)
        else:
            try:
                dataset = load(ref, folder)
                entry['spec'] = figure_spec(dataset)
                entry['notes'] = dataset.notes
                if entry['spec'] is None and dataset.text:
                    entry['text'] = dataset.text[:4000]
            except Exception as exc:  # report and continue
                entry['notes'].append('not drawn: %s' % exc)
        figures.append(entry)
    from ..gui.server import finite

    meta = '%s · %d datasets · %s' % (folder, len(figures), time.strftime('%Y-%m-%d %H:%M'))
    return TEMPLATE.format(
        title=html.escape(title or os.path.basename(folder) or 'PAOFLOW results'),
        meta=html.escape(meta),
        plotly=bundle.read_text(encoding='utf-8').replace('</script>', '<\\/script>'),
        palette=json.dumps(PALETTE),
        figures=json.dumps(finite(figures), allow_nan=False).replace('</', '<\\/'),
    )


def write_report(
    folder: str, output: str, title: str | None = None, kinds: list[str] | None = None
) -> str:
    """Write the report of ``folder`` to ``output``.

    Parameters
    ----------
    folder : str
        Results folder.
    output : str
        HTML file.
    title : str, optional
        Report title.
    kinds : list of str, optional
        Only these dataset kinds.

    Returns
    -------
    str
        Absolute path written.
    """
    text = build_report(folder, title=title, kinds=kinds)
    with open(output, 'w', encoding='utf-8') as handle:
        handle.write(text)
    return os.path.abspath(output)
