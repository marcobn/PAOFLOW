// Plotly rendering of the server's figure specifications: theme colours and chrome, table view.

import { api } from './api.js';
import { h } from './dom.js';

let plotlyPromise = null;
let palette = null;

export function loadPlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (!plotlyPromise) {
    plotlyPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = '/vendor/plotly.min.js';
      script.onload = () => resolve(window.Plotly);
      script.onerror = () => reject(new Error('Plotly is not available: pip install "PAOFLOW[gui]"'));
      document.head.append(script);
    });
  }
  return plotlyPromise;
}

export async function loadPalette() {
  if (!palette) palette = await api.get('/api/results/palette');
  return palette;
}

function theme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

// Replace colour roles (series-3, ord-2, neutral, axis, role:sequential) by the theme's colours.
export function resolveRoles(spec, mode = theme()) {
  const table = palette[mode];
  const role = (value) => {
    if (value.startsWith('series-')) return table.series[(Number(value.split('-')[1]) - 1) % table.series.length];
    if (value.startsWith('ord-')) return table.ordinal[(Number(value.split('-')[1]) - 1) % table.ordinal.length];
    return typeof table[value] === 'string' ? table[value] : value;
  };
  const copy = structuredClone(spec);
  const walk = (node) => {
    if (Array.isArray(node)) node.forEach(walk);
    else if (node && typeof node === 'object') {
      for (const key of Object.keys(node)) {
        const value = node[key];
        if (key === 'color' && typeof value === 'string') node[key] = role(value);
        else if (key === 'colorscale' && typeof value === 'string' && value.startsWith('role:')) node[key] = table.scales[value.slice(5)];
        else walk(value);
      }
    }
  };
  walk(copy);
  const layout = copy.layout;
  Object.assign(layout, {
    paper_bgcolor: table.surface,
    plot_bgcolor: table.surface,
    font: { family: getComputedStyle(document.body).fontFamily, color: table.text, size: 12 },
    hoverlabel: { bgcolor: table.surface, bordercolor: table.axis, font: { color: table.text } },
  });
  if (layout.title) layout.title.font = { ...(layout.title.font || {}), color: table.text };
  for (const name of ['xaxis', 'yaxis']) {
    if (layout[name]) Object.assign(layout[name], { gridcolor: table.grid, linecolor: table.axis, tickcolor: table.axis, zerolinecolor: table.axis });
  }
  if (layout.scene) {
    for (const name of ['xaxis', 'yaxis', 'zaxis']) Object.assign(layout.scene[name], { color: table.muted });
  }
  if (layout.legend) layout.legend.font = { color: table.muted };
  return copy;
}

export async function renderFigure(node, spec, filename = 'paoflow', { keepTitle = false } = {}) {
  const Plotly = await loadPlotly();
  await loadPalette();
  const resolved = resolveRoles(spec);
  if (!keepTitle) {
    // The tab shows the title above the figure; keep the plot area for the data.
    delete resolved.layout.title;
    resolved.layout.margin = { ...(resolved.layout.margin || {}), t: 16 };
  }
  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    toImageButtonOptions: { format: 'svg', filename },
  };
  await Plotly.react(node, resolved.data, resolved.layout, config);
  node.dataset.rendered = 'true';
}

export function purge(node) {
  if (window.Plotly && node.dataset.rendered) window.Plotly.purge(node);
  delete node.dataset.rendered;
}

// Table view twin of a line figure: x and y of every trace (bands: the first 500 points).
export function tableView(spec, limit = 500) {
  const traces = (spec.data || []).filter((t) => t.type === 'scatter' && Array.isArray(t.x));
  if (!traces.length) return h('div', { class: 'empty' }, 'No table view for this figure (use the CSV export).');
  const table = h('table', { class: 'data-table' });
  table.append(h('tr', {}, h('th', {}, 'series'), h('th', {}, 'x'), h('th', {}, 'y')));
  let rows = 0;
  for (const trace of traces) {
    for (let i = 0; i < trace.x.length && rows < limit; i += 1) {
      if (trace.x[i] === null || trace.y[i] === null) continue;
      table.append(h('tr', {}, h('td', {}, trace.name), h('td', {}, Number(trace.x[i]).toPrecision(6)), h('td', {}, Number(trace.y[i]).toPrecision(6))));
      rows += 1;
    }
  }
  return h('div', {}, table, rows >= limit ? h('div', { class: 'small muted' }, `first ${limit} rows; the CSV export has everything`) : null);
}
