// Results tab: datasets of a folder, interactive figures, options, comparison, exports.

import { api } from './api.js';
import { pickFolder } from './browser.js';
import { codeView } from './codeview.js';
import {
  clear,
  copyText,
  debounce,
  dialog,
  formatBytes,
  formatTime,
  h,
  storage,
  toast,
} from './dom.js';
import { loadPalette, purge, renderFigure, roleColor, tableView } from './plots.js';

const LIVE_INTERVAL = 5000;

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = h('a', { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function postForBlob(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = (await response.json()).error || message;
    } catch {
      /* not JSON */
    }
    throw new Error(message);
  }
  return response.blob();
}

// ------------------------------------------------------------------------- axis limits
// The y axes of a figure ('y', 'y2', ... of the stacked panels of an overlay) after 'x'.
function figureAxes(figure) {
  const layout = figure?.layout || {};
  if (!layout.xaxis || layout.scene) return [];
  const ys = Object.keys(layout)
    .filter((key) => /^yaxis\d*$/.test(key))
    .map((key) => `y${key.slice(5)}`)
    .sort();
  return ['x', ...(ys.length ? ys : ['y'])];
}

function layoutKey(axis) {
  return `${axis[0]}axis${axis.slice(1)}`;
}

// Smallest and largest finite value the traces put on an axis (the automatic end of a range
// set on one side only).
function dataExtent(figure, axis) {
  let low = Infinity;
  let high = -Infinity;
  for (const trace of figure.data || []) {
    if (axis !== 'x' && (trace.yaxis || 'y') !== axis) continue;
    for (const value of (axis === 'x' ? trace.x : trace.y) || []) {
      if (typeof value === 'number' && Number.isFinite(value)) {
        if (value < low) low = value;
        if (value > high) high = value;
      }
    }
  }
  return Number.isFinite(low) ? [low, high] : null;
}

// The figure with the limits of the Axes card: explicit ranges (log axes in decades, as Plotly
// wants them); an end left blank takes the extent of the data, both blank stay automatic.
function withLimits(figure, limits) {
  if (!limits || !Object.keys(limits).length) return figure;
  const layout = { ...figure.layout };
  for (const axis of figureAxes(figure)) {
    const pair = limits[axis];
    if (!pair || (pair[0] === null && pair[1] === null)) continue;
    const extent = dataExtent(figure, axis);
    const low = pair[0] ?? extent?.[0];
    const high = pair[1] ?? extent?.[1];
    if (low === undefined || high === undefined || !(low < high)) continue;
    const key = layoutKey(axis);
    const log = layout[key]?.type === 'log';
    if (log && low <= 0) continue;
    layout[key] = {
      ...layout[key],
      autorange: false,
      range: log ? [Math.log10(low), Math.log10(high)] : [low, high],
    };
  }
  return { ...figure, layout };
}

function shortNumber(value) {
  return Number(Number(value).toPrecision(6));
}

export class ResultsView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.session = app.session;
    this.folder = storage(`results-folder:${this.session.root}`) || null;
    this.datasets = [];
    this.current = null;
    this.optionsById = {};
    this.overlays = {}; // open dataset id -> [{id, slot}]: properties drawn on its x axis
    this.limits = storage(`limits:${this.session.root}`) || {}; // dataset id -> {x: [lo, hi], y, y2}
    this.overlaySeries = {}; // open dataset id -> series names of the overlay (components)
    this.compare = [];
    this.experiment = null;
    this.live = false;
    this.timer = null;
    this.signature = '';
    this.collapsed = new Set(storage('results-collapsed') || []);
    this.build();
    window.addEventListener('paoflow-theme', () => this.redraw());
    this.refreshFigure = debounce(() => this.showDataset(this.current, false), 150);
  }

  build() {
    this.side = h('div', { class: 'pane side' });
    this.center = h('div', { class: 'pane results-center' });
    this.right = h('div', { class: 'pane side' });
    this.root.append(
      h('div', { class: 'columns results-grid' }, this.side, this.center, this.right),
    );
    this.folderCard = h('div', { class: 'card' });
    this.search = h('input', {
      type: 'text',
      placeholder: 'filter datasets…',
      oninput: () => this.renderTree(),
    });
    this.tree = h('div', { class: 'tree' });
    this.side.append(this.folderCard, h('div', { class: 'card' }, this.search, this.tree));
    this.center.append(h('div', { class: 'empty' }, 'Open a folder with PAOFLOW outputs.'));
  }

  activate() {
    if (!this.loaded) {
      this.loaded = true;
      this.open(this.folder || this.defaultFolder || this.app.workdir || '.');
    }
  }

  setDefaultFolder(path) {
    this.defaultFolder = path;
  }

  async open(folder) {
    this.folder = folder;
    storage(`results-folder:${this.session.root}`, folder);
    this.loaded = true;
    await this.scan(true);
  }

  async scan(selectFirst = false) {
    let result;
    try {
      result = await api.post('/api/results/scan', { path: this.folder });
    } catch (error) {
      toast(error.message, 'bad');
      if (this.folder !== '.') return this.open('.');
      return;
    }
    const signature = result.datasets.map((d) => `${d.id}:${d.mtime}`).join('|');
    const changed = signature !== this.signature;
    const before = new Map(this.datasets.map((d) => [d.id, d.mtime]));
    this.signature = signature;
    this.folder = result.folder;
    this.absolute = result.absolute;
    this.categories = result.categories;
    this.fresh = new Set(
      result.datasets
        .filter((d) => before.size && (!before.has(d.id) || before.get(d.id) !== d.mtime))
        .map((d) => d.id),
    );
    this.datasets = result.datasets;
    this.renderFolder();
    if (changed) this.renderTree();
    const stillThere = this.current && this.datasets.some((d) => d.id === this.current);
    if (stillThere && this.fresh.has(this.current)) this.showDataset(this.current, false);
    else if (!stillThere && (selectFirst || this.current)) {
      const remembered = storage(`results-current:${this.session.root}:${this.folder}`);
      const first =
        this.datasets.find((d) => d.id === remembered) ||
        this.datasets.find((d) => d.family !== 'text' && d.family !== 'npz') ||
        this.datasets[0];
      if (first) this.showDataset(first.id);
      else this.renderEmpty();
    }
  }

  renderEmpty() {
    clear(this.center).append(
      h('div', { class: 'empty' }, `No PAOFLOW outputs found under ${this.folder}.`),
    );
    clear(this.right);
  }

  renderFolder() {
    clear(this.folderCard);
    const liveBox = h('input', {
      type: 'checkbox',
      checked: this.live,
      onchange: () => this.setLive(liveBox.checked),
    });
    this.folderCard.append(
      h('h3', {}, 'Results folder'),
      h(
        'div',
        { class: 'row' },
        h(
          'code',
          { class: 'grow ellipsis', title: this.absolute },
          this.folder === '.' ? '(root)' : this.folder,
        ),
        h(
          'button',
          {
            class: 'btn small',
            onclick: async () => {
              const chosen = await pickFolder(this.folder, {
                title: 'Folder with PAOFLOW outputs',
              });
              if (chosen !== null) this.open(chosen);
            },
          },
          'Change…',
        ),
      ),
      h(
        'div',
        { class: 'row', style: { marginTop: '8px' } },
        h('button', { class: 'btn small', onclick: () => this.scan() }, 'Rescan'),
        h(
          'label',
          { class: 'row small muted' },
          h('span', { class: 'switch' }, liveBox, h('span')),
          'live',
        ),
        h('span', { class: 'grow' }),
        h(
          'button',
          {
            class: 'btn small',
            title: 'Self-contained HTML report of every figure',
            onclick: () => this.report(),
          },
          'Report',
        ),
      ),
      h(
        'div',
        { class: 'small muted', style: { marginTop: '6px' } },
        `${this.datasets.length} datasets`,
      ),
    );
  }

  setLive(on) {
    this.live = on;
    clearInterval(this.timer);
    if (on) this.timer = setInterval(() => this.scan(), LIVE_INTERVAL);
  }

  renderTree() {
    clear(this.tree);
    const filter = this.search.value.trim().toLowerCase();
    const byCategory = new Map();
    for (const dataset of this.datasets) {
      const text = `${dataset.title} ${dataset.folder} ${dataset.files.join(' ')}`.toLowerCase();
      if (filter && !text.includes(filter)) continue;
      if (!byCategory.has(dataset.category)) byCategory.set(dataset.category, []);
      byCategory.get(dataset.category).push(dataset);
    }
    const order = [...(this.categories || []), ...byCategory.keys()];
    for (const category of [...new Set(order)]) {
      const items = byCategory.get(category);
      if (!items) continue;
      const open = !this.collapsed.has(category) || filter;
      const head = h(
        'button',
        {
          class: 'tree-head',
          onclick: () => {
            if (this.collapsed.has(category)) this.collapsed.delete(category);
            else this.collapsed.add(category);
            storage('results-collapsed', [...this.collapsed]);
            this.renderTree();
          },
        },
        h('span', {}, open ? '▾' : '▸'),
        h('span', { class: 'grow' }, category),
        h('span', { class: 'chip' }, String(items.length)),
      );
      this.tree.append(head);
      if (!open) continue;
      for (const dataset of items) {
        const where = dataset.folder === '.' ? '' : dataset.folder;
        this.tree.append(
          h(
            'button',
            {
              class: `tree-item${dataset.id === this.current ? ' selected' : ''}`,
              dataset: { id: dataset.id },
              title: dataset.files.join('\n'),
              onclick: () => this.showDataset(dataset.id),
            },
            h(
              'div',
              { class: 'row' },
              h('span', { class: 'grow ellipsis' }, dataset.title),
              this.fresh?.has(dataset.id) ? h('span', { class: 'chip ok' }, 'new') : null,
              dataset.files.length > 1
                ? h('span', { class: 'chip' }, `${dataset.files.length}`)
                : null,
            ),
            where ? h('div', { class: 'small faint ellipsis' }, where) : null,
          ),
        );
      }
    }
    if (!this.tree.children.length)
      this.tree.append(h('div', { class: 'empty' }, filter ? 'nothing matches' : 'no datasets'));
  }

  // ----------------------------------------------------------------------- one dataset
  async showDataset(id, resetView = true) {
    if (!id) return;
    this.current = id;
    storage(`results-current:${this.session.root}:${this.folder}`, id);
    for (const node of this.tree.querySelectorAll('.tree-item'))
      node.classList.toggle('selected', node.dataset.id === id);
    const options = this.optionsById[id] || {};
    let result;
    if (resetView)
      clear(this.center).append(
        h('div', { class: 'empty' }, h('span', { class: 'spinner' }), ' loading…'),
      );
    else this.center.style.opacity = '0.6';
    const overlay = this.overlays[id] || [];
    try {
      result = overlay.length
        ? await api.post('/api/results/overlay', {
            path: this.folder,
            id,
            options,
            items: overlay.map((item) => ({
              id: item.id,
              slot: item.slot,
              options: this.optionsById[item.id] || {},
            })),
            series: this.overlaySeries[id] || null,
          })
        : await api.post('/api/results/figure', {
            path: this.folder,
            id,
            options,
            experiment: this.experiment && this.experiment.id === id ? this.experiment : null,
          });
    } catch (error) {
      if (overlay.length) {
        // The overlay cannot be drawn (a dataset changed or went away): show the dataset alone.
        toast(error.message, 'bad', 7000);
        delete this.overlays[id];
        this.center.style.opacity = '';
        return this.showDataset(id, resetView);
      }
      this.center.style.opacity = '';
      clear(this.center).append(
        h('div', { style: { padding: '16px' } }, h('div', { class: 'callout bad' }, error.message)),
      );
      return;
    }
    this.center.style.opacity = '';
    this.result = result;
    this.renderCenter(result);
    this.renderRight(result);
  }

  renderCenter(result) {
    if (this.plot) purge(this.plot);
    clear(this.center);
    const ref = result.ref;
    this.center.append(
      h(
        'div',
        { class: 'result-head' },
        h(
          'div',
          { class: 'grow' },
          h('h2', {}, result.title),
          h(
            'div',
            { class: 'small muted mono ellipsis', title: ref.files.join('\n') },
            ref.files.length > 3
              ? `${ref.files.slice(0, 3).join(', ')} … (+${ref.files.length - 3})`
              : ref.files.join(', '),
          ),
        ),
        h('div', { class: 'small faint' }, `${formatBytes(ref.size)} · ${formatTime(ref.mtime)}`),
      ),
    );
    if (result.figure) {
      this.plot = h('div', { class: 'plot' });
      this.center.append(this.plot);
      const rendered = this.drawPlot(result).catch((error) => {
        clear(this.plot).append(h('div', { class: 'callout bad' }, error.message));
      });
      if (result.family === 'map') {
        this.cuts = h(
          'div',
          { class: 'cuts' },
          h(
            'div',
            { class: 'small muted' },
            'Click the map for the energy and momentum cuts (EDC and MDC) through that point.',
          ),
        );
        this.center.append(this.cuts);
        rendered.then(() => {
          if (this.plot.on) this.plot.on('plotly_click', (event) => this.showCuts(event, result));
        });
      }
      if (result.family !== 'volume' && result.family !== 'map') {
        const box = h('details', { class: 'table-view' }, h('summary', {}, 'Table view'));
        box.addEventListener('toggle', () => {
          if (box.open && box.children.length === 1) box.append(tableView(result.figure));
        });
        this.center.append(box);
      }
    } else if (result.text) {
      this.center.append(h('div', { class: 'text-view' }, codeView(result.text, ref.files[0])));
    }
    if (result.metrics) {
      const m = result.metrics;
      this.center.append(
        h(
          'div',
          { class: 'callout' },
          `Experiment vs ${m.compared_with}: ${m.points} points, RMS ${fmt(m.rms)}, bias ${fmt(m.bias)}, relative RMS ${fmt(m.relative_rms)}`,
        ),
      );
    }
    if (result.table?.length) {
      const table = h('table', { class: 'data-table' });
      for (const row of result.table)
        table.append(
          h(
            'tr',
            {},
            h('th', {}, row.name),
            h('td', {}, typeof row.value === 'number' ? fmt(row.value) : String(row.value)),
            h('td', { class: 'muted' }, row.unit || ''),
          ),
        );
      this.center.append(
        h('div', { class: 'card', style: { marginTop: '10px' } }, h('h3', {}, 'Values'), table),
      );
    }
    for (const note of result.notes || [])
      this.center.append(h('div', { class: 'small muted note' }, note));
  }

  // Energy- and momentum-distribution curves through the clicked point of a map.
  showCuts(event, result) {
    const point = (event.points || []).find((p) => p.data && p.data.type === 'heatmap');
    if (!point) return;
    const map = result.figure.data[0];
    const [row, column] = Array.isArray(point.pointIndex)
      ? point.pointIndex
      : [point.pointNumber[0], point.pointNumber[1]];
    const xLabel = result.figure.layout.xaxis.title ? result.figure.layout.xaxis.title.text : 'k';
    const yLabel = result.figure.layout.yaxis.title ? result.figure.layout.yaxis.title.text : 'E';
    const zLabel = map.colorbar?.title ? map.colorbar.title.text : 'intensity';
    const x = map.x[column];
    const y = map.y[row];
    const axis = (title) => ({
      title: { text: title },
      showline: true,
      linewidth: 1,
      ticks: 'outside',
      automargin: true,
      zeroline: false,
    });
    const cut = (name, xs, ys, xTitle, marker) => ({
      data: [
        {
          type: 'scatter',
          mode: 'lines',
          name,
          x: xs,
          y: ys,
          line: { color: 'series-1', width: 2 },
          hovertemplate: '%{x:.4g}, %{y:.4g}<extra></extra>',
        },
      ],
      layout: {
        xaxis: axis(xTitle),
        yaxis: axis(zLabel),
        margin: { l: 60, r: 12, t: 30, b: 50 },
        title: { text: name, x: 0, xanchor: 'left', font: { size: 13 } },
        shapes: [
          {
            type: 'line',
            yref: 'paper',
            y0: 0,
            y1: 1,
            xref: 'x',
            x0: marker,
            x1: marker,
            line: { color: 'series-2', width: 1, dash: 'dot' },
          },
        ],
        showlegend: false,
      },
    });
    const edc = cut(
      `EDC at ${xLabel.split(' (')[0]} = ${Number(x).toPrecision(4)}`,
      map.y,
      map.z.map((r) => r[column]),
      yLabel,
      y,
    );
    const mdc = cut(
      `MDC at ${yLabel.split(' (')[0]} = ${Number(y).toPrecision(4)}`,
      map.x,
      map.z[row],
      xLabel,
      x,
    );
    clear(this.cuts);
    const edcBox = h('div', { class: 'cut-plot' });
    const mdcBox = h('div', { class: 'cut-plot' });
    this.cuts.append(h('div', { class: 'cut-row' }, edcBox, mdcBox));
    renderFigure(edcBox, edc, 'edc', { keepTitle: true });
    renderFigure(mdcBox, mdc, 'mdc', { keepTitle: true });
    const colour = roleColor('series-2');
    const shapes = (this.plot.layout.shapes || []).filter((shape) => shape.name !== 'cut');
    shapes.push(
      {
        name: 'cut',
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        line: { color: colour, width: 1 },
      },
      {
        name: 'cut',
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0,
        x1: 1,
        y0: y,
        y1: y,
        line: { color: colour, width: 1 },
      },
    );
    window.Plotly.relayout(this.plot, { shapes });
  }

  // Theme change: the figure and the colour swatches of the cards take the new theme's colours.
  // The figure of the open dataset with its axis limits; a mouse zoom shows in the Axes card.
  async drawPlot(result) {
    const limits = this.limits[result.ref?.id];
    await renderFigure(
      this.plot,
      withLimits(result.figure, limits),
      result.title.replace(/\W+/g, '_'),
    );
    if (this.plot.removeAllListeners) this.plot.removeAllListeners('plotly_relayout');
    if (this.plot.on) this.plot.on('plotly_relayout', () => this.showRanges());
  }

  // Fill the Axes fields with the ranges on screen (after a zoom, a pan or a double click).
  showRanges() {
    const layout = this.plot?._fullLayout;
    if (!layout || !this.limitFields) return;
    for (const [axis, [low, high]] of Object.entries(this.limitFields)) {
      const scale = layout[layoutKey(axis)];
      if (!scale?.range) continue;
      const toData = (v) => (scale.type === 'log' ? 10 ** v : v);
      if (document.activeElement !== low) low.value = shortNumber(toData(scale.range[0]));
      if (document.activeElement !== high) high.value = shortNumber(toData(scale.range[1]));
    }
  }

  setLimits(id, axis, low, high) {
    const parse = (text) => (text.trim() === '' ? null : Number(text));
    const pair = [parse(low), parse(high)];
    if (pair.some((v) => v !== null && !Number.isFinite(v)))
      return toast('Limits are numbers (blank: automatic)', 'warn');
    if (pair[0] !== null && pair[1] !== null && pair[0] >= pair[1])
      return toast('The minimum must be below the maximum', 'warn');
    const limits = { ...(this.limits[id] || {}) };
    if (pair[0] === null && pair[1] === null) delete limits[axis];
    else limits[axis] = pair;
    if (Object.keys(limits).length) this.limits[id] = limits;
    else delete this.limits[id];
    storage(`limits:${this.session.root}`, this.limits);
    if (this.result?.ref?.id === id) this.drawPlot(this.result);
  }

  // x and y limits typed in, instead of the zoom of the figure; kept per dataset and used by the
  // PDF, PNG, SVG and Python exports.
  axesCard(result) {
    const axes = figureAxes(result.figure);
    if (!axes.length || result.family === 'volume') return null;
    const id = result.ref.id;
    const limits = this.limits[id] || {};
    const card = h('div', { class: 'card' }, h('h3', {}, 'Axes'));
    this.limitFields = {};
    for (const axis of axes) {
      const title = result.figure.layout[layoutKey(axis)]?.title?.text || axis;
      const field = (value) =>
        h('input', {
          type: 'text',
          inputmode: 'decimal',
          class: 'limit',
          placeholder: 'auto',
          value: value ?? '',
          'aria-label': `${axis} limit`,
        });
      const low = field(limits[axis]?.[0]);
      const high = field(limits[axis]?.[1]);
      const commit = () => this.setLimits(id, axis, low.value, high.value);
      for (const input of [low, high]) {
        input.addEventListener('change', commit);
        input.addEventListener('keydown', (event) => {
          if (event.key === 'Enter') commit();
        });
      }
      this.limitFields[axis] = [low, high];
      card.append(
        h(
          'div',
          { class: 'option' },
          h('div', { class: 'small muted ellipsis', title }, `${axis} · ${title}`),
          h('div', { class: 'row limits' }, low, h('span', { class: 'small muted' }, 'to'), high),
        ),
      );
    }
    card.append(
      h(
        'div',
        { class: 'row', style: { marginTop: '6px' } },
        h(
          'button',
          {
            class: 'btn small',
            onclick: () => {
              delete this.limits[id];
              storage(`limits:${this.session.root}`, this.limits);
              for (const [low, high] of Object.values(this.limitFields)) {
                low.value = '';
                high.value = '';
              }
              this.drawPlot(result);
            },
          },
          'Automatic',
        ),
      ),
      h(
        'div',
        { class: 'small muted', style: { marginTop: '4px' } },
        'Enter applies; blank is automatic. A zoom with the mouse shows its range here.',
      ),
    );
    return card;
  }

  redraw() {
    if (this.plot && this.result?.figure) this.drawPlot(this.result);
    if (this.result?.ref) this.renderRight(this.result);
  }

  setOption(name, value) {
    if (!this.optionsById[this.current]) this.optionsById[this.current] = {};
    this.optionsById[this.current][name] = value;
    this.refreshFigure();
  }

  renderRight(result) {
    clear(this.right);
    const options = Object.entries(result.options || {});
    if (options.length) {
      const card = h('div', { class: 'card' }, h('h3', {}, 'Options'));
      for (const [name, option] of options) card.append(this.optionControl(name, option));
      this.right.append(card);
    }
    const axesCard = this.axesCard(result);
    if (axesCard) this.right.append(axesCard);
    const overlayCard = this.overlayCard(result);
    if (overlayCard) this.right.append(overlayCard);
    const ref = result.ref;
    const exportCard = h('div', { class: 'card' }, h('h3', {}, 'Export'));
    const exports = h('div', { class: 'row wrap' });
    const exportAs = (format) => async () => {
      const limits = this.limits[ref.id] || {};
      const body = {
        path: this.folder,
        id: ref.id,
        options: this.optionsById[ref.id] || {},
        limits: { x: limits.x || null, y: limits.y || null }, // the Axes card (first panel)
        format,
      };
      try {
        if (format === 'csv' || format === 'python') {
          const out = await api.post('/api/results/export', body);
          if (format === 'csv') download(new Blob([out.text], { type: 'text/csv' }), out.filename);
          else
            dialog({
              title: 'Python',
              body: codeView(out.text, out.filename),
              actions: [
                { label: 'Copy', value: () => copyText(out.text) },
                { label: 'Close', value: null },
              ],
              wide: true,
            });
        } else {
          const blob = await postForBlob('/api/results/export', body);
          download(blob, `${result.title.replace(/\W+/g, '_')}.${format}`);
        }
      } catch (error) {
        toast(error.message, 'bad');
      }
    };
    exports.append(
      h('button', { class: 'btn small', onclick: exportAs('csv') }, 'CSV'),
      h('button', { class: 'btn small', onclick: exportAs('python') }, 'Python'),
    );
    if (result.figure && result.family !== 'volume') {
      exports.append(
        h('button', { class: 'btn small', title: 'matplotlib', onclick: exportAs('pdf') }, 'PDF'),
        h('button', { class: 'btn small', title: 'matplotlib', onclick: exportAs('png') }, 'PNG'),
        h('button', { class: 'btn small', title: 'matplotlib', onclick: exportAs('svg') }, 'SVG'),
      );
    }
    exportCard.append(
      exports,
      h(
        'div',
        { class: 'small muted', style: { marginTop: '6px' } },
        result.overlay
          ? `The buttons export ${result.ref.title} alone; the camera icon of the figure saves the overlay.`
          : 'The camera icon of the figure saves what is on screen.',
      ),
    );
    this.right.append(exportCard);
    this.right.append(this.compareCard(result));
    if (result.figure && !result.overlay && result.family !== 'volume' && result.family !== 'map')
      this.right.append(this.experimentCard(result));
  }

  // Other properties of the folder on the x axis of the open one (Re and Im ε, n and k, ...): one
  // figure, a panel per unit (never a second y axis), the colour per property, the line style per
  // series (component).
  overlayCard(result) {
    const base = this.datasets.find((d) => d.id === result.ref.id);
    if (!base?.overlayable) return null;
    const candidates = this.datasets.filter(
      (d) => d.id !== base.id && d.overlayable && d.x === base.x,
    );
    if (!candidates.length) return null;
    const chosen = this.overlays[base.id] || [];
    const card = h(
      'div',
      { class: 'card' },
      h('h3', {}, 'Overlay properties'),
      h(
        'div',
        { class: 'small muted', style: { marginBottom: '6px' } },
        `Other datasets against ${base.x}. One unit shares a panel; another unit gets its own panel below.`,
      ),
    );
    // The colour of a slot in the current theme, once the palette is loaded.
    const swatch = (slot) => {
      const dot = h('span', { class: 'dot-swatch' });
      loadPalette().then(() => {
        dot.style.background = roleColor(`series-${slot}`);
      });
      return dot;
    };
    const list = h(
      'div',
      { class: 'option-list overlay-list' },
      h('label', { class: 'row small muted' }, swatch(1), h('span', {}, `${base.title} (open)`)),
    );
    for (const candidate of candidates) {
      const entry = chosen.find((item) => item.id === candidate.id);
      const box = h('input', { type: 'checkbox', checked: Boolean(entry) });
      box.addEventListener('change', () => this.toggleOverlay(base.id, candidate.id, box));
      const where = candidate.folder && candidate.folder !== '.' ? ` · ${candidate.folder}` : '';
      list.append(
        h(
          'label',
          { class: 'row small' },
          box,
          entry ? swatch(entry.slot) : null,
          h(
            'span',
            { class: 'grow ellipsis', title: candidate.files.join(', ') },
            candidate.title + where,
          ),
        ),
      );
    }
    card.append(list);
    if (result.overlay && result.overlay.series.length > 1) {
      const shown = new Set(result.overlay.shown);
      const names = result.overlay.series;
      const boxes = names.map((name) => h('input', { type: 'checkbox', checked: shown.has(name) }));
      for (const box of boxes) {
        box.addEventListener('change', () => {
          const values = names.filter((_name, i) => boxes[i].checked);
          if (!values.length) {
            box.checked = true;
            toast('Keep at least one series', 'warn');
            return;
          }
          this.overlaySeries[base.id] = values;
          this.refreshFigure();
        });
      }
      card.append(
        h(
          'div',
          { class: 'option' },
          h('div', { class: 'small muted' }, 'Series (line style)'),
          h(
            'div',
            { class: 'option-list' },
            boxes.map((box, i) => h('label', { class: 'row small' }, box, names[i])),
          ),
        ),
      );
    }
    if (chosen.length) {
      card.append(
        h(
          'button',
          {
            class: 'btn small',
            style: { marginTop: '6px' },
            onclick: () => {
              delete this.overlays[base.id];
              this.refreshFigure();
            },
          },
          'Clear the overlay',
        ),
      );
    }
    return card;
  }

  // A colour slot stays with its property while others come and go (colour follows the entity).
  toggleOverlay(baseId, id, box) {
    const list = this.overlays[baseId] || [];
    if (box.checked) {
      if (list.length >= 3) {
        box.checked = false;
        toast('At most 3 properties on top of the open one', 'warn');
        return;
      }
      const used = new Set(list.map((item) => item.slot));
      let slot = 2;
      while (used.has(slot)) slot += 1;
      list.push({ id, slot });
    } else {
      const index = list.findIndex((item) => item.id === id);
      if (index >= 0) list.splice(index, 1);
    }
    this.overlays[baseId] = list;
    this.refreshFigure();
  }

  optionControl(name, option) {
    const label = h('div', { class: 'small muted' }, option.label);
    let control;
    if (option.range && Array.isArray(option.value)) {
      const low = h('input', { type: 'number', step: 'any', value: option.value[0] });
      const high = h('input', { type: 'number', step: 'any', value: option.value[1] });
      const apply = () => this.setOption(name, [Number(low.value), Number(high.value)]);
      low.addEventListener('change', apply);
      high.addEventListener('change', apply);
      control = h(
        'div',
        { class: 'row' },
        low,
        h('span', { class: 'muted' }, '–'),
        high,
        h(
          'button',
          {
            class: 'icon-btn',
            title: 'Full range',
            onclick: () => this.setOption(name, option.range),
          },
          '⤢',
        ),
      );
    } else if (option.range) {
      const input = h('input', {
        type: 'range',
        min: option.range[0],
        max: option.range[1],
        step: option.integer ? 1 : 'any',
        value: option.value,
      });
      const shown = h('span', { class: 'small mono' }, String(option.value));
      input.addEventListener('input', () => {
        shown.textContent = input.value;
      });
      input.addEventListener('change', () => this.setOption(name, Number(input.value)));
      control = h('div', { class: 'row' }, input, shown);
    } else if (option.number) {
      const input = h('input', { type: 'number', step: 'any', value: option.value });
      input.addEventListener('change', () => this.setOption(name, Number(input.value)));
      control = input;
    } else if (option.multiple) {
      const selected = new Set(option.value.map(String));
      const boxes = option.choices.map((choice) =>
        h('input', { type: 'checkbox', checked: selected.has(String(choice)) }),
      );
      for (const box of boxes) {
        box.addEventListener('change', () => {
          const values = option.choices.filter((_c, i) => boxes[i].checked);
          if (option.max && values.length > option.max) {
            box.checked = false;
            toast(`At most ${option.max} at once`, 'warn');
            return;
          }
          this.setOption(name, values);
        });
      }
      control = h(
        'div',
        { class: 'option-list' },
        boxes.map((box, i) => h('label', { class: 'row small' }, box, String(option.choices[i]))),
      );
    } else if (option.choices?.length && typeof option.choices[0] === 'boolean') {
      const box = h('input', {
        type: 'checkbox',
        checked: Boolean(option.value),
        onchange: () => this.setOption(name, box.checked),
      });
      control = h('label', { class: 'switch' }, box, h('span'));
    } else {
      const select = h(
        'select',
        {
          onchange: () =>
            this.setOption(
              name,
              Number.isNaN(Number(select.value)) || typeof option.value === 'string'
                ? select.value
                : Number(select.value),
            ),
        },
        option.choices.map((choice) =>
          h(
            'option',
            { value: String(choice), selected: String(choice) === String(option.value) },
            String(choice),
          ),
        ),
      );
      control = select;
    }
    return h('div', { class: 'option' }, label, control);
  }

  compareCard(result) {
    const card = h('div', { class: 'card' }, h('h3', {}, 'Compare runs'));
    const list = h('div', { class: 'stack' });
    const refresh = () => {
      clear(list);
      for (const [index, item] of this.compare.entries()) {
        list.append(
          h(
            'div',
            { class: 'row small' },
            h('span', { class: 'dot-swatch', dataset: { slot: String(index + 1) } }),
            h(
              'span',
              { class: 'grow ellipsis', title: `${item.path} · ${item.title}` },
              `${item.label}`,
            ),
            h(
              'button',
              {
                class: 'icon-btn',
                onclick: () => {
                  this.compare.splice(index, 1);
                  refresh();
                },
              },
              '✕',
            ),
          ),
        );
      }
      if (!this.compare.length)
        list.append(
          h(
            'div',
            { class: 'small muted' },
            'Add the same kind of dataset from other folders (SOC vs no SOC, basis presets, kernel levels).',
          ),
        );
    };
    refresh();
    card.append(
      list,
      h(
        'div',
        { class: 'row', style: { marginTop: '8px' } },
        h(
          'button',
          {
            class: 'btn small',
            onclick: () => {
              if (this.compare.length && this.compare[0].kind !== result.ref.kind)
                return toast('Compare datasets of the same kind', 'warn');
              if (this.compare.some((c) => c.id === result.ref.id && c.path === this.folder))
                return toast('Already in the list', 'warn');
              this.compare.push({
                path: this.folder,
                id: result.ref.id,
                kind: result.ref.kind,
                title: result.title,
                label: this.folder === '.' ? 'root' : this.folder,
                options: this.optionsById[result.ref.id] || {},
              });
              refresh();
            },
          },
          '+ this dataset',
        ),
        h(
          'button',
          { class: 'btn small primary', disabled: false, onclick: () => this.showComparison() },
          'Overlay',
        ),
      ),
    );
    return card;
  }

  async showComparison() {
    if (this.compare.length < 2) return toast('Add at least two datasets', 'warn');
    try {
      const result = await api.post('/api/results/compare', { items: this.compare });
      if (this.plot) purge(this.plot);
      clear(this.center).append(
        h(
          'div',
          { class: 'result-head' },
          h('h2', { class: 'grow' }, result.title),
          h(
            'button',
            { class: 'btn small', onclick: () => this.showDataset(this.current) },
            'Back',
          ),
        ),
      );
      this.plot = h('div', { class: 'plot' });
      this.center.append(this.plot);
      this.result = { figure: result.figure };
      await renderFigure(this.plot, result.figure, 'comparison');
      for (const note of result.notes)
        this.center.append(h('div', { class: 'small muted note' }, note));
    } catch (error) {
      toast(error.message, 'bad');
    }
  }

  experimentCard(result) {
    const card = h('div', { class: 'card' }, h('h3', {}, 'Experiment overlay'));
    const active = this.experiment && this.experiment.id === result.ref.id ? this.experiment : null;
    const file = h('code', { class: 'grow ellipsis small' }, active ? active.file : 'no file');
    const xCol = h('input', {
      type: 'number',
      min: 0,
      value: active ? active.x_column : 0,
      style: { width: '56px' },
    });
    const yCol = h('input', {
      type: 'number',
      min: 0,
      value: active ? active.y_column : 1,
      style: { width: '56px' },
    });
    const scale = h('input', {
      type: 'number',
      step: 'any',
      value: active ? active.scale : 1,
      style: { width: '72px' },
    });
    let chosen = active ? active.file : null;
    card.append(
      h(
        'div',
        { class: 'row' },
        file,
        h(
          'button',
          {
            class: 'btn small',
            onclick: async () => {
              const picked = await pickFolder(this.folder, {
                title: 'Experimental data (columns, CSV or dataset file)',
                allowFiles: true,
              });
              if (picked) {
                chosen = picked;
                file.textContent = picked;
              }
            },
          },
          'File…',
        ),
      ),
      h(
        'div',
        { class: 'row small', style: { marginTop: '6px' } },
        'x col',
        xCol,
        'y col',
        yCol,
        '×',
        scale,
      ),
      h(
        'div',
        { class: 'row', style: { marginTop: '8px' } },
        h(
          'button',
          {
            class: 'btn small primary',
            onclick: () => {
              if (!chosen) return toast('Choose a file', 'warn');
              this.experiment = {
                id: result.ref.id,
                file: chosen,
                x_column: Number(xCol.value),
                y_column: Number(yCol.value),
                scale: Number(scale.value) || 1,
              };
              this.showDataset(result.ref.id, false);
            },
          },
          'Overlay',
        ),
        active
          ? h(
              'button',
              {
                class: 'btn small',
                onclick: () => {
                  this.experiment = null;
                  this.showDataset(result.ref.id, false);
                },
              },
              'Remove',
            )
          : null,
      ),
    );
    return card;
  }

  launchReport() {
    return this.report();
  }

  async report() {
    toast('Building the report…', 'info', 2500);
    try {
      const blob = await postForBlob('/api/results/report', { path: this.folder });
      download(
        blob,
        `${this.folder === '.' ? 'results' : this.folder.split('/').pop()}_report.html`,
      );
    } catch (error) {
      toast(error.message, 'bad');
    }
  }
}

function fmt(value) {
  if (value === null || value === undefined) return '–';
  if (typeof value !== 'number') return String(value);
  const abs = Math.abs(value);
  return abs !== 0 && (abs < 1e-3 || abs >= 1e5)
    ? value.toExponential(3)
    : Number(value.toPrecision(5)).toString();
}
