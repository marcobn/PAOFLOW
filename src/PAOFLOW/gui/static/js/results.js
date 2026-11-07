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
import { purge, renderFigure, roleColor, tableView } from './plots.js';

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

export class ResultsView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.session = app.session;
    this.folder = storage(`results-folder:${this.session.root}`) || null;
    this.datasets = [];
    this.current = null;
    this.optionsById = {};
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
    try {
      result = await api.post('/api/results/figure', {
        path: this.folder,
        id,
        options,
        experiment: this.experiment && this.experiment.id === id ? this.experiment : null,
      });
    } catch (error) {
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
      const rendered = renderFigure(
        this.plot,
        result.figure,
        result.title.replace(/\W+/g, '_'),
      ).catch((error) => {
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

  redraw() {
    if (this.plot && this.result?.figure) renderFigure(this.plot, this.result.figure);
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
    const ref = result.ref;
    const exportCard = h('div', { class: 'card' }, h('h3', {}, 'Export'));
    const exports = h('div', { class: 'row wrap' });
    const exportAs = (format) => async () => {
      const body = {
        path: this.folder,
        id: ref.id,
        options: this.optionsById[ref.id] || {},
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
        'The camera icon of the figure saves what is on screen.',
      ),
    );
    this.right.append(exportCard);
    this.right.append(this.compareCard(result));
    if (result.figure && result.family !== 'volume' && result.family !== 'map')
      this.right.append(this.experimentCard(result));
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
      const boxes = option.choices.map((choice) => {
        const box = h('input', { type: 'checkbox', checked: selected.has(String(choice)) });
        box.addEventListener('change', () => {
          const values = option.choices.filter((_c, i) => boxes[i].checked);
          if (option.max && values.length > option.max) {
            box.checked = false;
            toast(`At most ${option.max} at once`, 'warn');
            return;
          }
          this.setOption(name, values);
        });
        return h('label', { class: 'row small' }, box, String(choice));
      });
      control = h('div', { class: 'option-list' }, boxes);
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
