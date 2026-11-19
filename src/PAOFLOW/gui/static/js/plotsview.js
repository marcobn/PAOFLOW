// Plots tab: the plot scripts of the working folder (plot*.py), run without a display, and the
// figures they make (PNG shown here, PDF to download).  The server runs a script through the job
// manager with `capture` (python -m PAOFLOW.graphics.figure_capture): every figure it shows or
// saves goes to figures/<script>/ next to it, listed in figures.json.

import { api } from './api.js';
import { clear, dialog, formatTime, h, storage, toast } from './dom.js';

function fileUrl(path, mtime, download = false) {
  const query = new URLSearchParams({ path, v: String(mtime || 0) });
  if (download) query.set('download', '1');
  return `/api/plots/file?${query}`;
}

function splitArguments(text) {
  return text.trim() ? text.trim().split(/\s+/) : [];
}

export class PlotsView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.session = app.session;
    this.folder = null; // the working folder (App.changeFolder), read when the tab is shown
    this.data = null;
    this.signature = '';
    this.selected = null;
    this.args = {}; // script -> arguments typed in the field
    this.log = { id: null, offset: 0, text: '' };
    this.build();
  }

  build() {
    this.side = h('div', { class: 'pane side' });
    this.center = h('div', { class: 'pane plots-center' });
    this.root.append(h('div', { class: 'columns plots-grid' }, this.side, this.center));
    this.folderCard = h('div', { class: 'card' });
    this.listCard = h('div', { class: 'card' });
    this.side.append(this.folderCard, this.listCard);
    this.center.append(h('div', { class: 'empty' }, 'Plot scripts of the working folder.'));
  }

  shown() {
    return (
      document.getElementById('view-plots').classList.contains('active') &&
      document.visibilityState === 'visible'
    );
  }

  // The tab is shown: the working folder, then a refresh every 2 s while it stays in view (the
  // figures of a running script appear one by one).
  activate() {
    const folder = this.folder || this.app.workdir || '.';
    if (folder !== this.data?.folder) this.open(folder);
    else this.refresh();
    clearInterval(this.timer);
    this.timer = setInterval(() => {
      if (this.shown()) this.refresh();
    }, 2000);
  }

  // The working folder chosen in another tab.
  follow(folder) {
    if (folder === this.folder) return;
    this.folder = folder;
    if (this.shown()) this.open(folder);
  }

  async open(folder, script = null) {
    this.folder = folder;
    this.selected = script || storage(`plots-selected:${this.session.root}:${folder}`);
    this.data = null;
    this.signature = '';
    clear(this.center).append(
      h('div', { class: 'empty' }, h('span', { class: 'spinner' }), ' loading…'),
    );
    await this.refresh();
  }

  // Runs: "Run in Plots" on a plot step.
  async runScript(folder, script) {
    await this.open(folder, script);
    const info = this.script(script);
    if (!info) return toast(`${script} is not in ${folder}`, 'bad');
    if (info.job?.status !== 'running') await this.run(info);
  }

  script(name) {
    return this.data?.scripts.find((s) => s.script === name) || null;
  }

  // One request at a time: a refresh asked for during another (the tab shown, then a run from
  // the Runs tab) runs after it, so whoever awaits it sees the current folder.
  async refresh() {
    while (this.pending) await this.pending;
    if (!this.folder) return;
    this.pending = this.load();
    try {
      await this.pending;
    } finally {
      this.pending = null;
    }
  }

  async load() {
    try {
      const data = await api.get('/api/plots', { path: this.folder });
      // What changes the page: the scripts, their figures and the state of their jobs (not the
      // elapsed time of a running job).
      const signature = JSON.stringify([
        data.folder,
        data.writable,
        data.scripts.map((s) => [s.script, s.capture, s.job?.id, s.job?.status]),
      ]);
      this.data = data;
      this.folder = data.folder;
      if (!this.script(this.selected)) {
        const withFigures = data.scripts.find((s) => s.capture?.figures.length);
        this.selected = (withFigures || data.scripts[0])?.script || null;
      }
      if (signature !== this.signature) {
        this.signature = signature;
        this.renderFolder();
        this.renderList();
        this.renderScript();
      }
      await this.followLog();
    } catch (error) {
      toast(error.message, 'bad');
    }
  }

  renderFolder() {
    clear(this.folderCard).append(
      h('h3', {}, 'Plot scripts'),
      h(
        'div',
        { class: 'row' },
        h(
          'code',
          { class: 'grow ellipsis', title: this.data.absolute },
          this.data.folder === '.' ? '(root)' : this.data.folder,
        ),
        h('button', { class: 'btn small', onclick: () => this.refresh() }, 'Rescan'),
      ),
      h(
        'div',
        { class: 'small muted', style: { marginTop: '6px' } },
        'The working folder of Setup (Change… there, or in Runs and Results).',
      ),
    );
  }

  renderList() {
    clear(this.listCard);
    if (!this.data.scripts.length) {
      this.listCard.append(
        h(
          'div',
          { class: 'small muted' },
          'No plot*.py here. Setup writes them with the files of a workflow when ',
          h('em', {}, 'Write the plot scripts'),
          ' is on (Output box).',
        ),
      );
      return;
    }
    const list = h('div', { class: 'tree' });
    for (const info of this.data.scripts) {
      const count = info.capture?.figures.length || 0;
      const running = info.job?.status === 'running';
      list.append(
        h(
          'button',
          {
            class: `tree-item${info.script === this.selected ? ' selected' : ''}`,
            title: info.title || info.script,
            onclick: () => this.select(info.script),
          },
          h(
            'div',
            { class: 'row' },
            h('code', { class: 'grow ellipsis' }, info.script),
            running ? h('span', { class: 'chip accent' }, 'running') : null,
            !running && info.job?.status === 'failed'
              ? h('span', { class: 'chip bad' }, 'failed')
              : null,
            count ? h('span', { class: 'chip' }, String(count)) : null,
          ),
          info.title ? h('div', { class: 'small muted ellipsis' }, info.title) : null,
        ),
      );
    }
    this.listCard.append(list);
  }

  select(script) {
    this.selected = script;
    storage(`plots-selected:${this.session.root}:${this.folder}`, script);
    for (const node of this.listCard.querySelectorAll('.tree-item'))
      node.classList.toggle('selected', node.querySelector('code')?.textContent === script);
    this.log = { id: null, offset: 0, text: '' };
    this.renderScript();
    this.followLog();
  }

  renderScript() {
    const info = this.script(this.selected);
    clear(this.center);
    if (!info) {
      this.center.append(
        h(
          'div',
          { class: 'empty' },
          this.data.scripts.length ? 'Choose a script.' : 'No plot scripts in this folder.',
        ),
      );
      return;
    }
    const job = info.job;
    const running = job?.status === 'running';
    const stored = storage(`plot-args:${this.session.root}:${this.folder}:${info.script}`);
    const value = this.args[info.script] ?? stored ?? info.arguments.join(' ');
    const argsField = h('input', {
      type: 'text',
      class: 'mono grow',
      value,
      placeholder: 'arguments',
      title: info.options.length ? `Options of ${info.script}: ${info.options.join(' ')}` : '',
      oninput: () => {
        this.args[info.script] = argsField.value;
        storage(`plot-args:${this.session.root}:${this.folder}:${info.script}`, argsField.value);
      },
      onkeydown: (event) => {
        if (event.key === 'Enter' && !running) this.run(info);
      },
    });
    const canRun = this.data.writable && !this.session.read_only;
    const capture = info.capture;
    this.center.append(
      h(
        'div',
        { class: 'plots-head' },
        h(
          'div',
          { class: 'row' },
          h('h3', { class: 'grow' }, info.title || info.script),
          running
            ? h('button', { class: 'btn small', onclick: () => this.cancel(job) }, 'Cancel')
            : null,
          h(
            'button',
            {
              class: 'btn small primary',
              disabled: !canRun || running,
              title: canRun ? '' : 'This folder is read-only',
              onclick: () => this.run(info),
            },
            running ? 'Running…' : capture ? 'Run again' : 'Run',
          ),
        ),
        h(
          'div',
          { class: 'row', style: { marginTop: '6px' } },
          h('code', {}, `python ${info.script}`),
          argsField,
        ),
        h(
          'div',
          { class: 'small muted', style: { marginTop: '4px' } },
          info.options.length ? `options: ${info.options.join(' ')}. ` : '',
          'Runs without a display; prompts take their defaults. Figures: ',
          h('code', {}, capture?.folder || `figures/${info.script.replace(/\.py$/, '')}`),
          ' (PNG and PDF).',
        ),
      ),
    );
    this.center.append(this.gallery(info));
    this.center.append(this.logBox(info));
  }

  gallery(info) {
    const capture = info.capture;
    const running = info.job?.status === 'running';
    if (!capture)
      return h(
        'div',
        { class: 'empty' },
        running ? 'Running… the figures appear here.' : 'Not run yet: Run makes its figures.',
      );
    if (!capture.figures.length)
      return h(
        'div',
        { class: 'empty' },
        running
          ? 'Running… the figures appear here.'
          : `${info.script} made no figure${capture.status ? ` (exit status ${capture.status})` : ''}: see the log below.`,
      );
    const grid = h('div', { class: 'figure-grid' });
    for (const figure of capture.figures) {
      const png = figure.files.png;
      const pdf = figure.files.pdf;
      grid.append(
        h(
          'figure',
          { class: 'figure-card' },
          png
            ? h('img', {
                src: fileUrl(png, figure.mtime),
                alt: figure.title,
                title: 'Click to enlarge',
                onclick: () => this.enlarge(figure),
              })
            : null,
          h(
            'figcaption',
            { class: 'row' },
            h('span', { class: 'grow ellipsis', title: figure.title }, figure.title),
            figure.saved_as
              ? h(
                  'span',
                  { class: 'chip', title: `the script saved it as ${figure.saved_as}` },
                  'saved',
                )
              : null,
            png
              ? h('a', { class: 'btn small', href: fileUrl(png, figure.mtime, true) }, 'PNG')
              : null,
            pdf
              ? h('a', { class: 'btn small', href: fileUrl(pdf, figure.mtime, true) }, 'PDF')
              : null,
          ),
        ),
      );
    }
    const when = capture.ended ? `made ${formatTime(capture.ended)}` : 'running';
    return h(
      'div',
      {},
      h(
        'div',
        { class: 'small muted', style: { margin: '10px 0 6px' } },
        `${capture.figures.length} figure${capture.figures.length === 1 ? '' : 's'}, ${when}`,
        capture.arguments?.length ? ` with ${capture.arguments.join(' ')}` : '',
      ),
      grid,
    );
  }

  enlarge(figure) {
    dialog({
      title: figure.title,
      body: h('img', {
        class: 'figure-large',
        src: fileUrl(figure.files.png, figure.mtime),
        alt: figure.title,
      }),
      wide: true,
    });
  }

  logBox(info) {
    const job = info.job;
    if (!job) return h('div');
    this.logBody = h(
      'pre',
      { class: 'log plots-log' },
      this.log.id === job.id ? this.log.text : '',
    );
    const failed = job.status === 'failed' || (info.capture?.status ?? 0) !== 0;
    const details = h(
      'details',
      { class: 'plots-log-box', open: failed || this.logOpen },
      h(
        'summary',
        { class: 'plots-log-title' },
        `Log of the last run (${job.status}${job.returncode ? `, exit ${job.returncode}` : ''})`,
      ),
      this.logBody,
    );
    details.addEventListener('toggle', () => {
      this.logOpen = details.open;
      if (details.open) this.followLog();
    });
    this.logDetails = details;
    return details;
  }

  // The log of the selected script's last job, read from where it was left.
  async followLog() {
    const job = this.script(this.selected)?.job;
    if (!job || !this.logDetails?.open) return;
    if (this.log.id !== job.id) this.log = { id: job.id, offset: 0, text: '' };
    if (this.log.done) return;
    try {
      const chunk = await api.get('/api/jobs/log', { id: job.id, offset: this.log.offset });
      this.log.text += chunk.text;
      this.log.offset = chunk.offset;
      this.log.done = chunk.status !== 'running' && !chunk.text;
      if (this.logBody) {
        this.logBody.textContent = this.log.text;
        this.logBody.scrollTop = this.logBody.scrollHeight;
      }
    } catch (_error) {
      // the log of a removed job: nothing to show
    }
  }

  async run(info) {
    const text = this.args[info.script] ?? argsFieldValue(this.center) ?? '';
    const step = {
      label: `Figures (${info.script})`,
      argv: ['python', info.script, ...splitArguments(text)],
      cwd: this.folder,
      capture: true,
    };
    try {
      await api.post('/api/jobs/launch', { step, ranks: 1 });
      this.log = { id: null, offset: 0, text: '' };
      toast(`Running ${info.script}`, 'ok');
      await this.refresh();
    } catch (error) {
      toast(error.message, 'bad', 7000);
    }
  }

  async cancel(job) {
    try {
      await api.post('/api/jobs/cancel', { id: job.id });
      toast('Cancel requested', 'warn');
    } catch (error) {
      toast(error.message, 'bad');
    }
  }
}

// The arguments shown in the field (its default when nothing was typed).
function argsFieldValue(center) {
  return center.querySelector('.plots-head input[type="text"]')?.value ?? null;
}
