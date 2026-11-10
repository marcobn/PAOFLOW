// PAOFLOW Studio: session, tabs, theme, and the links between Setup, Runs and Results.

import { api } from './api.js';
import { runFolderOf } from './browser.js';
import { h, storage, toast } from './dom.js';
import { HelpPanel, TAB_HELP } from './help.js';
import { SetupView } from './setup.js';

const TABS = ['setup', 'runs', 'results'];

class App {
  constructor(session) {
    this.session = session;
    this.workdir = session.start || '.';
    this.views = {};
    this.runs = null;
    this.results = null;
  }

  async init() {
    document.getElementById('version').textContent = this.session.version;
    const root = document.getElementById('root');
    root.textContent = this.session.root;
    root.title = `GUI root: ${this.session.root}`;
    document.getElementById('readonly').hidden = !this.session.read_only;
    this.checkServerCode(this.session);
    // Back to the window (after a pw.x run in a terminal, say): the folder may have changed.
    window.addEventListener('focus', () => {
      if (this.currentTab() === 'setup') this.setup?.refreshFolder();
    });
    document.addEventListener('visibilitychange', async () => {
      if (document.visibilityState !== 'visible') return;
      try {
        this.checkServerCode(await api.get('/api/session'));
      } catch (_error) {
        // the server stopped: the next request reports it
      }
    });
    this.initTheme();
    this.help = new HelpPanel();
    document.getElementById('help').addEventListener('click', () => this.help.toggle());
    for (const button of document.querySelectorAll('.tab')) {
      button.addEventListener('click', () => this.show(button.dataset.tab));
    }
    window.addEventListener('hashchange', () => this.show(this.currentTab(), false));
    this.setup = new SetupView(document.getElementById('view-setup'), this);
    await this.loadOptionalViews();
    await this.setup.init();
    // `paoflow-studio plot DIR` opens the Results tab on DIR, whatever the browser remembered.
    if (
      this.currentTab() === 'results' &&
      this.results &&
      this.session.start &&
      this.session.start !== '.'
    ) {
      await this.results.open(this.session.start);
    }
    this.show(this.currentTab(), false);
    if (!storage('guide-seen')) {
      // First visit: open the guide on its first page.
      storage('guide-seen', true);
      this.help.open('start');
    }
  }

  // A source checkout serves the front end from disk, but the server keeps the Python it started
  // with: after an update of PAOFLOW the page may call endpoints the old server lacks.
  checkServerCode(session) {
    if (!session.restart_needed || document.getElementById('restart-banner')) return;
    const started = new Date(session.started * 1000).toLocaleTimeString();
    const banner = h(
      'div',
      { id: 'restart-banner', class: 'callout warn restart-banner', role: 'alert' },
      h(
        'span',
        { class: 'grow' },
        `PAOFLOW was updated after this server started (${started}). Stop paoflow-studio ` +
          '(Ctrl-C in its terminal) and start it again; until then new features may not work.',
      ),
      h('button', { class: 'icon-btn', title: 'Hide', onclick: () => banner.remove() }, '×'),
    );
    document.querySelector('main').before(banner);
  }

  // The guide on the section of the current tab.
  helpForTab() {
    this.help.open(TAB_HELP[this.currentTab()]);
  }

  async loadOptionalViews() {
    try {
      const { RunsView } = await import('./runs.js');
      this.runs = new RunsView(document.getElementById('view-runs'), this);
    } catch (_error) {
      this.placeholder('runs', 'Runs', 'The run manager is not available in this version.');
    }
    try {
      const { ResultsView } = await import('./results.js');
      this.results = new ResultsView(document.getElementById('view-results'), this);
    } catch (_error) {
      this.placeholder(
        'results',
        'Results',
        'The results viewer is not available in this version.',
      );
    }
  }

  placeholder(name, title, text) {
    const view = document.getElementById(`view-${name}`);
    view.append(
      h('div', { class: 'empty', style: { margin: 'auto' } }, h('h2', {}, title), h('p', {}, text)),
    );
  }

  currentTab() {
    const name = (location.hash || '#setup').slice(1).split('/')[0];
    return TABS.includes(name) ? name : 'setup';
  }

  show(name, updateHash = true) {
    for (const button of document.querySelectorAll('.tab')) {
      button.setAttribute('aria-selected', String(button.dataset.tab === name));
    }
    for (const tab of TABS)
      document.getElementById(`view-${tab}`).classList.toggle('active', tab === name);
    if (updateHash && location.hash.slice(1).split('/')[0] !== name)
      history.replaceState(null, '', `#${name}`);
    const view = { setup: this.setup, runs: this.runs, results: this.results }[name];
    if (view?.activate) view.activate();
  }

  initTheme() {
    const saved = storage('theme');
    const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
    const apply = (theme) => {
      document.documentElement.dataset.theme = theme;
      window.dispatchEvent(new CustomEvent('paoflow-theme', { detail: theme }));
    };
    apply(saved || (prefersDark ? 'dark' : 'light'));
    document.getElementById('theme').addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      storage('theme', next);
      apply(next);
    });
  }

  // ---------------------------------------------------------------- links between tabs
  setWorkdir(path) {
    this.workdir = path;
    if (this.results?.setDefaultFolder) this.results.setDefaultFolder(path);
  }

  // One folder for every tab: a folder chosen in one tab (Change… in Setup or Results, Run a
  // pw.x input… in Runs) becomes the folder of the others. Setup and Runs work in the run folder,
  // so a Results folder named output/ gives them its parent; a folder they may not write (opened
  // for reading) stays in Results only.
  async changeFolder(path, source) {
    let working = path;
    if (source === 'results') {
      const parent = runFolderOf(path);
      if (parent !== this.setup.workdir) {
        const opened =
          this.results.writable &&
          (await this.setup.openFolder(parent, { propagate: false, fallback: false }));
        if (!opened) {
          toast(
            `Setup and Runs stay in ${this.setup.workdir}: ${path} is read-only. ` +
              'Choose it with Change… in Setup to write there.',
            'warn',
            7000,
          );
          return;
        }
      }
      working = this.setup.workdir;
    } else if (source === 'runs' && path !== this.setup.workdir) {
      await this.setup.openFolder(path, { propagate: false, fallback: false });
    }
    if (source !== 'runs') this.runs?.showFolder?.(working);
    if (source !== 'results') this.results?.follow?.(working);
  }

  notifyWritten(result) {
    if (this.runs?.setPlan) this.runs.setPlan(result.folder, result.plan);
  }

  showRuns(folder, plan, workflow) {
    if (!this.runs) return toast('The run manager is not available', 'warn');
    this.runs.setPlan(folder, plan, workflow);
    this.show('runs');
  }

  // A plan step done in the GUI: 'setup:<workflow>' opens that workflow on the folder.
  async doAction(action, folder) {
    const [tab, workflow] = action.split(':');
    if (tab !== 'setup') return;
    this.show('setup');
    await this.setup.openFolder(folder || this.workdir);
    if (workflow) this.setup.selectWorkflow(workflow);
  }

  runCommand(step) {
    if (!this.runs) return toast('The run manager is not available', 'warn');
    this.runs.launch({ ...step, rootRelative: true });
    this.show('runs');
  }

  showResults(folder) {
    if (!this.results) return toast('The results viewer is not available', 'warn');
    this.results.open(folder);
    this.show('results');
  }
}

async function start() {
  let session;
  try {
    session = await api.get('/api/session');
  } catch (error) {
    document.body.append(
      h('div', { class: 'empty' }, `Cannot reach the PAOFLOW Studio server: ${error.message}`),
    );
    return;
  }
  const app = new App(session);
  window.paoflowApp = app;
  try {
    await app.init();
  } catch (error) {
    console.error(error);
    toast(`Start-up error: ${error.message}`, 'bad', 0);
  }
}

start();
