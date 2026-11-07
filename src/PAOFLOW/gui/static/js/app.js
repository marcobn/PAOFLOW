// PAOFLOW Studio: session, tabs, theme, and the links between Setup, Runs and Results.

import { api } from './api.js';
import { h, storage, toast } from './dom.js';
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
    this.initTheme();
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
    const view = name === 'runs' ? this.runs : name === 'results' ? this.results : null;
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

  notifyWritten(result) {
    if (this.runs?.setPlan) this.runs.setPlan(result.folder, result.plan);
  }

  showRuns(folder, plan, workflow) {
    if (!this.runs) return toast('The run manager is not available', 'warn');
    this.runs.setPlan(folder, plan, workflow);
    this.show('runs');
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
