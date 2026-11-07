// Runs tab: the run plan of the last generated workflow, local jobs and their logs.

import { api } from './api.js';
import { clear, copyText, formatTime, h, storage, toast } from './dom.js';

const STATUS_CHIP = { running: 'accent', finished: 'ok', failed: 'bad', cancelled: 'warn', lost: 'warn' };

function joinPath(folder, relative) {
  if (!relative || relative === '.') return folder || '.';
  if (!folder || folder === '.') return relative;
  return `${folder}/${relative}`;
}

function duration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}

export class RunsView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.session = app.session;
    const saved = storage(`plan:${this.session.root}`) || {};
    this.folder = saved.folder || null;
    this.plan = saved.plan || [];
    this.workflow = saved.workflow || '';
    this.jobs = [];
    this.selected = null;
    this.offset = 0;
    this.follow = true;
    this.maxRanks = this.session.max_ranks || 1;
    this.build();
  }

  build() {
    this.side = h('div', { class: 'pane side' });
    this.center = h('div', { class: 'pane runs-center' });
    this.root.append(h('div', { class: 'columns runs-grid' }, this.side, this.center));
    this.planCard = h('div', { class: 'card' });
    this.jobsCard = h('div', { class: 'card' });
    this.side.append(this.planCard, this.jobsCard);
    this.renderPlan();
    this.renderLogShell();
  }

  activate() {
    this.refresh();
    clearInterval(this.timer);
    this.timer = setInterval(() => this.poll(), 1500);
  }

  setPlan(folder, plan, workflow) {
    this.folder = folder;
    this.plan = plan || [];
    this.workflow = workflow || this.workflow;
    storage(`plan:${this.session.root}`, { folder, plan: this.plan, workflow: this.workflow });
    this.renderPlan();
  }

  renderPlan() {
    clear(this.planCard);
    this.planCard.append(h('h3', {}, 'Run plan'));
    if (!this.plan.length) {
      this.planCard.append(h('div', { class: 'small muted' }, 'Write the files of a workflow in the Setup tab; its steps appear here.'));
      return;
    }
    this.planCard.append(h('div', { class: 'small muted', style: { marginBottom: '6px' } }, `${this.workflow || 'workflow'} in `, h('code', {}, this.folder)));
    const list = h('ol', { class: 'steps' });
    for (const step of this.plan) {
      const ranks = h('input', { type: 'number', min: 1, max: this.maxRanks, value: Math.min(4, this.maxRanks), style: { width: '64px' }, title: 'MPI ranks' });
      const usesRanks = step.mpi || step.ranks_env;
      const runnable = !step.interactive && !step.manual && !this.session.read_only;
      list.append(
        h(
          'li',
          {},
          h('div', { class: 'row' }, h('strong', { class: 'grow' }, step.label), usesRanks ? h('span', { class: 'chip accent' }, step.mpi ? 'MPI' : step.ranks_env) : null),
          step.manual ? null : h('div', { class: 'cmd' }, h('code', {}, step.command), h('button', { class: 'icon-btn', title: 'Copy', onclick: () => copyText(step.command) }, '⧉')),
          step.note && !step.interactive ? h('div', { class: 'small muted' }, step.note) : null,
          runnable
            ? h('div', { class: 'row', style: { marginTop: '4px' } }, usesRanks ? [h('span', { class: 'small muted' }, 'ranks'), ranks] : null, h('span', { class: 'grow' }), h('button', { class: 'btn small primary', onclick: () => this.launch(step, usesRanks ? Number(ranks.value) : 1) }, 'Run'))
            : step.interactive
              ? h('div', { class: 'small muted' }, 'needs a terminal or a display: run it there, or use the Results tab')
              : null,
        ),
      );
    }
    this.planCard.append(list, h('div', { class: 'small muted' }, `At most ${this.maxRanks} ranks per run (paoflow-studio --max-ranks). Runs continue if the GUI stops.`));
  }

  async launch(step, ranks = 1) {
    // Plan steps are relative to the folder the files were written to; the server wants root paths.
    const cwd = step.rootRelative ? step.cwd : joinPath(this.folder, step.cwd);
    const payload = { step: { ...step, cwd }, ranks };
    try {
      const job = await api.post('/api/jobs/launch', payload);
      toast(`Started: ${job.label}`, 'ok');
      await this.refresh();
      this.select(job.id);
    } catch (error) {
      toast(error.message, 'bad', 7000);
    }
  }

  async refresh() {
    try {
      const result = await api.get('/api/jobs');
      this.jobs = result.jobs;
      this.maxRanks = result.max_ranks;
    } catch (error) {
      toast(error.message, 'bad');
      return;
    }
    this.renderJobs();
  }

  renderJobs() {
    clear(this.jobsCard);
    this.jobsCard.append(h('h3', {}, 'Jobs'));
    if (!this.jobs.length) {
      this.jobsCard.append(h('div', { class: 'small muted' }, 'No runs yet.'));
      return;
    }
    for (const job of this.jobs.slice(0, 60)) {
      this.jobsCard.append(
        h(
          'button',
          { class: `tree-item${job.id === this.selected ? ' selected' : ''}`, onclick: () => this.select(job.id) },
          h('div', { class: 'row' }, h('span', { class: 'grow ellipsis' }, job.label), h('span', { class: `chip ${STATUS_CHIP[job.status] || ''}` }, job.status)),
          h('div', { class: 'small faint ellipsis' }, `${job.cwd} · ${duration(job.elapsed)} · ${new Date(job.started * 1000).toLocaleTimeString()}`),
        ),
      );
    }
  }

  renderLogShell() {
    clear(this.center);
    this.logHead = h('div', { class: 'result-head' });
    this.logBody = h('pre', { class: 'log' });
    this.logBody.addEventListener('scroll', () => {
      const atEnd = this.logBody.scrollTop + this.logBody.clientHeight >= this.logBody.scrollHeight - 8;
      this.follow = atEnd;
    });
    this.center.append(this.logHead, this.logBody);
    this.logHead.append(h('div', { class: 'muted' }, 'Select a job to see its log.'));
  }

  async select(id) {
    this.selected = id;
    this.offset = 0;
    this.follow = true;
    this.logBody.textContent = '';
    this.renderJobs();
    await this.poll(true);
  }

  async poll(force = false) {
    if (!document.getElementById('view-runs').classList.contains('active') && !force) return;
    const running = this.jobs.some((job) => job.status === 'running');
    if (running || force) await this.refresh();
    if (!this.selected) return;
    let chunk;
    try {
      chunk = await api.get('/api/jobs/log', { id: this.selected, offset: this.offset });
    } catch (error) {
      return;
    }
    this.offset = chunk.offset;
    if (chunk.text) {
      this.logBody.append(chunk.text);
      if (this.follow) this.logBody.scrollTop = this.logBody.scrollHeight;
    }
    this.renderLogHead(chunk.job);
  }

  renderLogHead(job) {
    clear(this.logHead);
    this.logHead.append(
      h('div', { class: 'grow' }, h('h2', {}, job.label), h('div', { class: 'small muted mono ellipsis', title: job.command }, job.command)),
      h('div', { class: 'row' },
        h('span', { class: `chip ${STATUS_CHIP[job.status] || ''}` }, job.status + (job.returncode !== null && job.status !== 'running' ? ` (${job.returncode})` : '')),
        h('span', { class: 'small muted' }, duration(job.elapsed)),
        job.status === 'running' ? h('button', { class: 'btn small danger', onclick: async () => {
          try { await api.post('/api/jobs/cancel', { id: job.id }); toast('Cancel requested', 'warn'); } catch (error) { toast(error.message, 'bad'); }
        } }, 'Cancel') : null,
        h('button', { class: 'btn small', onclick: () => this.app.showResults(job.cwd) }, 'Results'),
      ),
    );
  }
}
