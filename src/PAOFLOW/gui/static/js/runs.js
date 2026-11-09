// Runs tab: the run plan of the last generated workflow, local jobs and their logs.

import { api } from './api.js';
import { pickFolder } from './browser.js';
import { clear, copyText, h, joinPath, storage, toast } from './dom.js';
import { chosenQe, loadQe, qePicker } from './qe.js';

const STATUS_CHIP = {
  running: 'accent',
  finished: 'ok',
  failed: 'bad',
  cancelled: 'warn',
  lost: 'warn',
};

function actionLabel(action) {
  const [tab, workflow] = action.split(':');
  if (tab === 'setup')
    return workflow
      ? `Open Setup (${workflow === 'run' ? 'Property run' : workflow})`
      : 'Open Setup';
  return 'Open';
}

function shellWord(value) {
  return /^[\w./:+=-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`;
}

// The command a step runs, as the server builds it: `mpirun -np N` of an MPI step, its QE
// options, and for a driver that runs Quantum ESPRESSO itself (ACBN0, phonons, run_qe.sh) the
// rank variable and the PAOFLOW_* launch variables (PAOFLOW.utils.qe_launch). The variables go
// right before the program, inside the `(cd folder && ...)` of the step, so the text runs as is.
function commandWith(step, extra, launch = null) {
  const base = step.argv.join(' ');
  if (!launch || !step.command.includes(base)) {
    return extra.length ? step.command.replace(base, `${base} ${extra.join(' ')}`) : step.command;
  }
  let program = step.qe_launch || !extra.length ? base : `${base} ${extra.join(' ')}`;
  if (step.mpi && launch.ranks > 1) program = `mpirun -np ${launch.ranks} ${program}`;
  const variables = [];
  if (step.ranks_env) variables.push([step.ranks_env, String(launch.ranks)]);
  if (step.qe_launch) {
    variables.push(['PAOFLOW_MPI_QE', launch.ranks > 1 ? `mpirun -np ${launch.ranks}` : '']);
    if (launch.qeBin) variables.push(['PAOFLOW_QE_BIN', launch.qeBin]);
    if (extra.length) variables.push(['PAOFLOW_QE_OPTIONS', extra.join(' ')]);
  }
  if (variables.length) {
    program = `${variables.map(([name, value]) => `${name}=${shellWord(value)}`).join(' ')} ${program}`;
  }
  return step.command.replace(base, program);
}

// Parallel options of a QE program ("-nk 4 -nd 1") checked as the server checks them: known
// flags with positive integers, one flag per group, images x pools dividing the ranks, task and
// band groups dividing the ranks of a pool. Returns {tokens} or {error}.
export function parseQeOptions(text, groups, ranks) {
  const tokens = text.trim() ? text.trim().split(/\s+/) : [];
  if (!groups) return { tokens };
  const flagOf = {};
  const values = {};
  for (let index = 0; index < tokens.length; index += 2) {
    const flag = tokens[index];
    const value = tokens[index + 1];
    const group = groups.find((item) => item.flags.includes(flag));
    if (!group) {
      return {
        error: `${flag} is not a parallel option (${groups.map((g) => g.flags[0]).join(', ')})`,
      };
    }
    if (value === undefined || !/^[1-9]\d*$/.test(value)) {
      return { error: `${flag} takes a positive integer` };
    }
    if (flagOf[group.name]) {
      return { error: `${flagOf[group.name]} and ${flag} both set the ${group.name}` };
    }
    flagOf[group.name] = flag;
    values[group.name] = Number(value);
  }
  for (const [name, value] of Object.entries(values)) {
    if (value > ranks) {
      return { error: `${flagOf[name]} ${value} is more than the ${ranks} ranks of the run` };
    }
  }
  const split = (values.images || 1) * (values.pools || 1);
  if (ranks % split) {
    return { error: `images x pools (${split}) must divide the ${ranks} ranks` };
  }
  const perPool = ranks / split;
  for (const name of ['task groups', 'band groups']) {
    if (values[name] && perPool % values[name]) {
      return {
        error: `${flagOf[name]} ${values[name]} must divide the ${perPool} ranks of each pool`,
      };
    }
  }
  if ((values.diag || 1) > perPool) {
    return {
      error: `${flagOf.diag} ${values.diag} is more than the ${perPool} ranks of each pool`,
    };
  }
  return { tokens };
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
    this.qeInfo = null;
    this.build();
    loadQe()
      .then(async (info) => {
        this.qeInfo = info;
        this.qeBin = await chosenQe(this.session);
        this.renderPlan();
      })
      .catch(() => {});
    // The QE build chosen in a picker shows in the commands of the drivers that run pw.x.
    document.addEventListener('qe-change', (event) => {
      this.qeBin = event.detail;
      if (this.plan.some((step) => step.qe_launch)) this.renderPlan();
    });
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

  // The plan of a pw.x input chosen in the root (written earlier, by hand or by paoflow-gen-qe).
  async pickQeInput() {
    const setup = this.app.setup;
    const chosen = await pickFolder(this.folder || setup?.workdir || '.', {
      title: 'pw.x input to run',
      allowFiles: true,
      places: false,
    });
    if (!chosen) return;
    try {
      const result = await api.post('/api/gen/qe-input/plan', { path: chosen });
      this.setPlan(result.folder, result.plan, 'qe_input');
    } catch (error) {
      toast(error.message, 'bad', 7000);
    }
  }

  renderPlan() {
    clear(this.planCard);
    this.planCard.append(
      h(
        'div',
        { class: 'row' },
        h('h3', { class: 'grow' }, 'Run plan'),
        this.session.read_only
          ? null
          : h(
              'button',
              {
                class: 'btn small',
                title: 'Choose a pw.x input file in the root and load its run',
                onclick: () => this.pickQeInput(),
              },
              'Run a pw.x input…',
            ),
      ),
    );
    if (!this.session.read_only) this.planCard.append(qePicker(this.session));
    if (!this.plan.length) {
      this.planCard.append(
        h(
          'div',
          { class: 'small muted' },
          'Write the files of a workflow in the Setup tab; its steps appear here. For a pw.x ' +
            'input already on disk, use the button above.',
        ),
      );
      return;
    }
    this.planCard.append(
      h(
        'div',
        { class: 'small muted', style: { marginBottom: '6px' } },
        `${this.workflow || 'workflow'} in `,
        h('code', {}, this.folder),
      ),
    );
    const list = h('ol', { class: 'steps' });
    const groups = this.qeInfo?.parallel_groups || null;
    const programs = this.qeInfo?.programs || [];
    for (const step of this.plan) {
      const ranks = h('input', {
        type: 'number',
        min: 1,
        max: this.maxRanks,
        value: Math.min(4, this.maxRanks),
        style: { width: '64px' },
        title: 'MPI ranks',
      });
      const usesRanks = step.mpi || step.ranks_env || step.qe_launch;
      const runnable = !step.interactive && !step.manual && !this.session.read_only;
      // QE options: a QE program, or a driver that runs pw.x itself (its options are pw.x's).
      const program = step.qe_launch ? 'pw.x' : step.argv?.[0];
      const isQe =
        runnable && (step.qe_launch || programs.includes(program) || /\.x$/.test(program || ''));
      const code = h('code', {}, step.command);
      let parsed = { tokens: [] };
      const runButton = h(
        'button',
        {
          class: 'btn small primary',
          onclick: () =>
            this.launch(
              step.qe_launch
                ? { ...step, qe_options: parsed.tokens || [] }
                : { ...step, argv: [...step.argv, ...(parsed.tokens || [])] },
              usesRanks ? Number(ranks.value) : 1,
            ),
        },
        'Run',
      );
      let optionsRow = null;
      // The command follows the ranks (and the options): it is the one the server runs.
      let refresh = () => {
        code.textContent = commandWith(step, [], {
          ranks: usesRanks ? Number(ranks.value) : 1,
          qeBin: this.qeBin,
        });
      };
      if (isQe) {
        // Runtime options of the QE program (pw.x -nk 4 ...), remembered per program.
        const key = `qe-options:${this.session.root}:${program}`;
        const options = h('input', {
          type: 'text',
          class: 'grow mono',
          value: storage(key) || '',
          placeholder: '-nk 4 -nd 1',
          spellcheck: 'false',
          autocomplete: 'off',
          'aria-label': `${program} options`,
        });
        const problem = h('div', { class: 'option-error', hidden: true });
        refresh = () => {
          parsed = parseQeOptions(options.value, groups, usesRanks ? Number(ranks.value) : 1);
          problem.hidden = !parsed.error;
          problem.textContent = parsed.error || '';
          options.classList.toggle('invalid', Boolean(parsed.error));
          code.textContent = commandWith(step, parsed.tokens || [], {
            ranks: usesRanks ? Number(ranks.value) : 1,
            qeBin: this.qeBin,
          });
          runButton.disabled = Boolean(parsed.error);
        };
        options.addEventListener('input', () => {
          storage(key, options.value.trim());
          refresh();
        });
        const hint = groups
          ? groups.map((group) => `${group.flags[0]} ${group.what}`).join(' · ')
          : '-nk pools · -ni images · -nd linear algebra · -nt task groups · -nb band groups';
        optionsRow = h(
          'div',
          { class: 'qe-options' },
          h('div', { class: 'row' }, h('span', { class: 'small muted' }, 'options'), options),
          problem,
          h('div', { class: 'small faint' }, hint),
        );
      }
      if (runnable) {
        ranks.addEventListener('input', () => refresh());
        refresh();
      }
      list.append(
        h(
          'li',
          {},
          h(
            'div',
            { class: 'row' },
            h('strong', { class: 'grow' }, step.label),
            usesRanks
              ? h(
                  'span',
                  { class: 'chip accent' },
                  step.mpi ? 'MPI' : step.qe_launch ? 'pw.x MPI' : step.ranks_env,
                )
              : null,
          ),
          step.manual
            ? null
            : h(
                'div',
                { class: 'cmd' },
                code,
                h(
                  'button',
                  { class: 'icon-btn', title: 'Copy', onclick: () => copyText(code.textContent) },
                  '⧉',
                ),
              ),
          optionsRow,
          step.note && !step.interactive ? h('div', { class: 'small muted' }, step.note) : null,
          runnable
            ? h(
                'div',
                { class: 'row', style: { marginTop: '4px' } },
                usesRanks ? [h('span', { class: 'small muted' }, 'ranks'), ranks] : null,
                h('span', { class: 'grow' }),
                runButton,
              )
            : step.action
              ? h(
                  'div',
                  { class: 'row', style: { marginTop: '4px' } },
                  h('span', { class: 'grow' }),
                  h(
                    'button',
                    {
                      class: 'btn small primary',
                      onclick: () => this.app.doAction(step.action, this.folder),
                    },
                    actionLabel(step.action),
                  ),
                )
              : step.interactive
                ? h(
                    'div',
                    { class: 'small muted' },
                    'needs a terminal or a display: run it there, or use the Results tab',
                  )
                : null,
        ),
      );
    }
    this.planCard.append(
      list,
      h(
        'div',
        { class: 'small muted' },
        `At most ${this.maxRanks} ranks per run (paoflow-studio --max-ranks). Runs continue if the GUI stops.`,
      ),
    );
  }

  async launch(step, ranks = 1) {
    // Plan steps are relative to the folder the files were written to; the server wants root paths.
    const cwd = step.rootRelative ? step.cwd : joinPath(this.folder, step.cwd);
    // The chosen QE build runs the QE programs and goes first on the PATH of every run.
    const qeBin = await chosenQe(this.session).catch(() => null);
    const payload = { step: { ...step, cwd }, ranks, qe_bin: qeBin };
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
          {
            class: `tree-item${job.id === this.selected ? ' selected' : ''}`,
            onclick: () => this.select(job.id),
          },
          h(
            'div',
            { class: 'row' },
            h('span', { class: 'grow ellipsis' }, job.label),
            h('span', { class: `chip ${STATUS_CHIP[job.status] || ''}` }, job.status),
          ),
          h(
            'div',
            { class: 'small faint ellipsis' },
            `${job.cwd} · ${duration(job.elapsed)} · ${new Date(job.started * 1000).toLocaleTimeString()}`,
          ),
        ),
      );
    }
  }

  renderLogShell() {
    clear(this.center);
    this.logHead = h('div', { class: 'result-head' });
    this.logBody = h('pre', { class: 'log' });
    this.logBody.addEventListener('scroll', () => {
      const atEnd =
        this.logBody.scrollTop + this.logBody.clientHeight >= this.logBody.scrollHeight - 8;
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
    } catch (_error) {
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
      h(
        'div',
        { class: 'grow' },
        h('h2', {}, job.label),
        h('div', { class: 'small muted mono ellipsis', title: job.command }, job.command),
      ),
      h(
        'div',
        { class: 'row' },
        h(
          'span',
          { class: `chip ${STATUS_CHIP[job.status] || ''}` },
          job.status +
            (job.returncode !== null && job.status !== 'running' ? ` (${job.returncode})` : ''),
        ),
        h('span', { class: 'small muted' }, duration(job.elapsed)),
        job.status === 'running'
          ? h(
              'button',
              {
                class: 'btn small danger',
                onclick: async () => {
                  try {
                    await api.post('/api/jobs/cancel', { id: job.id });
                    toast('Cancel requested', 'warn');
                  } catch (error) {
                    toast(error.message, 'bad');
                  }
                },
              },
              'Cancel',
            )
          : null,
        h(
          'button',
          { class: 'btn small', onclick: () => this.app.showResults(job.cwd) },
          'Results',
        ),
      ),
    );
  }
}
