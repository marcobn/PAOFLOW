// Setup tab: working folder, paoflow-gen workflows as forms, live preview, writing the files.

import { api } from './api.js';
import { pickFolder, relativeTo } from './browser.js';
import { codeView, diffView } from './codeview.js';
import {
  clear,
  copyText,
  debounce,
  dialog,
  formatBytes,
  h,
  joinPath,
  storage,
  toast,
} from './dom.js';
import { FormView } from './form.js';
import { renderFigure } from './plots.js';
import { qePicker } from './qe.js';

const STATUS_LABEL = { new: 'new', changed: 'changed', identical: 'unchanged' };

function fmt(value, digits = 3) {
  return typeof value === 'number' ? Number(value.toFixed(digits)).toString() : value;
}

export class SetupView {
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.session = app.session;
    this.workdir = storage(`workdir:${this.session.root}`) || this.session.start || '.';
    this.workflow = null;
    this.workflows = [];
    this.detection = null;
    this.resolved = null;
    this.selectedFile = null;
    this.showDiff = false;
    this.requestId = 0;
    this.options = {};
    this.build();
    this.scheduleResolve = debounce(() => this.resolve(), 250);
  }

  build() {
    this.side = h('div', { class: 'pane side' });
    this.center = h('div', { class: 'pane' });
    this.preview = h('div', { class: 'pane preview' });
    this.root.append(
      h('div', { class: 'columns setup-grid' }, this.side, this.center, this.preview),
    );
    this.folderCard = h('div', { class: 'card' });
    this.workflowCard = h('div', { class: 'card' });
    this.sectionCard = h('div', { class: 'card', hidden: true });
    this.side.append(this.folderCard, this.workflowCard, this.sectionCard);
  }

  async init() {
    const { workflows } = await api.get('/api/gen/workflows');
    this.workflows = workflows;
    await this.openFolder(this.workdir);
  }

  // ----------------------------------------------------------------------- folder
  async openFolder(path) {
    try {
      this.detection = await api.get('/api/workspace/detect', { path });
    } catch (error) {
      toast(error.message, 'bad');
      if (path !== '.') return this.openFolder('.');
      return;
    }
    this.workdir = this.detection.path;
    storage(`workdir:${this.session.root}`, this.workdir);
    this.app.setWorkdir(this.workdir);
    this.renderFolder();
    this.renderWorkflows();
    const remembered = storage(`workflow:${this.session.root}:${this.workdir}`);
    const suggested = this.detection.suggestions.find((s) =>
      this.workflows.some((w) => w.name === s.workflow),
    );
    this.selectWorkflow(remembered || suggested?.workflow || 'run');
  }

  // Files change outside the form (a pw.x run writes the save, a UPF is copied in): look at the
  // folder again and resolve, so the defaults (Save directory, UPFs, ibrav, band path) follow. Only
  // the edited fields keep their values.
  async refreshFolder(force = false) {
    if (!this.detection || this.refreshing) return;
    this.refreshing = true;
    try {
      const detection = await api.get('/api/workspace/detect', { path: this.workdir });
      const changed = JSON.stringify(detection) !== JSON.stringify(this.detection);
      if (changed) {
        this.detection = detection;
        this.renderFolder();
        this.renderWorkflows();
      }
      if ((changed || force) && this.form) await this.resolve();
    } catch (_error) {
      // a folder that went away is reported by the next action
    } finally {
      this.refreshing = false;
    }
  }

  // The Setup tab is shown: refresh now, then every few seconds while it stays in view.
  activate() {
    this.refreshFolder(true);
    clearInterval(this.folderTimer);
    this.folderTimer = setInterval(() => {
      const shown = document.getElementById('view-setup').classList.contains('active');
      if (shown && document.visibilityState === 'visible') this.refreshFolder();
    }, 4000);
  }

  renderFolder() {
    const d = this.detection;
    clear(this.folderCard);
    this.folderCard.append(
      h('h3', {}, 'Working folder'),
      h(
        'div',
        { class: 'row' },
        h(
          'code',
          { class: 'grow ellipsis', title: d.absolute },
          d.path === '.' ? '(root)' : d.path,
        ),
        h(
          'button',
          {
            class: 'btn small',
            onclick: async () => {
              const chosen = await pickFolder(this.workdir, { places: false });
              if (chosen !== null) this.openFolder(chosen);
            },
          },
          'Change…',
        ),
      ),
    );
    const summary = d.save_summary;
    const facts = [];
    if (summary) {
      const lattice = summary.bravais_index ? `ibrav ${summary.bravais_index}` : 'cell';
      facts.push(['System', `${summary.formula} · ${summary.nat} atoms · ${lattice}`]);
      facts.push([
        'Electrons',
        `${fmt(summary.nelec, 1)} e · ${summary.nbnd} bands${summary.spinorbit ? ' · SOC' : summary.lsda ? ' · spin-polarised' : ''}`,
      ]);
      facts.push([
        'Character',
        summary.metal
          ? `metal (${summary.occupations})`
          : summary.gap_ev !== null
            ? `gap ${fmt(summary.gap_ev, 2)} eV`
            : summary.occupations || '?',
      ]);
      if (summary.kgrid) facts.push(['k-grid', summary.kgrid.join(' × ')]);
      facts.push([
        'Pseudos',
        summary.species
          .map(
            (s) =>
              `${s.name}${s.pseudo ? ` ${s.pseudo.pseudo_type}${s.pseudo.relativistic === 'full' ? ' FR' : ''}` : ' ?'}`,
          )
          .join(', '),
      ]);
      if (summary.two_dimensional) facts.push(['Geometry', '2D slab (vacuum along c)']);
    } else if (d.template_summary) {
      const t = d.template_summary;
      facts.push(['pw.x input', `${d.pw_inputs[0]} · prefix ${t.prefix || '?'}`]);
      facts.push([
        'Atoms',
        Object.entries(t.counts)
          .map(([k, v]) => `${k}${v > 1 ? v : ''}`)
          .join(' ') || '?',
      ]);
      if (t.nelec !== null) facts.push(['Valence e', fmt(t.nelec, 1)]);
      facts.push(['Occupations', t.occupations || 'fixed']);
    }
    if (facts.length) {
      this.folderCard.append(
        h(
          'dl',
          { class: 'kv', style: { marginTop: '8px' } },
          facts.map(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]),
        ),
      );
    }
    const found = [];
    if (d.saves.length) found.push(h('span', { class: 'chip accent' }, d.saves.join(', ')));
    if (d.pw_inputs.length)
      found.push(
        h(
          'span',
          { class: 'chip' },
          `${d.pw_inputs.length} pw.x input${d.pw_inputs.length > 1 ? 's' : ''}`,
        ),
      );
    if (d.drivers.length) found.push(h('span', { class: 'chip' }, d.drivers.join(', ')));
    if (d.system_py) found.push(h('span', { class: 'chip accent' }, 'system.py'));
    if (found.length)
      this.folderCard.append(h('div', { class: 'row wrap', style: { marginTop: '8px' } }, found));
    for (const warning of d.warnings)
      this.folderCard.append(
        h('div', { class: 'callout warn', style: { marginTop: '8px' } }, warning),
      );
  }

  renderWorkflows() {
    clear(this.workflowCard);
    this.workflowCard.append(h('h3', {}, 'Workflow'));
    const suggested = new Set(this.detection.suggestions.map((s) => s.workflow));
    const list = h('div', { class: 'workflow-list' });
    const group = (title, items) => {
      list.append(h('div', { class: 'workflow-group' }, title));
      for (const item of items) {
        list.append(
          h(
            'button',
            {
              class: `workflow-item${item.name === this.workflow ? ' selected' : ''}`,
              dataset: { name: item.name },
              onclick: () => this.selectWorkflow(item.name),
            },
            h(
              'div',
              { class: 'title' },
              item.title,
              suggested.has(item.name) ? h('span', { class: 'chip ok' }, 'suggested') : null,
            ),
            h('div', { class: 'desc' }, item.description),
          ),
        );
      }
    };
    group(
      'paoflow-gen',
      this.workflows.filter((w) => w.kind === 'workflow'),
    );
    group(
      'tools',
      this.workflows.filter((w) => w.kind === 'tool'),
    );
    this.workflowCard.append(list);
  }

  // ----------------------------------------------------------------------- workflow form
  answersKey() {
    return `answers:${this.session.root}:${this.workdir}:${this.workflow}`;
  }

  selectWorkflow(name) {
    if (!this.workflows.some((w) => w.name === name)) name = 'run';
    this.workflow = name;
    storage(`workflow:${this.session.root}:${this.workdir}`, name);
    for (const node of this.workflowCard.querySelectorAll('.workflow-item')) {
      node.classList.toggle('selected', node.dataset.name === name);
    }
    clear(this.center);
    clear(this.preview);
    this.resolved = null;
    this.selectedFile = null;
    this.requestId += 1; // a resolve still in flight belongs to the previous form: drop it
    const info = this.workflows.find((w) => w.name === name);
    if (info.kind === 'tool') {
      this.form = null; // tools have no form: a pending resolve then does nothing
      this.sectionCard.hidden = true;
      if (name === 'qe_input') this.renderQeInput(info);
      else this.renderBasis(info);
      return;
    }
    const saved = storage(this.answersKey()) || {};
    this.options = saved.options || {};
    const advanced = h('input', {
      type: 'checkbox',
      onchange: () => this.form.setShowAdvanced(advanced.checked),
    });
    this.center.append(
      h(
        'div',
        { class: 'form-head' },
        h(
          'div',
          { class: 'grow' },
          h('h2', {}, info.title),
          h('div', { class: 'muted' }, info.description),
        ),
        h(
          'label',
          { class: 'row small muted' },
          h('span', { class: 'switch' }, advanced, h('span')),
          'advanced',
        ),
        h(
          'button',
          {
            class: 'btn small',
            title: 'Forget the answers of this form',
            onclick: () => {
              storage(this.answersKey(), {});
              this.selectWorkflow(name);
            },
          },
          'Reset',
        ),
      ),
      this.renderOptions(name),
    );
    const formBox = h('div');
    this.center.append(formBox);
    this.form = new FormView(formBox, {
      onChange: (answers, immediate) => {
        storage(this.answersKey(), { answers, options: this.options });
        if (immediate) this.resolve();
        else this.scheduleResolve();
      },
      browse: async (item) => {
        const chosen = await pickFolder(this.workdir, { title: item.label });
        return chosen === null ? null : relativeTo(this.workdir, chosen);
      },
    });
    this.form.reset(saved.answers || {});
    this.renderPreviewShell();
    this.resolve();
  }

  renderOptions(name) {
    const set = (key, value) => {
      if (value === '' || value === false) delete this.options[key];
      else this.options[key] = value;
      storage(this.answersKey(), { answers: this.form.answers, options: this.options });
      this.scheduleResolve();
    };
    const box = h('div', { class: 'section' }, h('h3', {}, 'Output'));
    const field = (label, control, hint) =>
      h(
        'div',
        { class: 'field' },
        h(
          'div',
          {},
          h('label', { class: 'name' }, label),
          hint ? h('div', { class: 'hint' }, hint) : null,
        ),
        h('div', {}, h('div', { class: 'control' }, control)),
      );
    if (name === 'elphon') {
      const folder = h('input', {
        type: 'text',
        value: this.options.folder || '',
        placeholder: this.workdir,
        oninput: () => set('folder', folder.value.trim()),
      });
      box.append(
        field(
          'System folder',
          [
            folder,
            h(
              'button',
              {
                class: 'btn small',
                onclick: async () => {
                  const chosen = await pickFolder(this.workdir, {
                    title: 'System folder (created if missing below a chosen folder)',
                    places: false,
                  });
                  if (chosen !== null) {
                    folder.value = chosen;
                    set('folder', chosen);
                  }
                },
              },
              'Browse…',
            ),
          ],
          'relative to the root; default: the working folder (paoflow-gen -o)',
        ),
      );
    } else {
      const defaults = {
        run: 'main.py',
        acbn0: 'main.acbn0.py',
        phonon: 'main.phonon.py',
        sscha: 'main.sscha.py',
      };
      const out = h('input', {
        type: 'text',
        value: this.options.out || '',
        placeholder: defaults[name],
        oninput: () => set('out', out.value.trim()),
      });
      box.append(field('Driver file', out, 'in the working folder (paoflow-gen -o)'));
      if (name !== 'sscha') {
        const plot = h('input', {
          type: 'checkbox',
          checked: Boolean(this.options.plot),
          onchange: () => set('plot', plot.checked),
        });
        box.append(
          field(
            'Always write the plot script',
            h('label', { class: 'switch' }, plot, h('span')),
            'paoflow-gen --plot',
          ),
        );
      }
    }
    return box;
  }

  async resolve() {
    if (!this.form) return;
    const id = ++this.requestId;
    let resolved;
    try {
      resolved = await api.post('/api/gen/resolve', this.payload());
    } catch (error) {
      if (id === this.requestId) toast(error.message, 'bad');
      return;
    }
    if (id !== this.requestId) return;
    this.resolved = resolved;
    this.form.render(resolved);
    this.renderSections();
    this.renderPreview();
    this.renderWidgets();
  }

  // Special fields: the band path gets a preview of the path in the first Brillouin zone.
  renderWidgets() {
    const items = this.resolved.items;
    const value = (key) => items.find((i) => i.key === key)?.value;
    for (const item of items) {
      if (item.type !== 'question' || item.widget !== 'bandpath' || !item.visible) continue;
      const field = this.form.fields.get(item.key);
      if (!field || field.root.hidden) continue;
      if (!field.preview) {
        field.caption = h('div', { class: 'small muted bz-caption' });
        field.preview = h('div', { class: 'bz-preview' });
        field.root.append(field.caption, field.preview);
      }
      const request = {
        workdir: this.workdir,
        savedir: value('savedir'),
        band_path: item.value,
        points: value('high_sym') || [],
      };
      const key = JSON.stringify(request);
      if (key === field.previewKey) continue;
      field.previewKey = key;
      api
        .post('/api/gen/bandpath', request)
        .then((result) => {
          if (field.previewKey !== key) return;
          if (result.error) {
            const current = value('savedir');
            const saves = (result.saves || []).filter((name) => name !== current);
            const savedir = this.form.fields.get('savedir');
            clear(field.caption).append(
              `Preview: ${result.error}`,
              ...saves.map((name) =>
                h(
                  'button',
                  {
                    class: 'btn small',
                    type: 'button',
                    style: { marginLeft: '6px' },
                    onclick: () => savedir?.pick(name),
                  },
                  `Use ${name}`,
                ),
              ),
            );
            field.preview.hidden = true;
            return;
          }
          field.preview.hidden = false;
          field.caption.textContent = `Path in the first Brillouin zone · length ${result.length.toFixed(3)} (2π/a) · drag to rotate`;
          renderFigure(field.preview, result.figure, 'band_path');
        })
        .catch((error) => {
          field.caption.textContent = error.message;
        });
    }
  }

  payload(extra = {}) {
    return {
      workflow: this.workflow,
      workdir: this.workdir,
      answers: this.form.answers,
      options: this.options,
      ...extra,
    };
  }

  renderSections() {
    const sections = this.form.visibleSections();
    clear(this.sectionCard);
    this.sectionCard.hidden = !sections.length;
    this.sectionCard.append(h('h3', {}, 'Sections'));
    for (const [name, box] of sections) {
      const errors = box.querySelectorAll('.field.invalid:not([hidden])').length;
      this.sectionCard.append(
        h(
          'div',
          {},
          h(
            'button',
            {
              class: 'btn link',
              onclick: () => box.scrollIntoView({ behavior: 'smooth', block: 'start' }),
            },
            name,
          ),
          errors
            ? h('span', { class: 'chip bad', style: { marginLeft: '6px' } }, `${errors}`)
            : null,
        ),
      );
    }
  }

  // ----------------------------------------------------------------------- preview
  renderPreviewShell() {
    clear(this.preview);
    this.previewTarget = h('span', { class: 'small muted ellipsis grow' });
    this.fileTabs = h('div', { class: 'file-tabs' });
    this.previewBody = h('div', { class: 'preview-body' });
    this.diffToggle = h('input', {
      type: 'checkbox',
      onchange: () => {
        this.showDiff = this.diffToggle.checked;
        this.renderFile();
      },
    });
    this.writeButton = h(
      'button',
      { class: 'btn primary', onclick: () => this.write() },
      'Write files',
    );
    this.statusLine = h('span', { class: 'small muted grow' });
    this.preview.append(
      h(
        'div',
        { class: 'preview-head' },
        h('h3', {}, 'Files'),
        this.previewTarget,
        h(
          'label',
          { class: 'row small muted' },
          h('span', { class: 'switch' }, this.diffToggle, h('span')),
          'diff',
        ),
      ),
      this.fileTabs,
      this.previewBody,
      h(
        'div',
        { class: 'preview-foot' },
        this.statusLine,
        h('button', { class: 'btn', onclick: () => this.showPython() }, 'Python'),
        h('button', { class: 'btn', onclick: () => this.showPlan() }, 'Next steps'),
        this.writeButton,
      ),
    );
    if (this.session.read_only) this.writeButton.disabled = true;
  }

  renderPreview() {
    const r = this.resolved;
    this.previewTarget.textContent = r.target ? `→ ${r.target}` : '';
    clear(this.fileTabs);
    if (!r.complete) {
      this.writeButton.disabled = true;
      const errors = Object.entries(r.errors);
      this.statusLine.textContent = `${errors.length} field${errors.length > 1 ? 's' : ''} to complete`;
      clear(this.previewBody);
      const list = h('ul');
      for (const [key, message] of errors) {
        list.append(
          h(
            'li',
            {},
            h(
              'button',
              {
                class: 'btn link',
                onclick: () => {
                  const node = this.center.querySelector(`.field[data-key="${CSS.escape(key)}"]`);
                  if (node) node.scrollIntoView({ behavior: 'smooth', block: 'center' });
                },
              },
              key.startsWith('_') ? 'configuration' : key,
            ),
            ': ',
            message,
          ),
        );
      }
      this.previewBody.append(
        h(
          'div',
          { style: { padding: '16px' } },
          h('div', { class: 'callout warn' }, 'Complete the form to see the files.'),
          list,
        ),
      );
      return;
    }
    this.writeButton.disabled = this.session.read_only;
    const counts = { new: 0, changed: 0, identical: 0 };
    for (const file of r.files) counts[file.status] += 1;
    this.statusLine.textContent = `${r.files.length} files: ${counts.new} new, ${counts.changed} changed, ${counts.identical} unchanged`;
    if (!this.selectedFile || !r.files.some((f) => f.path === this.selectedFile))
      this.selectedFile = r.files[0].path;
    for (const file of r.files) {
      this.fileTabs.append(
        h(
          'button',
          {
            class: 'file-tab',
            'aria-selected': String(file.path === this.selectedFile),
            title: `${file.location} (${STATUS_LABEL[file.status]})`,
            onclick: (ev) => {
              this.selectedFile = file.path;
              for (const node of this.fileTabs.children)
                node.setAttribute('aria-selected', String(node === ev.currentTarget));
              this.renderFile();
            },
          },
          h('span', { class: `dot ${file.status}` }),
          file.path,
        ),
      );
    }
    this.renderFile();
  }

  renderFile() {
    if (!this.resolved?.files) return;
    const file = this.resolved.files.find((f) => f.path === this.selectedFile);
    clear(this.previewBody);
    if (!file) return;
    if (this.showDiff && file.status === 'changed' && file.diff)
      this.previewBody.append(diffView(file.diff));
    else if (this.showDiff && file.status !== 'changed') {
      this.previewBody.append(
        h(
          'div',
          { class: 'empty' },
          file.status === 'new' ? 'new file (nothing to compare)' : 'identical to the file on disk',
        ),
      );
    } else this.previewBody.append(codeView(file.text, file.path));
  }

  // ----------------------------------------------------------------------- actions
  async write() {
    const r = this.resolved;
    if (!r?.complete) return;
    const changed = r.files.filter((f) => f.status === 'changed');
    let overwrite = [];
    let keepFiles = [];
    let backup = true;
    if (changed.length) {
      const boxes = changed.map((f) => [f.path, h('input', { type: 'checkbox', checked: true })]);
      const keep = h('input', { type: 'checkbox', checked: true });
      const answer = await dialog({
        title: 'Overwrite existing files?',
        body: h(
          'div',
          { class: 'stack' },
          h('div', {}, 'These files exist with a different content:'),
          boxes.map(([path, box]) => h('label', { class: 'row' }, box, h('code', {}, path))),
          h('div', { class: 'small muted' }, 'Unchecked files are left as they are.'),
          h('label', { class: 'row muted' }, keep, 'keep a backup copy (<file>.bak-<date>)'),
        ),
        actions: [
          { label: 'Cancel', value: null },
          { label: 'Write', kind: 'primary', value: true },
        ],
      });
      if (!answer) return;
      overwrite = boxes.filter(([, box]) => box.checked).map(([path]) => path);
      keepFiles = boxes.filter(([, box]) => !box.checked).map(([path]) => path);
      backup = keep.checked;
    }
    try {
      const result = await api.post(
        '/api/gen/write',
        this.payload({ overwrite, keep: keepFiles, backup }),
      );
      const kept = result.kept.length ? `, kept ${result.kept.length}` : '';
      toast(
        `Wrote ${result.written.length} file${result.written.length === 1 ? '' : 's'} in ${result.folder}${kept}`,
        'ok',
      );
      this.app.notifyWritten(result);
      await this.resolve();
      this.showPlan(result.plan, result.folder);
    } catch (error) {
      if (error.status === 409)
        toast(`Changed on disk meanwhile: ${error.details.conflicts.join(', ')}`, 'warn');
      else toast(error.message, 'bad');
    }
  }

  showPython() {
    if (!this.resolved?.complete) return toast('Complete the form first', 'warn');
    const text = this.resolved.python;
    dialog({
      title: 'Python equivalent',
      body: h(
        'div',
        {},
        h(
          'div',
          { class: 'small muted', style: { marginBottom: '8px' } },
          'Writes the same files without the GUI (paths relative to the working folder).',
        ),
        codeView(text, 'gen.py'),
      ),
      actions: [
        { label: 'Copy', value: () => copyText(text) },
        { label: 'Close', value: null },
      ],
      wide: true,
    });
  }

  showPlan(plan, folder, workflow) {
    if (!plan && workflow) {
      return toast(
        'The server sent no run plan: it is older than this page. Restart paoflow-studio.',
        'warn',
        8000,
      );
    }
    plan = plan || this.resolved?.plan;
    if (!plan) return toast('Complete the form first', 'warn');
    folder = folder || this.resolved.target;
    workflow = workflow || this.workflow;
    const list = h('ol', { class: 'steps' });
    for (const step of plan) {
      list.append(
        h(
          'li',
          {},
          h(
            'div',
            { class: 'row' },
            h('strong', { class: 'grow' }, step.label),
            step.mpi ? h('span', { class: 'chip accent' }, 'MPI') : null,
            step.ranks_env ? h('span', { class: 'chip accent' }, `${step.ranks_env}=ranks`) : null,
            step.interactive ? h('span', { class: 'chip' }, 'interactive') : null,
          ),
          step.manual
            ? null
            : h(
                'div',
                { class: 'cmd' },
                h('code', {}, step.command),
                h(
                  'button',
                  { class: 'icon-btn', title: 'Copy', onclick: () => copyText(step.command) },
                  '⧉',
                ),
              ),
          step.action ? h('div', { class: 'small muted' }, 'done in the Setup tab') : null,
          step.note ? h('div', { class: 'small muted' }, step.note) : null,
        ),
      );
    }
    const actions = [{ label: 'Close', value: null }];
    if (this.app.runs && !this.session.read_only)
      actions.unshift({ label: 'Open in Runs', kind: 'primary', value: 'runs' });
    dialog({ title: `Next steps in ${folder}`, body: list, actions }).then((choice) => {
      if (choice === 'runs') this.app.showRuns(folder, plan, workflow);
    });
  }

  // ----------------------------------------------------------------------- tools
  // A text input with a Browse… button; the pick is written relative to the working folder, or
  // as an absolute path inside a read-only folder (--allow-read, PAOFLOW's PSEUDOS and BASIS).
  withBrowse(node, onPick, options) {
    const button = h(
      'button',
      {
        class: 'btn small',
        type: 'button',
        onclick: async () => {
          const chosen = await pickFolder(this.workdir, options);
          if (chosen === null) return;
          node.value = relativeTo(this.workdir, chosen);
          onPick(node.value);
        },
      },
      'Browse…',
    );
    return [node, button];
  }

  renderQeInput(info) {
    const state = storage(`qe-input:${this.session.root}`) || {
      source: 'auto',
      ibrav: 'auto',
      metallic: 'auto',
      magnetic: 'auto',
      assume_isolated_2d: true,
    };
    const save = () => storage(`qe-input:${this.session.root}`, state);
    const input = (key, placeholder) => {
      const node = h('input', {
        type: 'text',
        value: state[key] ?? '',
        placeholder,
        oninput: () => {
          state[key] = node.value.trim();
          save();
        },
      });
      return node;
    };
    const select = (key, options) => {
      const node = h(
        'select',
        {
          onchange: () => {
            state[key] = node.value;
            save();
          },
        },
        options.map((o) => h('option', { value: o, selected: state[key] === o }, o)),
      );
      return node;
    };
    const check = (key) => {
      const node = h('input', {
        type: 'checkbox',
        checked: Boolean(state[key]),
        onchange: () => {
          state[key] = node.checked;
          save();
        },
      });
      return h('label', { class: 'switch' }, node, h('span'));
    };
    const field = (label, control, hint) =>
      h(
        'div',
        { class: 'field' },
        h(
          'div',
          {},
          h('label', { class: 'name' }, label),
          hint ? h('div', { class: 'hint' }, hint) : null,
        ),
        h('div', {}, h('div', { class: 'control' }, control)),
      );
    const kpoints = h('input', {
      type: 'text',
      value: (state.kpoints || []).join(' '),
      placeholder: 'database value',
      oninput: () => {
        const v = kpoints.value
          .trim()
          .split(/[\s,]+/)
          .filter(Boolean)
          .map(Number);
        state.kpoints = v.length === 3 ? v : null;
        save();
      },
    });
    const inputs = h('div', { class: 'section' });
    const listInputs = async () => {
      let listing;
      try {
        listing = await api.get('/api/fs/list', { path: this.workdir });
      } catch (_error) {
        return;
      }
      const files = listing.entries.filter((entry) => (entry.tags || []).includes('pw-input'));
      const rows = files.map((entry) =>
        h(
          'div',
          { class: 'row input-row' },
          h('code', { class: 'grow ellipsis' }, entry.name),
          h('span', { class: 'faint small' }, formatBytes(entry.size)),
          h(
            'button',
            {
              class: 'btn small primary',
              title: `pw.x -in ${entry.name}, in the Runs tab`,
              onclick: () => this.runQeInput(joinPath(listing.path, entry.name)),
            },
            'Run pw.x…',
          ),
        ),
      );
      // Native append: no arrays, no nulls.
      clear(inputs).append(
        h('h3', {}, 'pw.x inputs in this folder'),
        qePicker(this.session),
        ...(rows.length
          ? rows
          : [
              h(
                'div',
                { class: 'small muted' },
                'None yet: fetch an entry below and write it; it then appears here with its run.',
              ),
            ]),
      );
    };
    listInputs();
    this.center.append(
      h(
        'div',
        { class: 'form-head' },
        h(
          'div',
          { class: 'grow' },
          h('h2', {}, info.title),
          h('div', { class: 'muted' }, info.description),
        ),
      ),
      inputs,
      h(
        'div',
        { class: 'section' },
        h('h3', {}, 'Entry'),
        field(
          'Identifier',
          input('identifier', 'aflow:0a66d228d896a855 · 2C-1 · URL'),
          'AFLOW AUID, AFLOWDATA or material-page URL; C2DB UID',
        ),
        field('Database', select('source', ['auto', 'aflow', 'c2db'])),
        field(
          'Pseudopotential folder',
          this.withBrowse(
            input('pseudo_dir', '.'),
            (value) => {
              state.pseudo_dir = value;
              save();
            },
            { title: 'Pseudopotential folder' },
          ),
          'relative to the working folder, or a read-only folder: <element>.upf, PeriodicTableJSON.json, reference.json',
        ),
        field('Output file', input('filename', '<compound>.scf.in')),
      ),
      h(
        'div',
        { class: 'section' },
        h('h3', {}, 'Overrides'),
        field(
          'Spin-orbit cards',
          check('soc'),
          'automatic for fully relativistic pseudopotentials',
        ),
        field('Metallic', select('metallic', ['auto', 'true', 'false'])),
        field('Magnetic', select('magnetic', ['auto', 'true', 'false'])),
        field('k-point grid', kpoints, 'three integers (k3 = 1 for 2D)'),
        field('Lattice', select('ibrav', ['auto', '0']), 'auto: ibrav + celldm; 0: explicit cell'),
        field('Smearing degauss (Ry)', input('degauss', '0.05')),
        field('Bands nbnd', input('nbnd', 'from the extended PAO basis')),
        field('Symmetry tolerance', input('symprec', '1e-4')),
        field(
          "assume_isolated = '2D'",
          check('assume_isolated_2d'),
          'written for 2D entries (C2DB slabs) only, never for 3D ones',
        ),
      ),
    );
    const body = h(
      'div',
      { class: 'preview-body' },
      h('div', { class: 'empty' }, 'Fetch the entry to preview the input (needs the network).'),
    );
    const status = h('span', { class: 'small muted grow' });
    let last = null;
    const request = () => ({
      ...state,
      workdir: this.workdir,
      metallic: state.metallic === 'auto' || !state.metallic ? null : state.metallic === 'true',
      magnetic: state.magnetic === 'auto' || !state.magnetic ? null : state.magnetic === 'true',
    });
    const fetchButton = h(
      'button',
      {
        class: 'btn',
        onclick: async () => {
          clear(body).append(
            h('div', { class: 'empty' }, h('span', { class: 'spinner' }), ' fetching…'),
          );
          try {
            last = await api.post('/api/gen/qe-input/preview', request());
          } catch (error) {
            clear(body).append(
              h(
                'div',
                { style: { padding: '16px' } },
                h('div', { class: 'callout bad' }, error.message),
              ),
            );
            return;
          }
          const rec = last.record;
          status.textContent = `${last.filename} · ${STATUS_LABEL[last.status]} · ${rec.compound} (${rec.source}), ${rec.natoms} atoms, ${rec.dimensionality}${rec.metallic ? ', metallic' : ''}${rec.magnetic ? ', magnetic' : ''}`;
          clear(body).append(codeView(last.text, last.filename));
          writeButton.disabled = this.session.read_only;
        },
      },
      'Fetch & preview',
    );
    const writeButton = h(
      'button',
      {
        class: 'btn primary',
        disabled: true,
        onclick: async () => {
          let overwrite = false;
          if (last && last.status === 'changed') {
            overwrite = await dialog({
              title: 'Overwrite?',
              body: `${last.filename} exists with a different content.`,
              actions: [
                { label: 'Cancel', value: false },
                { label: 'Overwrite', kind: 'primary', value: true },
              ],
            });
            if (!overwrite) return;
          }
          try {
            const result = await api.post('/api/gen/qe-input/write', { ...request(), overwrite });
            toast(`Wrote ${result.written}`, 'ok');
            this.app.notifyWritten(result);
            listInputs();
            this.showPlan(result.plan, result.folder, 'qe_input');
          } catch (error) {
            toast(error.message, 'bad');
          }
        },
      },
      'Write',
    );
    if (!this.session.features.pw) {
      const note = h(
        'div',
        { class: 'callout warn', style: { marginBottom: '12px' } },
        'paoflow-studio found no Quantum ESPRESSO (pw.x on the PATH, next to it, in ~/Programs, ' +
          '~/Local/Programs, /opt, /usr/local or a conda environment), so the Runs tab cannot ' +
          'start the scf run. Load QE in the shell or add its folder with --qe-bin DIR, and start ' +
          'paoflow-studio again; or run pw.x yourself.',
      );
      this.center.insertBefore(note, this.center.children[1] || null);
    }
    this.preview.append(
      h('div', { class: 'preview-head' }, h('h3', {}, 'pw.x input')),
      body,
      h('div', { class: 'preview-foot' }, status, fetchButton, writeButton),
    );
  }

  // The run plan of a pw.x input on disk (path relative to the root), shown like a written one.
  async runQeInput(path) {
    try {
      const result = await api.post('/api/gen/qe-input/plan', { path });
      this.showPlan(result.plan, result.folder, 'qe_input');
    } catch (error) {
      toast(error.message, 'bad', 7000);
    }
  }

  renderBasis(info) {
    const state = storage(`basis:${this.session.root}`) || {
      pseudo_dir: '.',
      out: 'BASIS_PS',
      preset: 'standard',
    };
    const command = h('code', { class: 'grow' });
    let current = null;
    const refresh = debounce(async () => {
      storage(`basis:${this.session.root}`, state);
      try {
        current = await api.post('/api/gen/basis/command', { ...state, workdir: this.workdir });
        command.textContent = current.command;
      } catch (error) {
        command.textContent = error.message;
        current = null;
      }
    }, 200);
    const input = (key, placeholder) => {
      const node = h('input', {
        type: 'text',
        value: state[key] ?? '',
        placeholder,
        oninput: () => {
          state[key] = node.value.trim();
          refresh();
        },
      });
      return node;
    };
    const field = (label, control, hint) =>
      h(
        'div',
        { class: 'field' },
        h(
          'div',
          {},
          h('label', { class: 'name' }, label),
          hint ? h('div', { class: 'hint' }, hint) : null,
        ),
        h('div', {}, h('div', { class: 'control' }, control)),
      );
    const preset = h(
      'div',
      { class: 'segmented' },
      ['minimal', 'standard', 'extended'].map((p) =>
        h(
          'button',
          {
            'aria-pressed': String(state.preset === p),
            onclick: (event) => {
              state.preset = p;
              for (const b of event.currentTarget.parentNode.children)
                b.setAttribute('aria-pressed', String(b === event.currentTarget));
              refresh();
            },
          },
          p,
        ),
      ),
    );
    const noOverwrite = h('input', {
      type: 'checkbox',
      checked: Boolean(state.no_overwrite),
      onchange: () => {
        state.no_overwrite = noOverwrite.checked;
        refresh();
      },
    });
    const upfs = (this.detection.upfs || []).join(', ');
    this.center.append(
      h(
        'div',
        { class: 'form-head' },
        h(
          'div',
          { class: 'grow' },
          h('h2', {}, info.title),
          h('div', { class: 'muted' }, info.description),
        ),
      ),
      h(
        'div',
        { class: 'section' },
        h('h3', {}, 'Pseudopotentials'),
        field(
          'Pseudopotential folder',
          this.withBrowse(
            input('pseudo_dir', '.'),
            (value) => {
              state.pseudo_dir = value;
              refresh();
            },
            { title: 'Pseudopotential folder' },
          ),
          upfs ? `found here: ${upfs}` : 'relative to the working folder',
        ),
        field(
          'or one UPF file',
          this.withBrowse(
            input('pseudo', ''),
            (value) => {
              state.pseudo = value;
              refresh();
            },
            { title: 'One UPF file', allowFiles: true },
          ),
          'overrides the folder',
        ),
        field('Shells', input('shells', 'from the preset'), 'e.g. 3S,3P,3D (single UPF only)'),
      ),
      h(
        'div',
        { class: 'section' },
        h('h3', {}, 'Basis'),
        field(
          'Preset',
          preset,
          'extended ⊃ standard ⊃ minimal; the optics and NLO runs use extended',
        ),
        field('Output folder', input('out', 'BASIS_PS')),
        field('Box radius (Bohr)', input('r_box', 'min(upf r_max, 10)')),
        field('Mesh points', input('n_points', '2000')),
        field('Keep existing tables', h('label', { class: 'switch' }, noOverwrite, h('span'))),
      ),
    );
    const runButton = h(
      'button',
      {
        class: 'btn primary',
        disabled: this.session.read_only || !this.app.runs,
        onclick: () => {
          if (current)
            this.app.runCommand({
              label: 'Pseudo-atom basis',
              argv: current.argv,
              cwd: current.cwd,
            });
        },
      },
      'Run',
    );
    this.preview.append(
      h('div', { class: 'preview-head' }, h('h3', {}, 'Command')),
      h(
        'div',
        { class: 'preview-body', style: { padding: '14px' } },
        h(
          'div',
          { class: 'steps' },
          h(
            'div',
            { class: 'cmd' },
            command,
            h(
              'button',
              {
                class: 'icon-btn',
                title: 'Copy',
                onclick: () => current && copyText(current.command),
              },
              '⧉',
            ),
          ),
        ),
        h(
          'p',
          { class: 'small muted' },
          'Run it in the working folder; main.py also generates missing tables on its first run.',
        ),
      ),
      h('div', { class: 'preview-foot' }, h('span', { class: 'grow' }), runButton),
    );
    refresh();
  }
}
