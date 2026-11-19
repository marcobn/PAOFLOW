// Form of a generator workflow, rendered from the server's resolved questions.
//
// Fields are created once per workflow and updated in place on every resolve (visibility,
// defaults, choices, errors), so typing never loses focus.  `answers` holds only what the user
// changed: a missing key means "the default", exactly like a blank answer on the terminal.

import { append, clear, h } from './dom.js';

function sameList(a, b) {
  return JSON.stringify(a || []) === JSON.stringify(b || []);
}

class Field {
  constructor(form, item) {
    this.form = form;
    this.key = item.key;
    this.kind = item.kind;
    this.level = item.level;
    this.choices = null;
    this.error = h('div', { class: 'error', hidden: true });
    this.warning = h('div', { class: 'field-warning', hidden: true });
    this.control = h('div', { class: 'control' });
    this.resetButton = h(
      'button',
      {
        class: 'icon-btn reset',
        title: 'Back to the default',
        'aria-label': 'Reset to default',
        onclick: () => this.reset(),
      },
      '↺',
    );
    const wide = ['menu', 'names', 'rows'].includes(item.kind);
    const label = h(
      'label',
      { class: 'name', for: `field-${item.key}`, title: item.prompt.trim() },
      item.label,
      item.level === 'advanced'
        ? h('span', { class: 'badge-advanced' }, item.gui_only ? 'advanced · GUI' : 'advanced')
        : null,
    );
    const hints = [];
    if (item.hint) hints.push(h('div', { class: 'hint' }, item.hint));
    if (item.help) hints.push(h('div', { class: 'hint' }, item.help));
    if (item.doc) {
      hints.push(
        h(
          'details',
          { class: 'doc' },
          h('summary', {}, `docs · ${item.doc.source}`),
          h('div', { class: 'hint' }, item.doc.text),
        ),
      );
    }
    this.root = h(
      'div',
      { class: `field${wide ? ' wide' : ''}`, dataset: { key: item.key } },
      h('div', {}, label, hints),
      h('div', {}, this.control),
      this.error,
      this.warning,
    );
    this.build(item);
  }

  get edited() {
    return Object.hasOwn(this.form.answers, this.key);
  }

  set(value, immediate = false) {
    this.form.answers[this.key] = value;
    this.root.classList.add('edited');
    this.form.changed(immediate);
  }

  // A value picked from a button (a found save, the folder picker): shown and resolved at once.
  pick(value) {
    if (this.input) this.input.value = value;
    this.set(value, true);
  }

  reset() {
    delete this.form.answers[this.key];
    this.root.classList.remove('edited');
    this.form.changed(true);
  }

  build(item) {
    const kind = item.kind;
    if (kind === 'bool') {
      this.input = h('input', {
        type: 'checkbox',
        id: `field-${item.key}`,
        onchange: () => this.set(this.input.checked),
      });
      this.control.append(h('label', { class: 'switch' }, this.input, h('span')), this.resetButton);
    } else if (kind === 'choice') {
      this.choiceBox = h('div', { class: 'grow' });
      this.control.append(this.choiceBox, this.resetButton);
    } else if (kind === 'menu' || kind === 'names') {
      this.grid = h('div', { class: 'choice-grid grow' });
      this.control.append(this.grid, this.resetButton);
    } else if (kind === 'rows') {
      this.table = h('table', { class: 'rows' });
      this.control.append(
        h(
          'div',
          { class: 'grow' },
          this.table,
          h('button', { class: 'btn small', onclick: () => this.addRow() }, '+ add row'),
        ),
      );
      this.columns = item.columns.length ? item.columns : ['value'];
    } else {
      this.input = h('input', {
        type: 'text',
        id: `field-${item.key}`,
        spellcheck: 'false',
        autocomplete: 'off',
        oninput: () => this.set(this.input.value),
      });
      const browse =
        item.widget === 'folder' && this.form.browse
          ? h(
              'button',
              {
                class: 'btn small',
                type: 'button',
                onclick: async () => {
                  const chosen = await this.form.browse(item);
                  if (chosen !== null && chosen !== undefined) this.pick(chosen);
                },
              },
              'Browse…',
            )
          : null;
      append(this.control, [
        this.input,
        item.unit ? h('span', { class: 'unit' }, item.unit) : null,
        browse,
        this.resetButton,
      ]);
    }
  }

  renderChoices(item) {
    const choices = item.choices || [];
    if (sameList(choices, this.choices)) return;
    this.choices = choices;
    if (this.kind === 'choice') {
      clear(this.choiceBox);
      const short = choices.length <= 4 && choices.every((c) => String(c).length <= 18);
      if (short) {
        this.segments = choices.map((choice) =>
          h(
            'button',
            { type: 'button', 'aria-pressed': 'false', onclick: () => this.set(choice) },
            String(choice),
          ),
        );
        this.choiceBox.append(h('div', { class: 'segmented' }, this.segments));
        this.select = null;
      } else {
        this.select = h(
          'select',
          { id: `field-${item.key}`, onchange: () => this.set(this.select.value) },
          choices.map((choice) => h('option', { value: String(choice) }, String(choice))),
        );
        this.choiceBox.append(this.select);
        this.segments = null;
      }
    } else if (this.kind === 'menu' || this.kind === 'names') {
      clear(this.grid);
      this.boxes = new Map();
      for (const choice of choices) {
        const [value, label] = Array.isArray(choice) ? choice : [choice, choice];
        const box = h('input', { type: 'checkbox', onchange: () => this.toggle() });
        const card = h('label', { class: 'choice-card' }, box, h('span', {}, label));
        this.boxes.set(value, { box, card });
        this.grid.append(card);
      }
    }
  }

  toggle() {
    const values = [];
    for (const [value, { box }] of this.boxes) if (box.checked) values.push(value);
    this.set(values);
  }

  addRow(cells) {
    const inputs = this.columns.map((column, index) =>
      h('input', {
        type: 'text',
        placeholder: column,
        value: cells ? String(cells[index] ?? '') : '',
        oninput: () => this.collectRows(),
      }),
    );
    const row = h(
      'tr',
      {},
      inputs.map((input) => h('td', {}, input)),
      h(
        'td',
        {},
        h(
          'button',
          {
            class: 'icon-btn',
            title: 'Remove row',
            onclick: () => {
              row.remove();
              this.collectRows();
            },
          },
          '✕',
        ),
      ),
    );
    this.table.append(row);
    if (!cells) inputs[0].focus();
  }

  collectRows() {
    const rows = [];
    for (const row of this.table.querySelectorAll('tr.data, tr')) {
      const inputs = [...row.querySelectorAll('input')];
      if (!inputs.length) continue;
      const cells = inputs.map((input) => input.value);
      if (cells.some((cell) => cell.trim())) rows.push(cells);
    }
    this.set(rows);
  }

  // A non-blocking warning of the server (a Save directory that holds no save) with one button per
  // suggestion (the saves found in the working folder).
  renderWarning(item) {
    clear(this.warning);
    this.warning.hidden = !item.warning;
    if (!item.warning) return;
    const suggestions = (item.suggestions || []).filter((s) => s !== item.value);
    this.warning.append(
      item.warning,
      ...suggestions.map((suggestion) =>
        h(
          'button',
          { class: 'btn small', type: 'button', onclick: () => this.pick(suggestion) },
          `Use ${suggestion}`,
        ),
      ),
    );
  }

  update(item) {
    this.root.hidden =
      !item.visible ||
      (this.level === 'advanced' && !this.form.showAdvanced && !this.edited && !item.error);
    if (!item.visible) return;
    this.renderChoices(item);
    this.root.classList.toggle('edited', this.edited);
    this.root.classList.toggle('invalid', Boolean(item.error));
    this.error.hidden = !item.error;
    this.error.textContent = item.error || '';
    this.renderWarning(item);
    const value = item.value;
    if (this.kind === 'bool') {
      if (!this.edited) this.input.checked = Boolean(value);
    } else if (this.kind === 'choice') {
      const current = this.edited ? this.form.answers[this.key] : value;
      if (this.segments) {
        for (const [index, button] of this.segments.entries()) {
          button.setAttribute('aria-pressed', String(this.choices[index] === current));
        }
      } else if (this.select && document.activeElement !== this.select) {
        this.select.value = String(current);
      }
    } else if (this.kind === 'menu' || this.kind === 'names') {
      const selected = new Set(this.edited ? this.form.answers[this.key] : value || []);
      for (const [choice, { box, card }] of this.boxes) {
        box.checked = selected.has(choice);
        card.classList.toggle('checked', box.checked);
      }
    } else if (this.kind === 'rows') {
      // An untouched table follows its default (the points of a save that appeared meanwhile).
      const shown = JSON.stringify(value || []);
      const stale = !this.edited && shown !== this.rowsShown;
      if (Array.isArray(value) && (stale || !this.table.children.length)) {
        clear(this.table);
        this.table.append(
          h(
            'tr',
            {},
            this.columns.map((column) => h('th', {}, column)),
            h('th'),
          ),
        );
        for (const row of value) this.addRow(row);
        if (!value.length) this.addRow(['', '']);
        this.rowsShown = shown;
      }
    } else if (document.activeElement !== this.input) {
      if (!this.edited) this.input.value = value ?? '';
      this.input.classList.toggle('is-default', !this.edited);
      this.input.placeholder = item.default ?? '';
    }
  }
}

export class FormView {
  constructor(container, { onChange, browse = null }) {
    this.container = container;
    this.onChange = onChange;
    this.browse = browse; // (item) => Promise<path or null>, for fields with widget 'folder'
    this.answers = {};
    this.showAdvanced = false;
    this.fields = new Map();
    this.notes = [];
    this.sections = new Map();
    this.lastResolved = null;
  }

  reset(answers = {}) {
    clear(this.container);
    this.answers = { ...answers };
    this.fields = new Map();
    this.notes = [];
    this.sections = new Map();
    this.lastResolved = null;
  }

  changed(immediate = false) {
    this.onChange(this.answers, immediate);
  }

  section(name) {
    const title = name || 'Settings';
    if (!this.sections.has(title)) {
      const box = h(
        'div',
        { class: 'section', id: `section-${title.replace(/\W+/g, '-')}` },
        h('h3', {}, title),
      );
      this.container.append(box);
      this.sections.set(title, box);
    }
    return this.sections.get(title);
  }

  render(resolved) {
    this.lastResolved = resolved;
    resolved.items.forEach((item, index) => {
      if (item.type === 'note') {
        let note = this.notes[index];
        if (!note) {
          note = h('div', { class: 'callout', style: { margin: '6px 0' } });
          this.notes[index] = note;
          this.section(item.section).append(note);
        }
        note.hidden = !item.visible || !item.text;
        note.textContent = item.text;
        return;
      }
      let field = this.fields.get(item.key);
      if (!field) {
        field = new Field(this, item);
        this.fields.set(item.key, field);
        this.section(item.section).append(field.root);
      }
      field.update(item);
    });
    for (const box of this.sections.values()) {
      const visible = [...box.querySelectorAll('.field, .callout')].some((node) => !node.hidden);
      box.hidden = !visible;
    }
  }

  setShowAdvanced(value) {
    this.showAdvanced = value;
    if (this.lastResolved) this.render(this.lastResolved);
  }

  visibleSections() {
    return [...this.sections.entries()]
      .filter(([, box]) => !box.hidden)
      .map(([name, box]) => [name, box]);
  }
}
