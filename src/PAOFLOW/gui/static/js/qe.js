// Quantum ESPRESSO build of the runs: the installations the server found, and the user's choice.
//
// The server accepts only the bin folders of its own list (paoflow-studio --qe-bin adds one). The
// choice is remembered per GUI root; every picker on the page follows the last change.

import { api } from './api.js';
import { h, storage, toast } from './dom.js';

let pending = null;

export function loadQe(rescan = false) {
  if (!pending || rescan) {
    pending = api.get('/api/qe', rescan ? { rescan: 1 } : {}).catch((error) => {
      pending = null;
      throw error;
    });
  }
  return pending;
}

function storageKey(session) {
  return `qe-bin:${session.root}`;
}

// The chosen bin folder: the remembered one while the server still lists it, else the default
// (the build on the PATH of the server, else the newest).
export async function chosenQe(session) {
  const { installations, default: fallback } = await loadQe();
  const stored = storage(storageKey(session));
  return installations.some((item) => item.bin === stored) ? stored : fallback;
}

export function qePicker(session) {
  const select = h('select', {
    class: 'grow qe-select',
    title: 'Quantum ESPRESSO build that runs pw.x, ph.x, ... and the QE scripts',
    onchange: () => {
      storage(storageKey(session), select.value);
      document.dispatchEvent(new CustomEvent('qe-change', { detail: select.value }));
    },
  });
  const box = h('div', { class: 'qe-picker' });
  const fill = async (rescan = false) => {
    let result;
    let current;
    try {
      result = await loadQe(rescan);
      current = await chosenQe(session);
    } catch (error) {
      box.replaceChildren(h('div', { class: 'small muted' }, error.message));
      return;
    }
    if (!result.installations.length) {
      box.replaceChildren(
        h(
          'div',
          { class: 'callout warn' },
          'No Quantum ESPRESSO found: put pw.x on the PATH of the shell that starts ' +
            'paoflow-studio, or start it with --qe-bin DIR.',
        ),
      );
      return;
    }
    select.replaceChildren(
      ...result.installations.map((item) =>
        h(
          'option',
          { value: item.bin, selected: item.bin === current, title: item.bin },
          item.label,
        ),
      ),
    );
    box.replaceChildren(
      h('label', { class: 'small muted' }, 'Quantum ESPRESSO'),
      h(
        'div',
        { class: 'row' },
        select,
        h(
          'button',
          {
            class: 'icon-btn',
            type: 'button',
            title: 'Search the machine again',
            onclick: async () => {
              await fill(true);
              toast(`${result.installations.length} Quantum ESPRESSO build(s) found`, 'ok');
            },
          },
          '↻',
        ),
      ),
    );
  };
  const follow = (event) => {
    if (!box.isConnected) {
      document.removeEventListener('qe-change', follow);
      return;
    }
    if (select.value !== event.detail) select.value = event.detail;
  };
  document.addEventListener('qe-change', follow);
  fill();
  return box;
}
