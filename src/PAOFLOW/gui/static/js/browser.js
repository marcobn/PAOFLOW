// Folder picker over the sandboxed file listing of the server.

import { api } from './api.js';
import { clear, dialog, formatBytes, h, toast } from './dom.js';

const TAG_LABELS = {
  'qe-save': 'QE save',
  upf: 'UPF',
  'pw-input': 'pw.x input',
  driver: 'driver',
  'elphon-system': 'e-ph system',
  'paoflow-output': 'outputs',
  'dynamical-matrices': 'ph.x dyn',
};

export function tagChips(tags) {
  return (tags || []).map((tag) => h('span', { class: `chip ${tag === 'qe-save' || tag === 'elphon-system' ? 'accent' : ''}` }, TAG_LABELS[tag] || tag));
}

// Resolves with the chosen folder (relative to the root) or null.
export async function pickFolder(start, { title = 'Choose the working folder', allowFiles = false } = {}) {
  let current = start || '.';
  let selectedFile = null;
  const crumbs = h('div', { class: 'crumbs' });
  const list = h('div', { class: 'browser-list' });
  const status = h('div', { class: 'small muted', style: { marginTop: '6px' } });

  async function load(path) {
    let listing;
    try {
      listing = await api.get('/api/fs/list', { path });
    } catch (error) {
      toast(error.message, 'bad');
      return;
    }
    current = listing.path;
    selectedFile = null;
    clear(crumbs);
    const parts = current === '.' ? [] : current.split('/');
    crumbs.append(h('button', { onclick: () => load('.') }, 'root'));
    parts.forEach((part, index) => {
      crumbs.append(' / ', h('button', { onclick: () => load(parts.slice(0, index + 1).join('/')) }, part));
    });
    clear(list);
    if (listing.parent !== null) {
      list.append(
        h('div', { class: 'browser-entry dir', onclick: () => load(listing.parent) }, h('span', { class: 'icon' }, '↰'), h('span', { class: 'muted' }, '..')),
      );
    }
    for (const entry of listing.entries) {
      const path = current === '.' ? entry.name : `${current}/${entry.name}`;
      const isDir = entry.kind === 'dir';
      const row = h(
        'div',
        {
          class: `browser-entry ${isDir ? 'dir' : 'file'}`,
          onclick: () => {
            if (isDir) load(path);
            else if (allowFiles) {
              selectedFile = path;
              for (const node of list.children) node.style.background = '';
              row.style.background = 'var(--accent-soft)';
            }
          },
        },
        h('span', { class: 'icon' }, isDir ? '▸' : '·'),
        h('span', { class: 'grow ellipsis' }, entry.name),
        tagChips(entry.tags),
        isDir ? null : h('span', { class: 'faint small' }, formatBytes(entry.size)),
      );
      list.append(row);
    }
    if (!listing.entries.length) list.append(h('div', { class: 'empty' }, 'empty folder'));
    status.textContent = listing.absolute;
  }

  await load(current);
  return dialog({
    title,
    body: h('div', {}, crumbs, list, status),
    actions: [
      { label: 'Cancel', value: null },
      { label: allowFiles ? 'Use the selected file' : 'Use this folder', kind: 'primary', value: () => (allowFiles ? selectedFile : current) },
    ],
  });
}
