// Folder picker over the sandboxed file listing of the server.
//
// Paths are relative to the GUI root, or absolute inside a read-only folder (--allow-read and
// PAOFLOW's own PSEUDOS and BASIS). The picker shows those folders as places.

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
  return (tags || []).map((tag) =>
    h(
      'span',
      { class: `chip ${tag === 'qe-save' || tag === 'elphon-system' ? 'accent' : ''}` },
      TAG_LABELS[tag] || tag,
    ),
  );
}

function join(folder, name) {
  if (folder === '.') return name;
  return folder.endsWith('/') ? `${folder}${name}` : `${folder}/${name}`;
}

// `path` (relative to the root, or absolute) as seen from the root-relative folder `from`:
// root-relative paths become relative to `from`, absolute paths stay absolute.
export function relativeTo(from, path) {
  if (path.startsWith('/')) return path;
  const base = from === '.' ? [] : from.split('/');
  const target = path === '.' ? [] : path.split('/');
  let common = 0;
  while (common < base.length && common < target.length && base[common] === target[common]) {
    common += 1;
  }
  const up = base.slice(common).map(() => '..');
  const relative = [...up, ...target.slice(common)].join('/');
  return relative || '.';
}

// Resolves with the chosen folder (or file) or null. `places: false` keeps the picker inside the
// writable root (the working folder of the generators).
export async function pickFolder(
  start,
  { title = 'Choose the working folder', allowFiles = false, places = true } = {},
) {
  let current = start || '.';
  let selectedFile = null;
  const session = window.paoflowApp ? window.paoflowApp.session : { read_roots: [] };
  const readRoots = places ? session.read_roots || [] : [];
  const placesBar = h('div', { class: 'places' });
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
    clear(placesBar);
    if (readRoots.length) {
      placesBar.append(h('span', { class: 'small muted' }, 'Places'));
      for (const [label, target] of [
        ['root', '.'],
        ...readRoots.map((folder) => [folder, folder]),
      ]) {
        const active = listing.place.path === target;
        placesBar.append(
          h(
            'button',
            {
              class: `chip${active ? ' accent' : ''}`,
              title: target === '.' ? session.root : `${target} (read-only)`,
              onclick: () => load(target),
            },
            label === 'root' ? 'root' : label.split('/').slice(-2).join('/'),
            target === '.' ? null : ' 🔒',
          ),
        );
      }
    }
    clear(crumbs);
    const place = listing.place;
    const inner =
      current === place.path ? '' : current.slice(place.path === '.' ? 0 : place.path.length + 1);
    const parts = inner ? inner.split('/') : [];
    crumbs.append(h('button', { onclick: () => load(place.path) }, place.label));
    parts.forEach((part, index) => {
      const target = join(place.path, parts.slice(0, index + 1).join('/'));
      crumbs.append(' / ', h('button', { onclick: () => load(target) }, part));
    });
    if (!place.writable)
      crumbs.append(h('span', { class: 'chip warn', style: { marginLeft: '8px' } }, 'read-only'));
    clear(list);
    if (listing.parent !== null) {
      list.append(
        h(
          'div',
          { class: 'browser-entry dir', onclick: () => load(listing.parent) },
          h('span', { class: 'icon' }, '↰'),
          h('span', { class: 'muted' }, '..'),
        ),
      );
    }
    for (const entry of listing.entries) {
      const path = join(current, entry.name);
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
    body: h('div', {}, placesBar, crumbs, list, status),
    actions: [
      { label: 'Cancel', value: null },
      {
        label: allowFiles ? 'Use the selected file' : 'Use this folder',
        kind: 'primary',
        value: () => (allowFiles ? selectedFile : current),
      },
    ],
  });
}
