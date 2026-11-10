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

// The run folder of a results folder: the parent of an output/ folder (PAOFLOW's outputdir),
// the folder itself otherwise.  Paths relative to the root, or absolute.
export function runFolderOf(path) {
  const parts = path.split('/');
  if (parts[parts.length - 1] !== 'output') return path;
  if (parts.length === 1) return '.';
  const parent = parts.slice(0, -1).join('/');
  return parent === '' ? '/' : parent;
}

// Resolves with the chosen folder (or file) or null.  The picker browses the whole computer
// (Home, /) unless the server runs with --fixed-root; a folder chosen outside the folders of the
// session is opened first (/api/fs/open): writable for a working folder (`places: false`), read
// only for the other pickers (pseudopotentials, saves, results, files).
export async function pickFolder(
  start,
  { title = 'Choose the working folder', allowFiles = false, places = true } = {},
) {
  let current = start || '.';
  let outside = false;
  let selectedFile = null;
  const session = window.paoflowApp ? window.paoflowApp.session : { read_roots: [] };
  const writable = places === false;
  const placesBar = h('div', { class: 'places' });
  const crumbs = h('div', { class: 'crumbs' });
  const list = h('div', { class: 'browser-list' });
  const status = h('div', { class: 'small muted', style: { marginTop: '6px' } });

  function placeList() {
    const items = [['root', '.', session.root]];
    if (!session.fixed_root) {
      items.push(['home', session.home, session.home], ['computer', '/', '/']);
      for (const folder of session.opened || []) {
        if (!writable || folder.writable)
          items.push([folder.path.split('/').slice(-2).join('/'), folder.path, folder.path]);
      }
    }
    if (!writable)
      for (const folder of session.read_roots || [])
        items.push([
          `${folder.split('/').slice(-2).join('/')} 🔒`,
          folder,
          `${folder} (read-only)`,
        ]);
    return items;
  }

  async function load(path) {
    let listing;
    try {
      listing = await api.get('/api/fs/list', { path });
    } catch (error) {
      toast(error.message, 'bad');
      return;
    }
    current = listing.path;
    outside = Boolean(listing.place.outside);
    selectedFile = null;
    clear(placesBar);
    const items = placeList();
    if (items.length > 1) {
      placesBar.append(h('span', { class: 'small muted' }, 'Places'));
      for (const [label, target, hint] of items) {
        const active = target === listing.path;
        placesBar.append(
          h(
            'button',
            { class: `chip${active ? ' accent' : ''}`, title: hint, onclick: () => load(target) },
            label,
          ),
        );
      }
    }
    clear(crumbs);
    const place = listing.place;
    if (outside) {
      // An absolute path anywhere on the computer: every folder of it is a step.
      const parts = current.split('/').filter(Boolean);
      crumbs.append(h('button', { onclick: () => load('/') }, '/'));
      parts.forEach((part, index) => {
        const target = `/${parts.slice(0, index + 1).join('/')}`;
        crumbs.append(index ? ' / ' : '', h('button', { onclick: () => load(target) }, part));
      });
      crumbs.append(
        h(
          'span',
          { class: 'chip warn', style: { marginLeft: '8px' }, title: 'outside the Studio folders' },
          writable ? 'opened for writing when chosen' : 'opened for reading when chosen',
        ),
      );
    } else {
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
    }
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
  const chosen = await dialog({
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
  if (!chosen || !outside) return chosen;
  // Outside the folders of the session: open the folder (a file's folder) before using it.
  const folder = allowFiles ? chosen.slice(0, chosen.lastIndexOf('/')) || '/' : chosen;
  try {
    const opened = await api.post('/api/fs/open', { path: folder, writable });
    if (window.paoflowApp) window.paoflowApp.session.opened = opened.opened;
    if (writable && !opened.writable) {
      toast(`${folder} is read-only for this Studio (--read-only)`, 'warn', 6000);
      return null;
    }
    toast(`Opened ${folder}${opened.writable ? '' : ' (read-only)'}`, 'ok');
  } catch (error) {
    toast(error.message, 'bad', 7000);
    return null;
  }
  return chosen;
}
