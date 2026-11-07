// Read-only code view with line numbers, light syntax colouring and unified diffs.

import { h } from './dom.js';

const PYTHON_KEYWORDS = new Set(
  (
    'False None True and as assert async await break class continue def del elif else except ' +
    'finally for from global if import in is lambda nonlocal not or pass raise return try while ' +
    'with yield'
  ).split(' '),
);

function language(path) {
  if (/\.py$/.test(path)) return 'python';
  if (/\.sh$/.test(path)) return 'shell';
  if (/\.(in|kpoints|fmt|dat)$/.test(path)) return 'namelist';
  return 'plain';
}

// Tokens of one line: [kind, text] pairs (kind '' for plain text).
function tokenize(line, lang, state) {
  const tokens = [];
  let rest = line;
  const push = (kind, text) => tokens.push([kind, text]);
  if (state.triple) {
    const end = rest.indexOf(state.triple);
    if (end < 0) {
      push('str', rest);
      return tokens;
    }
    push('str', rest.slice(0, end + 3));
    rest = rest.slice(end + 3);
    state.triple = null;
  }
  const comment = lang === 'namelist' ? /^[!#]/ : /^#/;
  while (rest.length) {
    let match;
    if (comment.test(rest) && lang !== 'plain') {
      push('com', rest);
      break;
    }
    if (lang === 'python' && (match = /^("""|''')/.exec(rest))) {
      const end = rest.indexOf(match[1], 3);
      if (end < 0) {
        push('str', rest);
        state.triple = match[1];
        break;
      }
      push('str', rest.slice(0, end + 3));
      rest = rest.slice(end + 3);
      continue;
    }
    if ((match = /^(?:[rbfu]{0,2})("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/i.exec(rest)) && lang !== 'plain') {
      push('str', match[0]);
      rest = rest.slice(match[0].length);
      continue;
    }
    if ((match = /^\d+(?:\.\d*)?(?:[eEdD][+-]?\d+)?/.exec(rest)) && lang !== 'plain') {
      push('num', match[0]);
      rest = rest.slice(match[0].length);
      continue;
    }
    if ((match = /^[A-Za-z_][A-Za-z_0-9]*/.exec(rest))) {
      const word = match[0];
      if (lang === 'python' && PYTHON_KEYWORDS.has(word)) push('key', word);
      else if (lang === 'python' && /^[A-Z][A-Z_0-9]{2,}$/.test(word)) push('def', word);
      else push('', word);
      rest = rest.slice(word.length);
      continue;
    }
    if (lang === 'namelist' && (match = /^&\w+/.exec(rest))) {
      push('key', match[0]);
      rest = rest.slice(match[0].length);
      continue;
    }
    push('', rest[0]);
    rest = rest.slice(1);
  }
  return tokens;
}

export function codeView(text, path = '') {
  const lang = language(path);
  const state = { triple: null };
  const block = h('pre', { class: 'code' });
  const lines = text.endsWith('\n') ? text.slice(0, -1).split('\n') : text.split('\n');
  for (const line of lines) {
    const node = h('span', { class: 'line' });
    for (const [kind, piece] of tokenize(line, lang, state)) {
      node.append(kind ? h('span', { class: `tok-${kind}` }, piece) : piece);
    }
    block.append(node);
  }
  return block;
}

export function diffView(diff) {
  const block = h('pre', { class: 'code diff' });
  for (const line of diff.replace(/\n$/, '').split('\n')) {
    let kind = '';
    if (line.startsWith('@@')) kind = 'hunk';
    else if (line.startsWith('+') && !line.startsWith('+++')) kind = 'add';
    else if (line.startsWith('-') && !line.startsWith('---')) kind = 'del';
    block.append(h('span', { class: `line ${kind}` }, line));
  }
  return block;
}
