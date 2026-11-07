// Small DOM helpers.  Text is always inserted as text nodes (never as HTML).

export function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'value' || key === 'checked' || key === 'selected' || key === 'disabled' || key === 'hidden') {
      node[key] = value;
    } else node.setAttribute(key, value === true ? '' : String(value));
  }
  append(node, children);
  return node;
}

export function append(node, children) {
  for (const child of [children].flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function toast(message, kind = 'info', timeout = 4500) {
  const box = document.getElementById('toasts');
  const item = h('div', { class: `toast ${kind}`, role: 'status' }, message);
  box.append(item);
  if (timeout) setTimeout(() => item.remove(), timeout);
  return item;
}

export function debounce(fn, wait = 250) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied to the clipboard', 'ok', 1800);
  } catch {
    const area = h('textarea', { value: text, style: { position: 'fixed', opacity: '0' } });
    document.body.append(area);
    area.select();
    document.execCommand('copy');
    area.remove();
    toast('Copied to the clipboard', 'ok', 1800);
  }
}

// Modal dialog: resolves with the value of the clicked action (or null when dismissed).
export function dialog({ title, body, actions = [{ label: 'Close', value: null }], wide = false }) {
  return new Promise((resolve) => {
    const foot = h('div', { class: 'dialog-foot' });
    const box = h(
      'dialog',
      { style: wide ? { width: 'min(1100px, 95vw)' } : null },
      h('div', { class: 'dialog-head' }, title),
      h('div', { class: 'dialog-body' }, body),
      foot,
    );
    for (const action of actions) {
      foot.append(
        h(
          'button',
          {
            class: `btn ${action.kind || ''}`,
            onclick: () => {
              const value = typeof action.value === 'function' ? action.value() : action.value;
              box.close();
              resolve(value);
            },
          },
          action.label,
        ),
      );
    }
    box.addEventListener('cancel', () => resolve(null));
    box.addEventListener('close', () => box.remove());
    document.body.append(box);
    box.showModal();
  });
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['kB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[index]}`;
}

export function formatTime(seconds) {
  return new Date(seconds * 1000).toLocaleString();
}

export function storage(key, value) {
  const name = `paoflow-studio:${key}`;
  if (value === undefined) {
    try {
      return JSON.parse(localStorage.getItem(name));
    } catch {
      return null;
    }
  }
  localStorage.setItem(name, JSON.stringify(value));
  return value;
}
