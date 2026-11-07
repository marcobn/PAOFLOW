// JSON client of the GUI server (the session token travels in the HttpOnly cookie).

export class ApiError extends Error {
  constructor(status, message, details) {
    super(message);
    this.status = status;
    this.details = details || {};
  }
}

function withQuery(path, params) {
  if (!params) return path;
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) query.set(key, value);
  }
  const text = query.toString();
  return text ? `${path}?${text}` : path;
}

async function request(method, path, body) {
  const options = { method, credentials: 'same-origin', headers: {} };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: text };
  }
  if (!response.ok) {
    throw new ApiError(response.status, (data && data.error) || response.statusText, data);
  }
  return data;
}

export const api = {
  get: (path, params) => request('GET', withQuery(path, params)),
  post: (path, body) => request('POST', path, body ?? {}),
};
