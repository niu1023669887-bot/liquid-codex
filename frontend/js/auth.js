const TOKEN_KEY = 'lc_token';

function authApiBase() {
  if (typeof getApiBase === 'function') return getApiBase();
  const h = window.location.hostname;
  if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8000';
  return (typeof API_BASE !== 'undefined') ? API_BASE : '';
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

async function verifyAuth() {
  const token = getToken();
  if (!token) return false;
  try {
    const res = await fetch(`${authApiBase()}/api/auth/verify?token=${encodeURIComponent(token)}`);
    return res.ok;
  } catch {
    return false;
  }
}

async function requireAuth() {
  const ok = await verifyAuth();
  if (!ok) {
    clearToken();
    window.location.href = '/index.html';
  }
  return ok;
}

async function login(password) {
  const res = await fetch(`${authApiBase()}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  if (!res.ok) throw new Error('Invalid password');
  const data = await res.json();
  setToken(data.token);
  return data.token;
}

function logout() {
  clearToken();
  window.location.href = '/index.html';
}
