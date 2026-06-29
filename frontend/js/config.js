// API base URL — shared by all pages
// Production: same-origin; vercel.json rewrites /api/* → Railway
const RAILWAY_API = 'https://dairen-liquid-codex-production.up.railway.app';

function getApiBase() {
  const h = window.location.hostname;
  if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8000';
  return '';
}

function getStreamApiBase() {
  const h = window.location.hostname;
  if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8000';
  // Long SSE (R1 + Perplexity) bypasses Vercel proxy ~60s idle timeout
  return RAILWAY_API;
}

const API_BASE = getApiBase();
const STREAM_API_BASE = getStreamApiBase();
