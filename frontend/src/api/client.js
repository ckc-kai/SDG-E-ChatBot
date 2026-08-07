const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
  || `http://${window.location.hostname}:8000`;

export async function askQuestion(question) {
  const res = await fetch(`${API_BASE_URL}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || errorBody.error || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  return res.ok;
}

export async function warmupModels() {
  const res = await fetch(`${API_BASE_URL}/api/warmup`, { method: 'POST' });
  if (!res.ok) {
    throw new Error(`Model warmup failed: ${res.status}`);
  }
  const body = await res.json();
  return body.models_ready === true;
}
