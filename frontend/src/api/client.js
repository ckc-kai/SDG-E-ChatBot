const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

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
  return res.json(); // { answer, sources }
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  return res.ok;
}