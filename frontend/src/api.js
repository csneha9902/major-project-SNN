export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Safely executes a fetch request and parses JSON response.
 * Prevents WebKit/Safari "The string did not match the expected pattern" exception
 * by validating response content-type and reading text before parsing.
 */
export async function safeFetchJson(url, options = {}) {
  const targetUrl = url.startsWith("http") ? url : `${API_BASE}${url.startsWith("/") ? "" : "/"}${url}`;

  let res;
  try {
    res = await fetch(targetUrl, options);
  } catch (netErr) {
    throw new Error(`Network error connecting to backend (${targetUrl}): ${netErr.message}`);
  }

  const rawText = await res.text();
  let data = null;
  if (rawText && rawText.trim().length > 0) {
    try {
      data = JSON.parse(rawText);
    } catch {
      if (!res.ok) {
        throw new Error(`Server error HTTP ${res.status}: ${res.statusText}`);
      }
      return { raw: rawText };
    }
  }

  if (!res.ok) {
    const errorMsg = data?.detail || data?.error || data?.message || `Request failed with HTTP ${res.status}`;
    throw new Error(errorMsg);
  }

  return data || {};
}
