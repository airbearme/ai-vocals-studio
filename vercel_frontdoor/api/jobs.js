import { listJobs, storageConfig } from "./_supabase.js";

function sendJson(res, status, payload) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return sendJson(res, 405, { ok: false, error: "Method not allowed" });
  }
  try {
    const cfg = storageConfig();
    try {
      const jobs = await listJobs(12);
      return sendJson(res, 200, { ok: true, configured: true, provider: cfg.provider, jobs });
    } catch (error) {
      return sendJson(res, 200, {
        ok: true,
        configured: true,
        provider: cfg.provider,
        jobs: [],
        warning: error.message || String(error),
      });
    }
  } catch (error) {
    return sendJson(res, 500, { ok: false, error: error.message || String(error) });
  }
}
