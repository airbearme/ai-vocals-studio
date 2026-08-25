import { createJob, uploadObject, supabaseConfig } from "./_supabase.js";
import { randomUUID } from "node:crypto";

export const config = {
  api: { bodyParser: { sizeLimit: "20mb" } },
  maxDuration: 60,
};

function sendJson(res, status, payload) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

function cleanName(value) {
  return String(value || "Authorized Voice").replace(/[^A-Za-z0-9_. -]+/g, "").trim().slice(0, 80) || "Authorized Voice";
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendJson(res, 405, { ok: false, error: "Method not allowed" });
  }

  try {
    if (!supabaseConfig().enabled) {
      return sendJson(res, 400, { ok: false, error: "Supabase must be configured before worker jobs can be queued." });
    }
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    const files = Array.isArray(body.files) ? body.files : [];
    const text = String(body.text || "").trim();
    if (!body.permission) return sendJson(res, 400, { ok: false, error: "Permission confirmation is required." });
    if (!files.length) return sendJson(res, 400, { ok: false, error: "Upload at least one voice sample." });
    if (!text) return sendJson(res, 400, { ok: false, error: "Voice-over text is required." });

    const totalBytes = files.reduce((sum, file) => sum + Number(file.size || 0), 0);
    if (totalBytes > 14 * 1024 * 1024) {
      return sendJson(res, 413, { ok: false, error: "Uploaded samples are too large for Vercel serverless. Use shorter clips or run the local app." });
    }

    const jobPrefix = `queued/${randomUUID()}`;
    const storedFiles = [];
    for (const file of files.slice(0, 6)) {
      if (!file || !file.data) continue;
      const safeName = cleanName(file.name || "voice-sample.wav").replace(/\s+/g, "_");
      const path = `${jobPrefix}/voices/${Date.now()}-${safeName}`;
      const buffer = Buffer.from(String(file.data), "base64");
      const publicUrl = await uploadObject(path, buffer, file.type || "application/octet-stream");
      storedFiles.push({
        name: safeName,
        type: file.type || "application/octet-stream",
        size: Number(file.size || buffer.length),
        path,
        url: publicUrl,
      });
    }

    const job = await createJob({
      status: "queued",
      job_type: "local_worker_voiceover",
      voice_name: cleanName(body.voiceName),
      engine: "local best available",
      permission_confirmed: true,
      input_files: storedFiles,
      text,
      mode: "bed",
      settings: {
        mood: body.mood || "default",
        stability: Number(body.stability || 0.55),
        similarity: Number(body.similarity || 0.9),
      },
      report: {
        queuedBy: "vercel",
        totalSampleMb: Math.round((totalBytes / (1024 * 1024)) * 100) / 100,
      },
    });

    return sendJson(res, 200, { ok: true, job, files: storedFiles });
  } catch (error) {
    return sendJson(res, 500, { ok: false, error: error.message || String(error) });
  }
}
