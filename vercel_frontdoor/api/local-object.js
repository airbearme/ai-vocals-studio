import fs from "node:fs/promises";
import { localObjectPath, supabaseConfig } from "./_supabase.js";

function sendJson(res, status, payload) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

function contentType(filePath) {
  const lower = String(filePath).toLowerCase();
  if (lower.endsWith(".mp3")) return "audio/mpeg";
  if (lower.endsWith(".flac")) return "audio/flac";
  if (lower.endsWith(".m4a")) return "audio/mp4";
  if (lower.endsWith(".wav")) return "audio/wav";
  return "application/octet-stream";
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return sendJson(res, 405, { ok: false, error: "Method not allowed" });
  }
  if (supabaseConfig().enabled) {
    return sendJson(res, 404, { ok: false, error: "Local object storage is not active while Supabase is configured." });
  }
  try {
    const objectPath = String(req.query?.path || "");
    if (!objectPath) return sendJson(res, 400, { ok: false, error: "Missing object path." });
    const filePath = localObjectPath(objectPath);
    const data = await fs.readFile(filePath);
    res.statusCode = 200;
    res.setHeader("Content-Type", contentType(filePath));
    res.setHeader("Cache-Control", "no-store");
    res.end(data);
  } catch (error) {
    if (error?.code === "ENOENT") return sendJson(res, 404, { ok: false, error: "Object not found." });
    return sendJson(res, 500, { ok: false, error: error.message || String(error) });
  }
}
