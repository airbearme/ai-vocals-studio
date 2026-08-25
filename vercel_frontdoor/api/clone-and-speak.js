import { createJob, updateJob, uploadAudio, supabaseConfig } from "./_supabase.js";
import { randomUUID } from "node:crypto";

const ELEVEN_API = "https://api.elevenlabs.io/v1";

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

async function elevenFetch(path, options) {
  const response = await fetch(`${ELEVEN_API}${path}`, options);
  if (response.ok) return response;
  let detail = "";
  try {
    detail = JSON.stringify(await response.json());
  } catch {
    detail = await response.text();
  }
  throw new Error(`ElevenLabs ${response.status}: ${detail || response.statusText}`);
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return sendJson(res, 405, { ok: false, error: "Method not allowed" });
  }
  let job = null;
  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    const apiKey = String(body.apiKey || process.env.ELEVENLABS_API_KEY || "").trim();
    const files = Array.isArray(body.files) ? body.files : [];
    const text = String(body.text || "").trim();
    if (!apiKey) return sendJson(res, 400, { ok: false, error: "Set ELEVENLABS_API_KEY on Vercel or enter an API key in the form." });
    if (!body.permission) return sendJson(res, 400, { ok: false, error: "Permission confirmation is required." });
    if (!files.length) return sendJson(res, 400, { ok: false, error: "Upload at least one voice sample." });
    if (!text) return sendJson(res, 400, { ok: false, error: "Voice-over text is required." });
    const totalBytes = files.reduce((sum, file) => sum + Number(file.size || 0), 0);
    if (totalBytes > 14 * 1024 * 1024) {
      return sendJson(res, 413, { ok: false, error: "Uploaded samples are too large for Vercel serverless. Use shorter clips or the Docker/local app." });
    }
    job = await createJob({
      status: "running",
      voice_name: cleanName(body.voiceName),
      engine: "ElevenLabs IVC",
      permission_confirmed: true,
      input_files: files.map((file) => ({
        name: cleanName(file.name || "voice-sample"),
        type: file.type || "application/octet-stream",
        size: Number(file.size || 0),
      })),
      report: {
        totalSampleMb: Math.round((totalBytes / (1024 * 1024)) * 100) / 100,
        supabaseConfigured: supabaseConfig().enabled,
      },
    });

    const voiceForm = new FormData();
    voiceForm.append("name", cleanName(body.voiceName));
    voiceForm.append("description", "Authorized instant voice clone generated from AI Vocals Studio Vercel app.");
    voiceForm.append("remove_background_noise", "true");
    for (const file of files.slice(0, 4)) {
      if (!file || !file.data) continue;
      const blob = new Blob([Buffer.from(String(file.data), "base64")], { type: file.type || "application/octet-stream" });
      voiceForm.append("files[]", blob, cleanName(file.name || "voice-sample.wav"));
    }

    const voiceResponse = await elevenFetch("/voices/add", {
      method: "POST",
      headers: { "xi-api-key": apiKey },
      body: voiceForm,
    });
    const voice = await voiceResponse.json();
    const voiceId = voice.voice_id;
    if (!voiceId) throw new Error("ElevenLabs did not return a voice_id.");

    const ttsResponse = await elevenFetch(`/text-to-speech/${encodeURIComponent(voiceId)}?output_format=mp3_44100_128`, {
      method: "POST",
      headers: { "xi-api-key": apiKey, "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model_id: "eleven_multilingual_v2",
        voice_settings: {
          stability: Number.isFinite(Number(body.stability)) ? Number(body.stability) : 0.55,
          similarity_boost: Number.isFinite(Number(body.similarity)) ? Number(body.similarity) : 0.9,
          style: 0.25,
          use_speaker_boost: true,
        },
      }),
    });

    const audioBuffer = Buffer.from(await ttsResponse.arrayBuffer());
    const outputPath = `${job?.id || randomUUID()}/voiceover.mp3`;
    const outputUrl = await uploadAudio(outputPath, audioBuffer, "audio/mpeg");
    let deleted = false;
    if (body.deleteVoice) {
      try {
        await elevenFetch(`/voices/${encodeURIComponent(voiceId)}`, { method: "DELETE", headers: { "xi-api-key": apiKey } });
        deleted = true;
      } catch {
        deleted = false;
      }
    }
    await updateJob(job?.id, {
      status: "complete",
      output_path: outputUrl ? outputPath : null,
      output_url: outputUrl,
      report: {
        voiceId,
        deleted,
        requiresVerification: Boolean(voice.requires_verification),
        sampleCount: files.length,
        totalSampleMb: Math.round((totalBytes / (1024 * 1024)) * 100) / 100,
        storage: outputUrl ? "supabase" : "inline",
      },
    });

    return sendJson(res, 200, {
      ok: true,
      jobId: job?.id || null,
      engine: "ElevenLabs IVC",
      voiceId,
      requiresVerification: Boolean(voice.requires_verification),
      deleted,
      mediaType: "audio/mpeg",
      audioBase64: audioBuffer.toString("base64"),
      outputUrl,
      report: {
        sampleCount: files.length,
        totalSampleMb: Math.round((totalBytes / (1024 * 1024)) * 100) / 100,
        notes: [
          "Hosted Vercel path uses ElevenLabs Instant Voice Cloning and text-to-speech.",
          "For local Qwen/XTTS/RVC/Demucs workflows, run the Docker/local app.",
        ],
      },
    });
  } catch (error) {
    try {
      if (job?.id) {
        await updateJob(job.id, { status: "failed", error: error.message || String(error) });
      }
    } catch {}
    return sendJson(res, 500, { ok: false, error: error.message || String(error) });
  }
}
