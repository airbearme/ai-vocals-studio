const DEFAULT_BUCKET = "voiceovers";

export function supabaseConfig() {
  const url = String(process.env.SUPABASE_URL || "").replace(/\/+$/, "");
  const key = String(process.env.SUPABASE_SERVICE_ROLE_KEY || "");
  return {
    enabled: Boolean(url && key),
    url,
    key,
    bucket: String(process.env.SUPABASE_STORAGE_BUCKET || DEFAULT_BUCKET),
  };
}

export async function supabaseRequest(path, options = {}) {
  const cfg = supabaseConfig();
  if (!cfg.enabled) throw new Error("Supabase is not configured");
  const response = await fetch(`${cfg.url}${path}`, {
    ...options,
    headers: {
      apikey: cfg.key,
      Authorization: `Bearer ${cfg.key}`,
      ...(options.headers || {}),
    },
  });
  if (response.ok) return response;
  const text = await response.text();
  throw new Error(`Supabase ${response.status}: ${text || response.statusText}`);
}

export async function createJob(payload) {
  if (!supabaseConfig().enabled) return null;
  const response = await supabaseRequest("/rest/v1/voiceover_jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Prefer: "return=representation",
    },
    body: JSON.stringify(payload),
  });
  const rows = await response.json();
  return rows?.[0] || null;
}

export async function updateJob(id, patch) {
  if (!id || !supabaseConfig().enabled) return null;
  const response = await supabaseRequest(`/rest/v1/voiceover_jobs?id=eq.${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Prefer: "return=representation",
    },
    body: JSON.stringify(patch),
  });
  const rows = await response.json();
  return rows?.[0] || null;
}

export async function uploadAudio(path, buffer, contentType = "audio/mpeg") {
  const cfg = supabaseConfig();
  if (!cfg.enabled) return null;
  const uploadPath = `/storage/v1/object/${encodeURIComponent(cfg.bucket)}/${path}`;
  await supabaseRequest(uploadPath, {
    method: "POST",
    headers: {
      "Content-Type": contentType,
      "x-upsert": "true",
    },
    body: buffer,
  });
  return `${cfg.url}/storage/v1/object/public/${encodeURIComponent(cfg.bucket)}/${path}`;
}

export async function listJobs(limit = 12) {
  if (!supabaseConfig().enabled) return [];
  const response = await supabaseRequest(
    `/rest/v1/voiceover_jobs?select=id,created_at,status,voice_name,engine,output_url,report,error&order=created_at.desc&limit=${Number(limit) || 12}`,
  );
  return response.json();
}
