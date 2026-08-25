# Supabase setup for the Vercel app

The Vercel deployment works without Supabase, but Supabase enables persistent
job history and stored output audio.

## 1. Create Supabase project

Create a Supabase project, then open the SQL editor and run:

```sql
-- supabase/schema.sql
```

The schema creates:

- `public.voiceover_jobs`
- Public Storage bucket `voiceovers`
- Service-role-only row policy for job writes

## 2. Vercel environment variables

Set these on the Vercel project:

```bash
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SUPABASE_STORAGE_BUCKET=voiceovers
ELEVENLABS_API_KEY=YOUR_ELEVENLABS_KEY
```

`SUPABASE_SERVICE_ROLE_KEY` must stay server-side only. Do not expose it in the browser.

## 3. Deploy

```bash
cd vercel_frontdoor
npx vercel --prod --yes --scope stephens-projects-8fbc16d0
```

The hosted app will then save completed output MP3s to Supabase Storage and list
recent jobs from `voiceover_jobs`.
