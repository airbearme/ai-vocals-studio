create table if not exists public.voiceover_jobs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  status text not null default 'queued',
  voice_name text,
  engine text,
  permission_confirmed boolean not null default false,
  input_files jsonb not null default '[]'::jsonb,
  output_path text,
  output_url text,
  report jsonb not null default '{}'::jsonb,
  error text
);

create index if not exists voiceover_jobs_created_at_idx
  on public.voiceover_jobs (created_at desc);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_voiceover_jobs_updated_at on public.voiceover_jobs;
create trigger set_voiceover_jobs_updated_at
before update on public.voiceover_jobs
for each row execute function public.set_updated_at();

alter table public.voiceover_jobs enable row level security;

drop policy if exists "service role manages voiceover jobs" on public.voiceover_jobs;
create policy "service role manages voiceover jobs"
on public.voiceover_jobs
for all
using (auth.role() = 'service_role')
with check (auth.role() = 'service_role');

insert into storage.buckets (id, name, public)
values ('voiceovers', 'voiceovers', true)
on conflict (id) do nothing;
