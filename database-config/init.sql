CREATE TABLE public.users(
    id uuid not null default gen_random_uuid (),
    email text not null,
    password text null,
    created_at timestamp without time zone null default now(),
    constraint users_pkey primary key (id),
    constraint users_email_key unique (email)
) TABLESPACE pg_default;