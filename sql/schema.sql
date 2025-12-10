-- Habilita la extensión para trabajar con vectores (Embeddings)
-- Esto es crucial para cumplir el requisito de "Uso de modelos IA" de la vacante.
create extension if not exists vector;

create table embeddings_log (
  id bigint primary key generated always as identity,
  timestamp timestamptz default now(),
  content text,                    -- El texto original (ej. la pregunta del usuario)
  embedding vector(1536),          -- El vector matemático (1536 es el tamaño estándar de OpenAI)
  type text                        -- 'query' (pregunta usuario) o 'doc' (documento del sistema)
);

create table embeddings_log (
  id bigint primary key generated always as identity,
  timestamp timestamptz default now(),
  content text,                    -- El texto original (ej. la pregunta del usuario)
  embedding vector(1536),          -- El vector matemático (1536 es el tamaño estándar de OpenAI)
  type text                        -- 'query' (pregunta usuario) o 'doc' (documento del sistema)
);