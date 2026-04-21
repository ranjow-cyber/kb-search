-- ============================================================
--  KNOWLEDGE BASE SEARCH — SQLite Schema
-- ============================================================

CREATE TABLE IF NOT EXISTS kb_categories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id  INTEGER REFERENCES kb_categories(id),
    name       TEXT NOT NULL,
    slug       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kb_articles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id  INTEGER REFERENCES kb_categories(id),
    title        TEXT NOT NULL,
    slug         TEXT NOT NULL UNIQUE,
    summary      TEXT,
    content      TEXT NOT NULL,
    author       TEXT,
    is_published INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kb_attachments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id     INTEGER NOT NULL REFERENCES kb_articles(id) ON DELETE CASCADE,
    file_name      TEXT NOT NULL,
    file_type      TEXT NOT NULL,
    file_path      TEXT NOT NULL,
    file_size_kb   INTEGER,
    extracted_text TEXT,
    is_indexed     INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Embeddingi przechowywane jako JSON (tak samo jak w wersji SQL Server)
CREATE TABLE IF NOT EXISTS kb_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL,   -- 'article' | 'attachment_chunk'
    source_id       INTEGER NOT NULL,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    chunk_text      TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_json  TEXT NOT NULL,   -- JSON array floatów
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_emb_source ON kb_embeddings(source_type, source_id);

CREATE TABLE IF NOT EXISTS kb_search_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    query          TEXT NOT NULL,
    search_type    TEXT NOT NULL,
    results_count  INTEGER,
    top_article_id INTEGER REFERENCES kb_articles(id),
    duration_ms    INTEGER,
    searched_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kb_tags (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS kb_article_tags (
    article_id INTEGER NOT NULL REFERENCES kb_articles(id) ON DELETE CASCADE,
    tag_id     INTEGER NOT NULL REFERENCES kb_tags(id)     ON DELETE CASCADE,
    PRIMARY KEY (article_id, tag_id)
);

-- FTS5 — wbudowane w SQLite pełnotekstowe przeszukiwanie
CREATE VIRTUAL TABLE IF NOT EXISTS kb_articles_fts USING fts5(
    title,
    content,
    summary,
    content='kb_articles',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_attachments_fts USING fts5(
    file_name,
    extracted_text,
    content='kb_attachments',
    content_rowid='id',
    tokenize='unicode61'
);

-- Triggery do automatycznej synchronizacji FTS
CREATE TRIGGER IF NOT EXISTS kb_articles_fts_insert AFTER INSERT ON kb_articles BEGIN
    INSERT INTO kb_articles_fts(rowid, title, content, summary)
    VALUES (new.id, new.title, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS kb_articles_fts_update AFTER UPDATE ON kb_articles BEGIN
    INSERT INTO kb_articles_fts(kb_articles_fts, rowid, title, content, summary)
    VALUES ('delete', old.id, old.title, old.content, old.summary);
    INSERT INTO kb_articles_fts(rowid, title, content, summary)
    VALUES (new.id, new.title, new.content, new.summary);
END;

CREATE TRIGGER IF NOT EXISTS kb_articles_fts_delete AFTER DELETE ON kb_articles BEGIN
    INSERT INTO kb_articles_fts(kb_articles_fts, rowid, title, content, summary)
    VALUES ('delete', old.id, old.title, old.content, old.summary);
END;

CREATE TRIGGER IF NOT EXISTS kb_attachments_fts_insert AFTER INSERT ON kb_attachments BEGIN
    INSERT INTO kb_attachments_fts(rowid, file_name, extracted_text)
    VALUES (new.id, new.file_name, new.extracted_text);
END;

CREATE TRIGGER IF NOT EXISTS kb_attachments_fts_update AFTER UPDATE ON kb_attachments BEGIN
    INSERT INTO kb_attachments_fts(kb_attachments_fts, rowid, file_name, extracted_text)
    VALUES ('delete', old.id, old.file_name, old.extracted_text);
    INSERT INTO kb_attachments_fts(rowid, file_name, extracted_text)
    VALUES (new.id, new.file_name, new.extracted_text);
END;
