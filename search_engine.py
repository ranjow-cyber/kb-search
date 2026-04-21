# ============================================================
#  search_engine.py — Silnik hybrydowy (SQLite FTS5 + Semantic)
#  Używa fastembed zamiast sentence-transformers (lżejszy, bez torch)
# ============================================================
from __future__ import annotations
import json
import time
import logging
from dataclasses import dataclass, field

import numpy as np
from fastembed import TextEmbedding

from database import get_conn

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # ~130MB, szybki
_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        logger.info("Ładowanie modelu embeddingów (fastembed)...")
        _model = TextEmbedding(model_name=MODEL_NAME)
        logger.info("Model załadowany.")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)


@dataclass
class SearchResult:
    article_id: int
    title: str
    summary: str | None
    slug: str
    category: str | None
    fts_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    matched_chunks: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)


class KnowledgeBaseSearch:

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid", category_id: int | None = None) -> list[SearchResult]:
        t0 = time.perf_counter()
        if mode == "fts":
            results = self._fts_search(query, top_k * 2, category_id)
        elif mode == "semantic":
            results = self._semantic_search(query, top_k * 2, category_id)
        else:
            fts = self._fts_search(query, top_k * 2, category_id)
            sem = self._semantic_search(query, top_k * 2, category_id)
            results = self._rrf(fts, sem)
        results = results[:top_k]
        duration_ms = int((time.perf_counter() - t0) * 1000)
        self._log(query, mode, len(results), results[0].article_id if results else None, duration_ms)
        return results

    def _fts_search(self, query: str, limit: int, category_id: int | None) -> list[SearchResult]:
        conn = get_conn()
        results: dict[int, SearchResult] = {}
        cat_filter = "AND a.category_id = :cat" if category_id else ""
        try:
            rows = conn.execute(f"""
                SELECT a.id, a.title, a.summary, a.slug, c.name AS category, rank AS fts_rank
                FROM kb_articles_fts f
                JOIN kb_articles a ON a.id = f.rowid
                LEFT JOIN kb_categories c ON c.id = a.category_id
                WHERE kb_articles_fts MATCH :q AND a.is_published = 1 {cat_filter}
                ORDER BY rank LIMIT :lim
            """, {"q": self._fts_query(query), "cat": category_id, "lim": limit}).fetchall()
        except Exception as e:
            logger.warning(f"FTS błąd: {e}")
            rows = []
        for r in rows:
            score = max(0.0, min(1.0, abs(float(r["fts_rank"])) / 10.0))
            results[r["id"]] = SearchResult(
                article_id=r["id"], title=r["title"], summary=r["summary"],
                slug=r["slug"], category=r["category"], fts_score=score, source_types=["article"],
            )
        conn.close()
        return sorted(results.values(), key=lambda x: x.fts_score, reverse=True)

    def _semantic_search(self, query: str, limit: int, category_id: int | None) -> list[SearchResult]:
        query_emb = embed_texts([query])[0]
        conn = get_conn()
        cat_filter = "AND a.category_id = :cat" if category_id else ""
        rows = conn.execute(f"""
            SELECT e.source_type, e.source_id, e.chunk_text, e.embedding_json,
                   a.id AS article_id, a.title, a.summary, a.slug, c.name AS category
            FROM kb_embeddings e
            LEFT JOIN kb_articles a ON (
                (e.source_type = 'article' AND a.id = e.source_id)
                OR (e.source_type = 'attachment_chunk'
                    AND a.id = (SELECT article_id FROM kb_attachments WHERE id = e.source_id))
            )
            LEFT JOIN kb_categories c ON c.id = a.category_id
            WHERE a.is_published = 1 {cat_filter}
        """, {"cat": category_id}).fetchall()
        conn.close()
        if not rows:
            return []
        emb_matrix = np.array([json.loads(r["embedding_json"]) for r in rows], dtype=np.float32)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_matrix = emb_matrix / np.where(norms == 0, 1, norms)
        query_emb = query_emb / (np.linalg.norm(query_emb) or 1)
        scores = emb_matrix @ query_emb
        article_map: dict[int, dict] = {}
        for idx, row in enumerate(rows):
            art_id = row["article_id"]
            if art_id is None:
                continue
            score = float(scores[idx])
            if art_id not in article_map or score > article_map[art_id]["score"]:
                article_map[art_id] = {
                    "score": score, "title": row["title"], "summary": row["summary"],
                    "slug": row["slug"], "category": row["category"],
                    "source_type": row["source_type"], "chunks": [],
                }
            if score > 0.3:
                article_map[art_id]["chunks"].append(row["chunk_text"][:300])
        results = [
            SearchResult(
                article_id=aid, title=v["title"], summary=v["summary"],
                slug=v["slug"], category=v["category"], semantic_score=v["score"],
                matched_chunks=v["chunks"][:3], source_types=[v["source_type"]],
            )
            for aid, v in article_map.items()
        ]
        return sorted(results, key=lambda x: x.semantic_score, reverse=True)[:limit]

    def _rrf(self, fts: list[SearchResult], sem: list[SearchResult], k: int = 60) -> list[SearchResult]:
        scores: dict[int, float] = {}
        for rank, r in enumerate(fts):
            scores[r.article_id] = scores.get(r.article_id, 0) + 1 / (k + rank + 1)
        for rank, r in enumerate(sem):
            scores[r.article_id] = scores.get(r.article_id, 0) + 1 / (k + rank + 1)
        merged: dict[int, SearchResult] = {}
        for r in fts + sem:
            if r.article_id not in merged:
                merged[r.article_id] = r
            else:
                e = merged[r.article_id]
                e.fts_score = max(e.fts_score, r.fts_score)
                e.semantic_score = max(e.semantic_score, r.semantic_score)
                e.matched_chunks = list(set(e.matched_chunks + r.matched_chunks))[:3]
                e.source_types = list(set(e.source_types + r.source_types))
        for aid, rrf_score in scores.items():
            if aid in merged:
                merged[aid].hybrid_score = rrf_score
        return sorted(merged.values(), key=lambda x: x.hybrid_score, reverse=True)

    @staticmethod
    def _fts_query(query: str) -> str:
        words = [w.strip() for w in query.split() if len(w.strip()) > 1]
        return " OR ".join(f'"{w}"*' for w in words) if words else query

    @staticmethod
    def _snippet(text: str, query: str, window: int = 200) -> str:
        lower = text.lower()
        for word in query.lower().split():
            pos = lower.find(word)
            if pos != -1:
                return text[max(0, pos - 80):min(len(text), pos + window)]
        return text[:window]

    def _log(self, query: str, mode: str, count: int, top_id: int | None, ms: int) -> None:
        try:
            conn = get_conn()
            conn.execute(
                "INSERT INTO kb_search_logs (query, search_type, results_count, top_article_id, duration_ms) VALUES (?,?,?,?,?)",
                (query, mode, count, top_id, ms)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
