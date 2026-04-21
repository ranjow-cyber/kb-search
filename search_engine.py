# ============================================================
#  search_engine.py — Silnik hybrydowy (SQLite FTS5 + Semantic)
# ============================================================
from __future__ import annotations
import json
import time
import logging
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from database import get_conn

logger = logging.getLogger(__name__)

MODEL_NAME  = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Ładowanie modelu embeddingów...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ------------------------------------------------------------
# Dataclass wyniku
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Główna klasa
# ------------------------------------------------------------
class KnowledgeBaseSearch:

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        category_id: int | None = None,
    ) -> list[SearchResult]:
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

    # ----------------------------------------------------------
    # FTS5 — pełnotekstowe
    # ----------------------------------------------------------
    def _fts_search(self, query: str, limit: int, category_id: int | None) -> list[SearchResult]:
        conn = get_conn()
        results: dict[int, SearchResult] = {}

        # wyszukiwanie w artykułach
        cat_filter = "AND a.category_id = :cat" if category_id else ""
        try:
            rows = conn.execute(f"""
                SELECT a.id, a.title, a.summary, a.slug,
                       c.name AS category,
                       rank AS fts_rank
                FROM kb_articles_fts f
                JOIN kb_articles a ON a.id = f.rowid
                LEFT JOIN kb_categories c ON c.id = a.category_id
                WHERE kb_articles_fts MATCH :q
                  AND a.is_published = 1
                  {cat_filter}
                ORDER BY rank
                LIMIT :lim
            """, {"q": self._fts_query(query), "cat": category_id, "lim": limit}).fetchall()
        except Exception as e:
            logger.warning(f"FTS artykuły błąd: {e}")
            rows = []

        for r in rows:
            score = max(0.0, min(1.0, abs(float(r["fts_rank"])) / 10.0))
            results[r["id"]] = SearchResult(
                article_id=r["id"], title=r["title"], summary=r["summary"],
                slug=r["slug"], category=r["category"],
                fts_score=score, source_types=["article"],
            )

        # wyszukiwanie w załącznikach
        try:
            att_rows = conn.execute(f"""
                SELECT a.id, a.title, a.summary, a.slug,
                       c.name AS category,
                       af.rank AS fts_rank,
                       att.extracted_text
                FROM kb_attachments_fts af
                JOIN kb_attachments att ON att.id = af.rowid
                JOIN kb_articles a ON a.id = att.article_id
                LEFT JOIN kb_categories c ON c.id = a.category_id
                WHERE kb_attachments_fts MATCH :q
                  AND a.is_published = 1
                  {cat_filter}
                ORDER BY rank
                LIMIT :lim
            """, {"q": self._fts_query(query), "cat": category_id, "lim": limit}).fetchall()
        except Exception as e:
            logger.warning(f"FTS załączniki błąd: {e}")
            att_rows = []

        for r in att_rows:
            score = max(0.0, min(1.0, abs(float(r["fts_rank"])) / 10.0))
            if r["id"] not in results:
                results[r["id"]] = SearchResult(
                    article_id=r["id"], title=r["title"], summary=r["summary"],
                    slug=r["slug"], category=r["category"],
                    fts_score=score, source_types=["attachment"],
                )
            else:
                results[r["id"]].fts_score = max(results[r["id"]].fts_score, score)
                results[r["id"]].source_types.append("attachment")

            if r["extracted_text"]:
                snippet = self._snippet(r["extracted_text"], query)
                results[r["id"]].matched_chunks.append(snippet)

        conn.close()
        return sorted(results.values(), key=lambda x: x.fts_score, reverse=True)

    # ----------------------------------------------------------
    # Semantic search
    # ----------------------------------------------------------
    def _semantic_search(self, query: str, limit: int, category_id: int | None) -> list[SearchResult]:
        query_emb = get_model().encode([query], normalize_embeddings=True)[0]

        conn = get_conn()
        cat_filter = "AND a.category_id = :cat" if category_id else ""

        rows = conn.execute(f"""
            SELECT e.source_type, e.source_id, e.chunk_text, e.embedding_json,
                   a.id AS article_id, a.title, a.summary, a.slug,
                   c.name AS category
            FROM kb_embeddings e
            LEFT JOIN kb_articles a ON (
                (e.source_type = 'article' AND a.id = e.source_id)
                OR (e.source_type = 'attachment_chunk'
                    AND a.id = (SELECT article_id FROM kb_attachments WHERE id = e.source_id))
            )
            LEFT JOIN kb_categories c ON c.id = a.category_id
            WHERE a.is_published = 1
              {cat_filter}
        """, {"cat": category_id}).fetchall()
        conn.close()

        if not rows:
            return []

        emb_matrix = np.array([json.loads(r["embedding_json"]) for r in rows], dtype=np.float32)
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
                slug=v["slug"], category=v["category"],
                semantic_score=v["score"],
                matched_chunks=v["chunks"][:3],
                source_types=[v["source_type"]],
            )
            for aid, v in article_map.items()
        ]
        return sorted(results, key=lambda x: x.semantic_score, reverse=True)[:limit]

    # ----------------------------------------------------------
    # Reciprocal Rank Fusion
    # ----------------------------------------------------------
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
                e.fts_score      = max(e.fts_score, r.fts_score)
                e.semantic_score = max(e.semantic_score, r.semantic_score)
                e.matched_chunks = list(set(e.matched_chunks + r.matched_chunks))[:3]
                e.source_types   = list(set(e.source_types + r.source_types))

        for aid, rrf_score in scores.items():
            if aid in merged:
                merged[aid].hybrid_score = rrf_score

        return sorted(merged.values(), key=lambda x: x.hybrid_score, reverse=True)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _fts_query(query: str) -> str:
        """Konwertuj na zapytanie FTS5 (każde słowo z * dla prefix search)."""
        words = [w.strip() for w in query.split() if len(w.strip()) > 1]
        return " OR ".join(f'"{w}"*' for w in words) if words else query

    @staticmethod
    def _snippet(text: str, query: str, window: int = 200) -> str:
        """Znajdź fragment tekstu zawierający słowa z zapytania."""
        lower = text.lower()
        for word in query.lower().split():
            pos = lower.find(word)
            if pos != -1:
                start = max(0, pos - 80)
                end   = min(len(text), pos + window)
                return text[start:end]
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
