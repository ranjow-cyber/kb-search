# ============================================================
#  search_engine.py — Silnik hybrydowy (SQLite FTS5 + Semantic)
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

MODEL_NAME = "BAAI/bge-small-en-v1.5"
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
    fts_score: float = 0.0        # 0.0 – 1.0 (znormalizowany)
    semantic_score: float = 0.0   # 0.0 – 1.0 (cosine similarity)
    hybrid_score: float = 0.0     # 0.0 – 1.0 (RRF znormalizowany)
    matched_chunks: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)


class KnowledgeBaseSearch:

    def search(self, query: str, top_k: int = 10, mode: str = "hybrid", category_id: int | None = None) -> list[SearchResult]:
        t0 = time.perf_counter()

        if mode == "fts":
            results = self._fts_search(query, top_k * 2, category_id)
            # dla trybu fts ustaw hybrid_score = fts_score żeby frontend miał co pokazać
            for r in results:
                r.hybrid_score = r.fts_score
        elif mode == "semantic":
            results = self._semantic_search(query, top_k * 2, category_id)
            # dla trybu semantic ustaw hybrid_score = semantic_score
            for r in results:
                r.hybrid_score = r.semantic_score
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

        # SQLite FTS5 zwraca ujemne ranki — im bardziej ujemny tym lepszy
        # Zbieramy wszystkie ranki żeby znormalizować do 0-1
        ranks = [abs(float(r["fts_rank"])) for r in rows if r["fts_rank"] is not None]
        max_rank = max(ranks) if ranks else 1.0

        for r in rows:
            raw_rank = abs(float(r["fts_rank"])) if r["fts_rank"] else 0.0
            # normalizacja: najlepszy wynik = 1.0, gorsze = mniej
            score = round(raw_rank / max_rank, 4) if max_rank > 0 else 0.0
            results[r["id"]] = SearchResult(
                article_id=r["id"], title=r["title"], summary=r["summary"],
                slug=r["slug"], category=r["category"],
                fts_score=score, source_types=["article"],
            )

        # załączniki
        try:
            att_rows = conn.execute(f"""
                SELECT a.id, a.title, a.summary, a.slug,
                       c.name AS category, af.rank AS fts_rank, att.extracted_text
                FROM kb_attachments_fts af
                JOIN kb_attachments att ON att.id = af.rowid
                JOIN kb_articles a ON a.id = att.article_id
                LEFT JOIN kb_categories c ON c.id = a.category_id
                WHERE kb_attachments_fts MATCH :q AND a.is_published = 1 {cat_filter}
                ORDER BY rank LIMIT :lim
            """, {"q": self._fts_query(query), "cat": category_id, "lim": limit}).fetchall()
        except Exception as e:
            logger.warning(f"FTS załączniki błąd: {e}")
            att_rows = []

        att_ranks = [abs(float(r["fts_rank"])) for r in att_rows if r["fts_rank"] is not None]
        max_att_rank = max(att_ranks) if att_ranks else 1.0

        for r in att_rows:
            raw_rank = abs(float(r["fts_rank"])) if r["fts_rank"] else 0.0
            score = round(raw_rank / max_att_rank, 4) if max_att_rank > 0 else 0.0
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
                results[r["id"]].matched_chunks.append(self._snippet(r["extracted_text"], query))

        conn.close()
        return sorted(results.values(), key=lambda x: x.fts_score, reverse=True)

    # ----------------------------------------------------------
    # Semantic search
    # ----------------------------------------------------------
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

        # normalizacja wektorów przed dot product = cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_matrix = emb_matrix / np.where(norms == 0, 1, norms)
        q_norm = np.linalg.norm(query_emb)
        query_emb = query_emb / (q_norm if q_norm > 0 else 1)

        # cosine similarity — wynik w zakresie -1 do 1, typowo 0 do 1
        scores = emb_matrix @ query_emb

        # grupuj po article_id, bierz max similarity
        article_map: dict[int, dict] = {}
        for idx, row in enumerate(rows):
            art_id = row["article_id"]
            if art_id is None:
                continue
            score = float(scores[idx])
            # cosine similarity: 0.0-1.0 → przekształć na procenty (0% = brak podobieństwa, 100% = identyczny)
            # odrzucamy ujemne (brak podobieństwa)
            score = max(0.0, score)
            if art_id not in article_map or score > article_map[art_id]["score"]:
                article_map[art_id] = {
                    "score": score, "title": row["title"], "summary": row["summary"],
                    "slug": row["slug"], "category": row["category"],
                    "source_type": row["source_type"], "chunks": [],
                }
            if score > 0.2:
                article_map[art_id]["chunks"].append(row["chunk_text"][:300])

        results = [
            SearchResult(
                article_id=aid, title=v["title"], summary=v["summary"],
                slug=v["slug"], category=v["category"],
                semantic_score=round(v["score"], 4),
                matched_chunks=v["chunks"][:3],
                source_types=[v["source_type"]],
            )
            for aid, v in article_map.items()
        ]
        return sorted(results, key=lambda x: x.semantic_score, reverse=True)[:limit]

    # ----------------------------------------------------------
    # Reciprocal Rank Fusion — łączy FTS i Semantic
    # ----------------------------------------------------------
    def _rrf(self, fts: list[SearchResult], sem: list[SearchResult], k: int = 60) -> list[SearchResult]:
        """
        RRF Score = suma 1/(k + pozycja) dla każdej listy.
        Artykuł wysoko w obu listach = najwyższy wynik końcowy.
        Wynik jest normalizowany do 0-1.
        """
        rrf_scores: dict[int, float] = {}
        for rank, r in enumerate(fts):
            rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0) + 1 / (k + rank + 1)
        for rank, r in enumerate(sem):
            rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0) + 1 / (k + rank + 1)

        # normalizacja RRF do 0-1
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

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

        for aid, rrf_score in rrf_scores.items():
            if aid in merged:
                # normalizuj do 0-1
                merged[aid].hybrid_score = round(rrf_score / max_rrf, 4)

        return sorted(merged.values(), key=lambda x: x.hybrid_score, reverse=True)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _fts_query(query: str) -> str:
        words = [w.strip() for w in query.split() if len(w.strip()) > 1]
        return " OR ".join(f'"{w}"*' for w in words) if words else f'"{query}"'

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
