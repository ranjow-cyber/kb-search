# ============================================================
#  main.py — FastAPI backend (SQLite edition)
#  Uruchomienie: uvicorn main:app --host 0.0.0.0 --port 8000
# ============================================================
from __future__ import annotations
import os
import json
import shutil
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import get_conn, init_db
from search_engine import KnowledgeBaseSearch, get_model
from extractor import extract_text, chunk_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("KB_UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="KB Search API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

searcher = KnowledgeBaseSearch()


# ------------------------------------------------------------------
# Startup — inicjalizacja bazy + dane demo
# ------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    init_db()
    _seed_demo_data()
    logger.info("✅ Aplikacja gotowa")


def _seed_demo_data():
    """Wstaw przykładowe artykuły jeśli baza jest pusta."""
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM kb_articles").fetchone()[0]
    if count > 0:
        conn.close()
        return

    logger.info("Wstawianie danych demo...")

    # kategorie
    conn.execute("INSERT INTO kb_categories (name, slug) VALUES ('Bezpieczeństwo', 'bezpieczenstwo')")
    conn.execute("INSERT INTO kb_categories (name, slug) VALUES ('Infrastruktura', 'infrastruktura')")
    conn.execute("INSERT INTO kb_categories (name, slug) VALUES ('Ogólne', 'ogolne')")
    conn.commit()

    articles = [
        (1, "Procedura odzyskiwania dostępu do systemu", "odzyskiwanie-dostepu",
         "Krok po kroku jak zresetować hasło i odzyskać dostęp do konta.",
         """## Odzyskiwanie dostępu

Jeśli użytkownik zapomniał hasła, może skorzystać z procedury resetowania.

### Kroki
1. Przejdź na stronę logowania i kliknij "Zapomniałem hasła"
2. Podaj adres e-mail powiązany z kontem
3. Sprawdź skrzynkę pocztową — otrzymasz link aktywacyjny
4. Kliknij link i ustaw nowe hasło (min. 12 znaków, duże litery, cyfry, znaki specjalne)
5. Zaloguj się nowym hasłem

### Blokada konta
Po 5 nieudanych próbach konto zostaje zablokowane na 30 minut.
Administrator może odblokować konto ręcznie przez panel zarządzania."""),

        (2, "Konfiguracja VPN dla pracowników zdalnych", "konfiguracja-vpn",
         "Instrukcja instalacji i konfiguracji klienta VPN na Windows i macOS.",
         """## Konfiguracja VPN

Wszyscy pracownicy pracujący zdalnie muszą korzystać z firmowego VPN.

### Windows
1. Pobierz klienta VPN ze strony IT: vpn.firma.pl/download
2. Uruchom instalator jako administrator
3. Wprowadź adres serwera: vpn.firma.pl:443
4. Zaloguj się danymi do Active Directory

### macOS
1. Otwórz Preferencje systemowe → Sieć
2. Kliknij + i wybierz VPN (IKEv2)
3. Wpisz adres serwera i identyfikator zdalny
4. Wprowadź dane logowania

### Rozwiązywanie problemów
- Brak połączenia: sprawdź czy port 443 nie jest blokowany
- Powolne połączenie: zmień protokół na UDP
- Kontakt: helpdesk@firma.pl"""),

        (3, "Polityka haseł i uwierzytelnianie dwuskładnikowe", "polityka-hasel",
         "Wymagania haseł i konfiguracja 2FA w organizacji.",
         """## Polityka haseł

### Wymagania
- Minimum 12 znaków
- Co najmniej jedna wielka litera
- Co najmniej jedna cyfra
- Co najmniej jeden znak specjalny (!@#$%^&*)
- Hasło nie może być takie samo jak poprzednie 5 haseł
- Zmiana hasła co 90 dni

### Uwierzytelnianie dwuskładnikowe (2FA)
2FA jest obowiązkowe dla wszystkich kont z dostępem administracyjnym.

Obsługiwane metody:
1. Aplikacja TOTP (Google Authenticator, Authy)
2. SMS na zarejestrowany numer
3. Klucz sprzętowy FIDO2/WebAuthn

### Konfiguracja 2FA
1. Zaloguj się do panelu konta
2. Przejdź do Ustawienia → Bezpieczeństwo
3. Kliknij "Włącz 2FA" i wybierz metodę
4. Zachowaj kody zapasowe w bezpiecznym miejscu"""),

        (3, "Onboarding nowego pracownika — lista kontrolna", "onboarding-checklist",
         "Kompletna lista zadań do wykonania przy wdrażaniu nowego pracownika.",
         """## Lista kontrolna onboardingu

### Przed pierwszym dniem (IT)
- [ ] Utworzenie konta w Active Directory
- [ ] Przypisanie licencji Microsoft 365
- [ ] Konfiguracja skrzynki e-mail
- [ ] Dostęp do systemów według roli
- [ ] Przygotowanie sprzętu (laptop, telefon)

### Pierwszy dzień
- [ ] Przekazanie sprzętu i danych logowania
- [ ] Szkolenie z polityki bezpieczeństwa
- [ ] Konfiguracja VPN
- [ ] Instalacja wymaganych aplikacji
- [ ] Zapoznanie z bazą wiedzy

### Pierwszy tydzień
- [ ] Szkolenie z systemów wewnętrznych
- [ ] Spotkanie z zespołem
- [ ] Dostęp do projektów
- [ ] Konfiguracja 2FA"""),

        (2, "Procedura tworzenia kopii zapasowych", "backup-procedura",
         "Harmonogram i procedury tworzenia backupów danych firmowych.",
         """## Kopie zapasowe

### Harmonogram automatycznych backupów
- **Codziennie o 2:00** — backup przyrostowy baz danych
- **Co niedzielę o 3:00** — pełny backup wszystkich serwerów
- **1. każdego miesiąca** — archiwum długoterminowe (przechowywane 1 rok)

### Lokalizacje przechowywania
1. Lokalny serwer NAS (szybki dostęp, 30 dni)
2. Azure Blob Storage (offsite, 90 dni)
3. Taśmy magnetyczne (archiwum roczne, sejf ognioodporny)

### Testowanie przywracania
Co kwartał wykonujemy test odtworzenia danych z kopii zapasowej.
Wyniki dokumentowane w raporcie audytu.

### Kontakt w przypadku awarii
- Dyżur 24/7: +48 22 000 0000
- E-mail: backup-alert@firma.pl"""),
    ]

    model = get_model()

    for cat_id, title, slug, summary, content in articles:
        cur = conn.execute(
            "INSERT INTO kb_articles (category_id, title, slug, summary, content, author) VALUES (?,?,?,?,?,?)",
            (cat_id, title, slug, summary, content, "Administrator")
        )
        article_id = cur.lastrowid
        conn.commit()

        # generuj embeddingi
        chunks = chunk_text(f"{title}\n\n{content}", chunk_size=500, overlap=50)
        embeddings = model.encode(chunks, normalize_embeddings=True)
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                "INSERT INTO kb_embeddings (source_type, source_id, chunk_index, chunk_text, embedding_model, embedding_json) VALUES (?,?,?,?,?,?)",
                ("article", article_id, idx, chunk, MODEL_NAME, json.dumps(emb.tolist()))
            )
        conn.commit()
        logger.info(f"  ✓ Artykuł '{title}' zaindeksowany ({len(chunks)} chunków)")

    conn.close()
    logger.info("✅ Dane demo wstawione")


# ------------------------------------------------------------------
# Pydantic schemas
# ------------------------------------------------------------------
class SearchResponse(BaseModel):
    article_id: int
    title: str
    summary: str | None
    slug: str
    category: str | None
    fts_score: float
    semantic_score: float
    hybrid_score: float
    matched_chunks: list[str]
    source_types: list[str]


class ArticleCreate(BaseModel):
    title: str
    content: str
    summary: str | None = None
    slug: str
    category_id: int | None = None
    author: str | None = None


# ------------------------------------------------------------------
# Endpointy
# ------------------------------------------------------------------
@app.get("/search", response_model=list[SearchResponse])
async def search(
    q: str = Query(..., min_length=2),
    mode: Literal["fts", "semantic", "hybrid"] = "hybrid",
    top_k: int = Query(10, ge=1, le=50),
    category_id: int | None = None,
):
    results = searcher.search(q, top_k=top_k, mode=mode, category_id=category_id)
    return [
        SearchResponse(
            article_id=r.article_id, title=r.title, summary=r.summary,
            slug=r.slug, category=r.category,
            fts_score=round(r.fts_score, 4), semantic_score=round(r.semantic_score, 4),
            hybrid_score=round(r.hybrid_score, 6),
            matched_chunks=r.matched_chunks, source_types=list(set(r.source_types)),
        )
        for r in results
    ]


@app.post("/articles")
async def create_article(article: ArticleCreate, background_tasks: BackgroundTasks):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO kb_articles (title, slug, summary, content, author, category_id) VALUES (?,?,?,?,?,?)",
        (article.title, article.slug, article.summary, article.content, article.author, article.category_id)
    )
    article_id = cur.lastrowid
    conn.commit()
    conn.close()
    background_tasks.add_task(_index_article, article_id, article.title, article.content)
    return {"id": article_id, "message": "Artykuł utworzony, indeksowanie w toku"}


@app.post("/articles/{article_id}/attachments")
async def upload_attachment(article_id: int, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".md", ".txt"}:
        raise HTTPException(415, "Nieobsługiwany format pliku")

    dest = UPLOAD_DIR / f"{article_id}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO kb_attachments (article_id, file_name, file_type, file_path, file_size_kb) VALUES (?,?,?,?,?)",
        (article_id, file.filename, suffix.lstrip("."), str(dest), dest.stat().st_size // 1024)
    )
    att_id = cur.lastrowid
    conn.commit()
    conn.close()

    background_tasks.add_task(_index_attachment, att_id, str(dest))
    return {"id": att_id, "message": "Załącznik wgrany, indeksowanie w toku"}


@app.get("/categories")
async def list_categories():
    conn = get_conn()
    rows = conn.execute("SELECT id, name, slug FROM kb_categories ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# Background tasks
# ------------------------------------------------------------------
def _index_article(article_id: int, title: str, content: str):
    model = get_model()
    chunks = chunk_text(f"{title}\n\n{content}")
    embeddings = model.encode(chunks, normalize_embeddings=True)
    conn = get_conn()
    conn.execute("DELETE FROM kb_embeddings WHERE source_type='article' AND source_id=?", (article_id,))
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        conn.execute(
            "INSERT INTO kb_embeddings (source_type, source_id, chunk_index, chunk_text, embedding_model, embedding_json) VALUES (?,?,?,?,?,?)",
            ("article", article_id, idx, chunk, MODEL_NAME, json.dumps(emb.tolist()))
        )
    conn.commit()
    conn.close()


def _index_attachment(att_id: int, file_path: str):
    try:
        text = extract_text(file_path)
    except Exception as e:
        logger.error(f"Ekstrakcja załącznika {att_id}: {e}")
        return
    conn = get_conn()
    conn.execute("UPDATE kb_attachments SET extracted_text=?, is_indexed=1 WHERE id=?", (text, att_id))
    conn.commit()
    model = get_model()
    chunks = chunk_text(text)
    embeddings = model.encode(chunks, normalize_embeddings=True)
    conn.execute("DELETE FROM kb_embeddings WHERE source_type='attachment_chunk' AND source_id=?", (att_id,))
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        conn.execute(
            "INSERT INTO kb_embeddings (source_type, source_id, chunk_index, chunk_text, embedding_model, embedding_json) VALUES (?,?,?,?,?,?)",
            ("attachment_chunk", att_id, idx, chunk, MODEL_NAME, json.dumps(emb.tolist()))
        )
    conn.commit()
    conn.close()


@app.get("/articles/list")
async def list_articles():
    conn = get_conn()
    rows = conn.execute("""
        SELECT a.id, a.title, a.summary, a.slug,
               c.name AS category
        FROM kb_articles a
        LEFT JOIN kb_categories c ON c.id = a.category_id
        WHERE a.is_published = 1
        ORDER BY a.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
