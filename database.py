# ============================================================
#  database.py — SQLite connection helper
# ============================================================
import sqlite3
import os
from pathlib import Path

DB_PATH = os.getenv("KB_DB_PATH", "knowledge_base.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")   # lepsza wydajność przy równoległych odczytach
    return conn


def init_db() -> None:
    """Utwórz tabele jeśli nie istnieją (przy pierwszym uruchomieniu)."""
    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    conn = get_conn()
    conn.executescript(schema)
    conn.commit()
    conn.close()
    print(f"✅ Baza danych gotowa: {DB_PATH}")
