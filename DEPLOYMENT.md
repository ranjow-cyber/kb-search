# 🚀 Instrukcja wdrożenia na Railway

## Co to jest Railway?
Railway to platforma hostingowa — wrzucasz kod, ona automatycznie
go uruchamia i daje Ci publiczny URL (https://...). Darmowy plan
wystarczy na demo.

---

## Krok 1 — Załóż konto GitHub (jeśli nie masz)

1. Wejdź na https://github.com
2. Kliknij "Sign up"
3. Podaj e-mail, hasło, nazwę użytkownika
4. Potwierdź e-mail

---

## Krok 2 — Wgraj kod na GitHub

1. Zaloguj się na https://github.com
2. Kliknij "+" (prawy górny róg) → "New repository"
3. Nazwa: `kb-search` (lub dowolna)
4. Widoczność: Private (prywatne — tylko Ty widzisz)
5. Kliknij "Create repository"

6. Na stronie repozytorium kliknij "uploading an existing file"
7. Przeciągnij WSZYSTKIE pliki z folderu `kb_demo`:
   - main.py
   - search_engine.py
   - database.py
   - extractor.py  (skopiuj z poprzedniego folderu kb_search)
   - schema.sql
   - requirements.txt
   - Dockerfile
   - railway.toml
8. Kliknij "Commit changes"

---

## Krok 3 — Załóż konto Railway

1. Wejdź na https://railway.app
2. Kliknij "Start a New Project"
3. Zaloguj się przez GitHub (kliknij "Continue with GitHub")
4. Autoryzuj Railway dostęp do GitHub

---

## Krok 4 — Wdróż projekt

1. Na Railway kliknij "New Project"
2. Wybierz "Deploy from GitHub repo"
3. Znajdź i kliknij swoje repozytorium `kb-search`
4. Railway automatycznie wykryje Dockerfile i zacznie budować

⏳ Pierwsze budowanie trwa **5-10 minut** (pobiera model AI ~90MB).
Kolejne wdrożenia są szybsze.

---

## Krok 5 — Uzyskaj publiczny URL

1. Po zbudowaniu kliknij na swój serwis w Railway
2. Przejdź do zakładki "Settings" → "Networking"
3. Kliknij "Generate Domain"
4. Dostaniesz URL w stylu: `https://kb-search-production.up.railway.app`

✅ Twoja aplikacja działa! Sprawdź:
- `https://TWOJ-URL.up.railway.app/health` → powinno zwrócić {"status":"ok"}
- `https://TWOJ-URL.up.railway.app/search?q=hasło` → wyniki wyszukiwania
- `https://TWOJ-URL.up.railway.app/docs` → interaktywna dokumentacja API

---

## Krok 6 — Połącz frontend z backendem

W pliku `06_frontend.jsx` zmień linię:
```
const API_BASE = "http://localhost:8000";
```
na:
```
const API_BASE = "https://TWOJ-URL.up.railway.app";
```

I wyłącz tryb DEMO (ustaw `demoMode` na `false` jako domyślny).

---

## Dane demo (wbudowane)

Aplikacja automatycznie wstawia 5 przykładowych artykułów przy pierwszym uruchomieniu:
- Procedura odzyskiwania dostępu
- Konfiguracja VPN
- Polityka haseł i 2FA
- Onboarding nowego pracownika
- Procedura backupów

---

## Limity darmowego planu Railway

| Co          | Limit               |
|-------------|---------------------|
| Godziny     | 500h/miesiąc        |
| RAM         | 512 MB              |
| Dysk        | 1 GB (na bazę SQLite)|
| Bandwidth   | 100 GB/miesiąc      |

Na demo w zupełności wystarcza.

---

## Rozwiązywanie problemów

**Build się nie kończy / błąd:**
→ Kliknij na serwis → "Deployments" → kliknij ostatnie wdrożenie → "View logs"

**Aplikacja nie startuje:**
→ Sprawdź logi, najczęstsza przyczyna: brakujący plik w repozytorium

**Model AI nie pobiera się:**
→ Railway może mieć timeout przy pierwszym buildzie — spróbuj ponownie wdrożyć
