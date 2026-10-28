# Wykrywanie spamu SMS: wpływ przetwarzania danych na jakość predykcji

Projekt z przedmiotu Reprezentacja Danych. Temat 4: **SMS Spam Collection** (klasy `ham` / `spam`).
Metryka główna: **F1 dla klasy spam**; dodatkowo precision i recall spamu, macro-F1, PR-AUC.

## Dane

| | |
|---|---|
| Zbiór | SMS Spam Collection v.1, UCI Machine Learning Repository, id 228 |
| Źródło | https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip |
| Plik | `SMSSpamCollection`, SHA-256 `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |
| Pobrano | 2026-10-29 |
| Licencja | CC BY 4.0 (UCI). Autorzy: T. A. Almeida, J. M. Gómez Hidalgo |
| Cytowanie | Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results.* DocEng 2011 |

Surowy plik nie jest w repozytorium. `smsspam.data.download()` pobiera go i odrzuca każdą kopię o innej sumie SHA-256.
Identyfikator rekordu `id` to numer linii (od 0) w oryginalnym pliku.

## Odtworzenie

```bash
uv sync
uv run python -m smsspam.split   # pobiera dane, grupuje bliskie duplikaty, zapisuje podział
uv run python -m smsspam.eda     # tabele i wykresy do reports/milestone1/
```

## Podział danych (zamrożony)

- Brak oficjalnego podziału, więc 70/15/15 (train/val/test) ze stratyfikacją po etykiecie, **seed 42**.
- Dokładne i bliskie duplikaty są grupowane i zawsze trafiają do tej samej części
  (`StratifiedGroupKFold`, 20 foldów: 3 test, 3 val, 14 train).
- Bliski duplikat: cosinus TF-IDF n-gramów znakowych 3-5 (`char_wb`) ≥ 0,7; składowe spójne grafu tworzą grupy.
  Wektoryzator do tej kontroli widzi wszystkie teksty, ale nie etykiety. To kontrola podziału, nie cecha modelu.
- Wynik: `data/splits/split_seed42.csv` (`id, label, split, dup_group, text_sha1`) oraz
  `data/splits/split_seed42.meta.json` (liczebności, konfiguracja, wersje bibliotek, kontrola przecieku).
- Wszystkie warianty (B0, B1, P1-P3, M, L) oceniamy na tych samych rekordach testowych.

Konfiguracja: `configs/data.toml`.

## Struktura

```
configs/data.toml        źródło danych, seed, proporcje, próg bliskich duplikatów
smsspam/data.py          pobranie + weryfikacja SHA-256, wczytanie, złączenie z podziałem
smsspam/split.py         grupy bliskich duplikatów, podział, kontrola przecieku
smsspam/patterns.py      wzorce (telefon, kod krótki, kwota, URL, ...) dla EDA i wariantów P
smsspam/eda.py           EDA Milestone 1
data/splits/             zamrożony podział (commitowany)
reports/milestone1/      eda.md, summary.json, wykresy, notatka z hipotezami
```

## Narzędzia generatywne

Część kodu i dokumentacji przygotowano z pomocą Claude Code (Anthropic). Zespół zweryfikował kod, wyniki i źródła.
