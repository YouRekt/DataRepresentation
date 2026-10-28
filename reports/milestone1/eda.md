# Milestone 1: dane i EDA (wygenerowane przez `python -m smsspam.eda`)

## Zbiór
- Rekordy: 5574; ham 4827, spam 747 (13.4% spamu).
- Braki / puste wiadomości: 0. Wiadomości ze spacją na początku/końcu: 187.
- Długość: 2-910 znaków. Wiadomości z emoji: 0.

## Duplikaty
- Unikalne teksty: 5171; nadmiarowe dokładne duplikaty: 403 (ham 309, spam 94; 6.4% hamów, 12.6% spamu).
- Teksty z konfliktem etykiet: 0.
- Po normalizacji (małe litery, bez interpunkcji, cyfry -> 0): 5103 unikalnych.
- Grupy bliskich duplikatów (cos char 3-5 >= 0,7): 4897; rekordy w grupach > 1: {'ham': 633, 'spam': 428}.

## Podział (seed 42, StratifiedGroupKFold, 70/15/15)

| część | ham | spam | razem | % spam |
|---|---:|---:|---:|---:|
| train | 3378 | 522 | 3900 | 13.4% |
| val | 725 | 112 | 837 | 13.4% |
| test | 724 | 113 | 837 | 13.5% |

## Długość (train)

| klasa | mediana znaków | Q1-Q3 | mediana słów | 120-160 znaków | > 160 znaków |
|---|---:|---:|---:|---:|---:|
| ham | 53 | 34-95 | 11 | 13.0% | 4.6% |
| spam | 149 | 134-157 | 25 | 74.7% | 8.8% |

## Znaczniki (train, % wiadomości w klasie)

| znacznik | ham | spam |
|---|---:|---:|
| Numer telefonu (>= 8 cyfr) | 0.1% | 61.5% |
| Kod krótki (5-6 cyfr) | 0.0% | 34.1% |
| Kwota (£, $, 150p) | 0.3% | 51.7% |
| URL / domena | 0.3% | 18.2% |
| Encja HTML (&lt; &gt; &amp;) | 6.7% | 0.0% |
| Emotikona :) ;-) | 3.3% | 0.0% |
| Wielokropek / !!! / ??? | 15.7% | 1.7% |
| Powtórzona litera (sooo) | 2.3% | 2.9% |
| Znak C1 z cp1252 (\x92) | 0.8% | 0.4% |
| Przewaga wielkich liter (>= 50%) | 2.3% | 3.1% |

## Słownictwo spoza treningu (słownik B1 z train, sprawdzany na val)
- Tokeny OOV: ham 6.8%, spam 12.7%.
- Tokeny liczbowe >= 5 cyfr w spamie (val): 98, z czego OOV 71.4%.
- Spam w val z numerem telefonu: 49.1%; spośród nich wszystkie numery nieznane z treningu: 85.5%.

## Najbardziej rozróżniające tokeny (train, tokenizacja B1, df >= 10)

| spam | df spam/ham | ham | df spam/ham |
|---|---:|---|---:|
| claim | 70/0 | gt | 0/177 |
| prize | 52/0 | lt | 0/175 |
| 150p | 48/0 | he | 0/126 |
| 18 | 40/0 | lor | 0/100 |
| tone | 38/0 | da | 0/95 |
| guaranteed | 34/0 | she | 0/87 |
| 500 | 33/0 | hope | 0/79 |
| cs | 31/0 | much | 0/79 |
| code | 27/0 | come | 1/152 |
| awarded | 27/0 | ask | 0/65 |
| 1000 | 26/0 | doing | 0/64 |
| uk | 50/1 | home | 1/115 |
| ringtone | 24/0 | later | 0/55 |
| mob | 24/0 | yeah | 0/55 |
| 100 | 23/0 | amp | 0/55 |
