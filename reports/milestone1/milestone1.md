# Milestone 1: dane, EDA, podział i hipotezy

Liczby pochodzą z `eda.md` / `summary.json` (generowane przez `python -m smsspam.eda`) oraz
`data/splits/split_seed42.meta.json`. Statystyki, na których opieramy hipotezy, liczymy tylko na treningu
(słownictwo spoza słownika: train → val). Test nie wpływa na wybór wariantów.

## 1. Dane i etykiety

- 5574 wiadomości: 4827 ham (86,6%), 747 spam (13,4%). Zgodne z readme zbioru.
- Braków i pustych wiadomości: 0. Długość 2-910 znaków. 187 wiadomości ma spację na początku lub końcu; zostawiamy surowy tekst.
- Kodowanie: UTF-8 poprawne, ale 35 wiadomości zawiera nieprzekodowane znaki cp1252 z zakresu C1 (`\x92` zamiast `’`).
- Emoji: 0 (korpus z 2011 r.). Obsługa emoji nie ma tu sensu jako wariant. Emotikony tekstowe `:)` występują tylko w ham (3,3%).
- Spam „wypełnia” limit jednego SMS-a: mediana 149 znaków, 74,7% spamu ma 120-160 znaków. Ham: mediana 53.

## 2. Duplikaty

- 403 nadmiarowe dokładne duplikaty (7,2%), 5171 unikalnych tekstów. Spam powtarza się dwa razy częściej (12,6% vs 6,4%).
- 0 tekstów z konfliktem etykiet.
- Bliskie duplikaty to głównie ten sam szablon spamu z innym numerem lub kwotą (np. „URGENT! … won a £800 prize … Call 0905…”).
  Przy cos ≥ 0,7: 4897 grup, największa 40 rekordów („Sorry, I'll call later…”), żadna para ham-spam.
- Poniżej 0,7 zaczynają się łączyć krótkie hamy o wspólnej frazie („I'm home.” / „I'm at home. Please call”), stąd próg 0,7.

**Decyzja:** zostawiamy wszystkie rekordy (naturalny rozkład i proporcje klas w teście), ale grupujemy duplikaty
przy podziale. Dodatkowo raportujemy wynik na teście po deduplikacji (781 unikalnych tekstów z 837).

## 3. Podział

| część | ham | spam | razem | % spam |
|---|---:|---:|---:|---:|
| train | 3378 | 522 | 3900 | 13,4% |
| val | 725 | 112 | 837 | 13,4% |
| test | 724 | 113 | 837 | 13,5% |

Kontrola przecieku (asercje w `smsspam/split.py`): żadna grupa ani dokładny tekst nie przechodzi między częściami;
0 rekordów val/test ma podobieństwo ≥ 0,7 do jakiegokolwiek rekordu treningowego.
Pozostaje pasmo 0,5-0,7: 120 rekordów testu (79 ham, 41 spam). To głównie te same kampanie spamowe w innej wersji.
Jest to realistyczne (kampanie wracają), ale może zawyżać wyniki, więc pokażemy też wynik na podzbiorze testu z podobieństwem < 0,5.

## 4. Co odróżnia spam (train)

| znacznik | ham | spam |
|---|---:|---:|
| Numer telefonu (≥ 8 cyfr) | 0,1% | 61,5% |
| Kwota (£, $, 150p) | 0,3% | 51,7% |
| Kod krótki (5-6 cyfr) | 0,0% | 34,1% |
| URL / domena | 0,3% | 18,2% |
| Encja HTML (`&lt;` `&gt;` `&amp;`) | 6,7% | 0,0% |
| Wielokropek / `!!!` / `???` | 15,7% | 1,7% |
| Powtórzona litera (`sooo`) | 2,3% | 2,9% |

- Słownik B1 (z treningu) a walidacja: 71,4% tokenów liczbowych ≥ 5 cyfr ze spamu nie występuje w treningu.
  W 85,5% spamów z walidacji z numerem telefonu **żaden** numer nie był widziany w treningu.
- Dwa najsilniejsze tokeny ham w tokenizacji B1 to `gt` (0/177) i `lt` (0/175): pozostałości po `&lt;#&gt;`,
  czyli placeholderze anonimizacji liczb w korpusie NUS, z którego pochodzi większość hamów. To artefakt zbierania danych, a nie cecha języka.

## 5. Hipotezy (warianty P przy tym samym modelu B1 i tych samych hiperparametrach)

**H1 / P1: maskowanie numerów telefonów, kodów krótkich i kwot** (`<PHONE>`, `<SHORTCODE>`, `<MONEY>`).
W B1 każdy numer to osobny rzadki token, którego model w większości nie zna. Maskowanie zamienia je we wspólne,
silnie spamowe cechy. Oczekujemy wzrostu recall spamu przy podobnej precyzji, czyli wzrostu F1 spamu.
Ryzyko: hamy z kwotami („$700”), fałszywe alarmy.

**H2 / P2: URL, maskowanie `<URL>` vs usuwanie.** URL jest w 18,2% spamu, ale tokenizator B1 już rozbija
„www.getzed.co.uk” na `www`, `getzed`, `co`, `uk` (`uk`: 50 spam / 1 ham). Oczekujemy: maskowanie ≈ bez zmian lub mały zysk,
usunięcie lekko obniży recall. Wariant mierzy, ile model zawdzięcza URL-om.

**H3 / P3: normalizacja powtórzeń znaków** (≥ 3 → 2: „sooo” → „soo”, „!!!!” → „!!”) plus test odporności na kontrolowane
literówki. Powtórzenia są rzadkie i podobnie częste w obu klasach, a tokenizator B1 pomija interpunkcję.
Oczekujemy braku istotnej zmiany na czystym teście (różnica w przedziale ufności). Ewentualny zysk może pojawić się na tekście z literówkami.

**Obserwacja dodatkowa / P4 opcjonalny: dekodowanie encji HTML.** Usuwa skrót `lt`/`gt` → ham. Spodziewamy się
niewielkiego spadku metryk, ale model mniej zależny od artefaktu korpusu. Pokażemy to jako analizę wrażliwości.

Po zmianach pojedynczych: połączenie najlepszych (np. P1 + P2).

## 6. Problemy i ryzyka

1. Mały test: 113 spamów, więc 1 wiadomość ≈ 0,9 p.p. recall. Różnice między wariantami P będą małe, dlatego potrzebny jest
   bootstrap sparowany na wspólnych rekordach testowych (≥ 1000 losowań).
2. Duplikaty w teście (65 ze 113 spamów należy do grup > 1) nadają niektórym kampaniom większą wagę: dodatkowo raportujemy test po deduplikacji.
3. Korpus z 2011 r., spam głównie brytyjski (£, numery 08/09). Wnioski nie przenoszą się wprost na współczesny spam.
4. Zbiór jest publiczny od 2011 r. (UCI, Kaggle), więc LLM i modele pretrained mogły go widzieć. Omówimy to w M4.
5. Do potwierdzenia z prowadzącym: zachowanie duplikatów z grupowaniem (zamiast deduplikacji) i próg 0,7.

## 7. Plan na Milestone 2

- B0: klasa większościowa (ham), więc F1 spamu = 0. Punkt odniesienia.
- B1: `TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")` na surowym tekście + `LogisticRegression`.
  Ustawienia zapisane w configu; `C` i próg decyzji dobierane wyłącznie na walidacji, potem zamrożone dla P1-P3.
- P1-P3 jako funkcje przetwarzania korzystające z `smsspam/patterns.py`; wyniki na val, potem bootstrap B1 vs najlepszy P na teście.
- Osobny eksperyment: n-gramy słowne vs znakowe.
