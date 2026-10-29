"""Milestone 1 EDA: labels, counts, missing values, duplicates, split, and the
text markers behind the preprocessing hypotheses.

Dataset-level facts (counts, duplicates, split) use all records. Everything
that motivates a preprocessing variant uses the training part only, and the
out-of-vocabulary check looks at train -> val, so the test part never shapes
the hypotheses.

Run: uv run python -m smsspam.eda
"""

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from smsspam import patterns as P
from smsspam.data import ROOT, load_split

OUT = ROOT / "reports" / "milestone1"
LABELS = ("ham", "spam")
# Reference categorical slots 1-2, validated light-mode palette.
COLORS = {"ham": "#2a78d6", "spam": "#eb6834"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# scikit-learn default; B1 will use the same pattern.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"

MARKERS = {
    "Numer telefonu (>= 8 cyfr)": P.PHONE,
    "Kod krótki (5-6 cyfr)": P.SHORTCODE,
    "Kwota (£, $, 150p)": P.MONEY,
    "URL / domena": P.URL,
    "Encja HTML (&lt; &gt; &amp;)": P.HTML_ENTITY,
    "Emotikona :) ;-)": P.EMOTICON,
    "Wielokropek / !!! / ???": P.REPEATED_PUNCT,
    "Powtórzona litera (sooo)": P.REPEATED_LETTER,
    "Znak C1 z cp1252 (\\x92)": P.C1_CONTROL,
}


def share(mask: pd.Series) -> float:
    return round(float(mask.mean()), 4)


def overview(df: pd.DataFrame) -> dict:
    text = df.text
    return {
        "n_records": len(df),
        "label_counts": df.label.value_counts().to_dict(),
        "spam_share": share(df.label == "spam"),
        "missing_or_empty": int((text.str.strip() == "").sum()),
        "leading_or_trailing_space": int((text != text.str.strip()).sum()),
        "chars_min_max": [int(text.str.len().min()), int(text.str.len().max())],
        "emoji_messages": int(text.str.contains(P.EMOJI).sum()),
    }


def duplicates(df: pd.DataFrame) -> dict:
    redundant = df.duplicated("text")
    loose_key = df.text.map(lambda t: " ".join(re.sub(r"\W+", " ", re.sub(r"\d", "0", t.lower())).split()))
    group_size = df.dup_group.map(df.dup_group.value_counts())
    return {
        "unique_texts": int(df.text.nunique()),
        "redundant_exact_rows": int(redundant.sum()),
        "redundant_exact_by_label": df[redundant].label.value_counts().to_dict(),
        "redundant_exact_share_of_label": {
            lab: share(redundant[df.label == lab]) for lab in LABELS
        },
        "texts_with_conflicting_labels": int((df.groupby("text").label.nunique() > 1).sum()),
        "unique_after_lowercase_punct_digit_normalisation": int(loose_key.nunique()),
        "near_dup_groups": int(df.dup_group.nunique()),
        "records_in_multi_groups_by_label": df[group_size > 1].label.value_counts().to_dict(),
        "top_repeated": df.text.value_counts().head(5).to_dict(),
    }


def split_table(df: pd.DataFrame) -> dict:
    counts = df.groupby(["split", "label"]).size().unstack().reindex(["train", "val", "test"])
    counts["total"] = counts.sum(axis=1)
    counts["spam_share"] = (counts.spam / counts.total).round(4)
    return counts.to_dict(orient="index")


def lengths(train: pd.DataFrame) -> dict:
    chars = train.text.str.len()
    words = train.text.str.split().str.len()
    out = {}
    for lab in LABELS:
        m = train.label == lab
        out[lab] = {
            "chars_median": float(chars[m].median()),
            "chars_q1_q3": [float(chars[m].quantile(0.25)), float(chars[m].quantile(0.75))],
            "words_median": float(words[m].median()),
            "share_120_to_160_chars": share(chars[m].between(120, 160)),
            "share_over_160_chars": share(chars[m] > 160),
        }
    return out


def marker_prevalence(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, pat in MARKERS.items():
        hit = train.text.map(lambda t: pat.search(t) is not None)
        rows.append({"marker": name, **{lab: share(hit[train.label == lab]) for lab in LABELS}})
    return pd.DataFrame(rows)


def uppercase_heavy(train: pd.DataFrame) -> dict:
    def heavy(t: str) -> bool:
        letters = [c for c in t if c.isalpha()]
        return len(letters) >= 10 and sum(c.isupper() for c in letters) / len(letters) >= 0.5

    hit = train.text.map(heavy)
    return {lab: share(hit[train.label == lab]) for lab in LABELS}


def oov(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """How much of val is unseen by a B1-style vocabulary fitted on train."""
    vocab = set(CountVectorizer(token_pattern=TOKEN_PATTERN).fit(train.text).vocabulary_)
    analyze = CountVectorizer(token_pattern=TOKEN_PATTERN).build_analyzer()
    out = {}
    for lab in LABELS:
        tokens = [tok for t in val.text[val.label == lab] for tok in analyze(t)]
        numeric = [tok for tok in tokens if re.fullmatch(r"\d{5,}", tok)]
        out[lab] = {
            "token_oov_rate": round(sum(tok not in vocab for tok in tokens) / len(tokens), 4),
            "long_number_tokens": len(numeric),
            "long_number_oov_rate": round(sum(tok not in vocab for tok in numeric) / max(len(numeric), 1), 4),
        }
    spam_val = val[val.label == "spam"]
    phones = spam_val.text.map(lambda t: [re.sub(r"\D", "", m) for m in P.PHONE.findall(t)])
    with_phone = phones.map(bool)
    all_unseen = phones[with_phone].map(lambda ps: all(p not in vocab for p in ps))
    out["val_spam_with_phone"] = share(with_phone)
    out["val_spam_with_phone_all_unseen_in_train"] = share(all_unseen)
    return out


def discriminative_tokens(train: pd.DataFrame, k: int = 15, min_df: int = 10) -> dict:
    """Smoothed log ratio of per-class document frequency (B1 tokenisation)."""
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, binary=True, min_df=min_df)
    X = vec.fit_transform(train.text)
    vocab = vec.get_feature_names_out()
    is_spam = (train.label == "spam").to_numpy()
    df_spam = np.asarray(X[is_spam].sum(axis=0)).ravel()
    df_ham = np.asarray(X[~is_spam].sum(axis=0)).ravel()
    score = np.log((df_spam + 1) / (is_spam.sum() + 2)) - np.log((df_ham + 1) / ((~is_spam).sum() + 2))
    order = np.argsort(score)

    def pick(idx):
        return [{"token": vocab[i], "spam_df": int(df_spam[i]), "ham_df": int(df_ham[i]), "log_ratio": round(float(score[i]), 2)} for i in idx]

    return {"spam": pick(order[::-1][:k]), "ham": pick(order[:k])}


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=13, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_lengths(train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    bins = np.arange(0, 330, 10)
    chars = train.text.str.len().clip(upper=bins[-1] - 1)
    for lab in LABELS:
        c = chars[train.label == lab]
        weights = np.full(len(c), 100 / len(c))
        ax.hist(c, bins=bins, weights=weights, histtype="step", linewidth=2, color=COLORS[lab], label=f"{lab} (n={len(c)})")
    ax.axvline(160, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(163, ax.get_ylim()[1] * 0.92, "limit 1 SMS = 160 znaków", color=INK_2, fontsize=12, va="top")
    ax.set_xlabel("Długość wiadomości [znaki], zbiór treningowy (ostatni koszyk: ≥ 320)", color=INK_2, fontsize=12.5)
    ax.set_ylabel("% wiadomości w klasie", color=INK_2, fontsize=12.5)
    ax.set_xlim(0, bins[-1])
    ax.legend(frameon=False, fontsize=13, labelcolor=INK, loc="upper right", bbox_to_anchor=(1, 0.8))
    fig.tight_layout()
    fig.savefig(OUT / "fig_length.png", dpi=200)
    plt.close(fig)


def plot_markers(prev: pd.DataFrame) -> None:
    prev = prev.assign(diff=prev.spam - prev.ham).sort_values("diff")
    y = np.arange(len(prev))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style(ax)
    for offset, lab in ((h / 2 + 0.02, "spam"), (-h / 2 - 0.02, "ham")):
        vals = prev[lab].to_numpy() * 100
        ax.barh(y + offset, vals, height=h, color=COLORS[lab], label=lab)
        for yi, v in zip(y + offset, vals):
            ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=11.5, color=INK_2)
    ax.set_yticks(y, prev.marker, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% wiadomości w klasie ze znacznikiem (zbiór treningowy)", color=INK_2, fontsize=12.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=13, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_markers.png", dpi=200)
    plt.close(fig)


def write_markdown(s: dict, prev: pd.DataFrame) -> None:
    pct = lambda x: f"{100 * x:.1f}%"
    ov, dup = s["overview"], s["duplicates"]
    lines = [
        "# Milestone 1: dane i EDA (wygenerowane przez `python -m smsspam.eda`)",
        "",
        "## Zbiór",
        f"- Rekordy: {ov['n_records']}; ham {ov['label_counts']['ham']}, spam {ov['label_counts']['spam']} ({pct(ov['spam_share'])} spamu).",
        f"- Braki / puste wiadomości: {ov['missing_or_empty']}. Wiadomości ze spacją na początku/końcu: {ov['leading_or_trailing_space']}.",
        f"- Długość: {ov['chars_min_max'][0]}-{ov['chars_min_max'][1]} znaków. Wiadomości z emoji: {ov['emoji_messages']}.",
        "",
        "## Duplikaty",
        f"- Unikalne teksty: {dup['unique_texts']}; nadmiarowe dokładne duplikaty: {dup['redundant_exact_rows']} "
        f"(ham {dup['redundant_exact_by_label'].get('ham', 0)}, spam {dup['redundant_exact_by_label'].get('spam', 0)}; "
        f"{pct(dup['redundant_exact_share_of_label']['ham'])} hamów, {pct(dup['redundant_exact_share_of_label']['spam'])} spamu).",
        f"- Teksty z konfliktem etykiet: {dup['texts_with_conflicting_labels']}.",
        f"- Po normalizacji (małe litery, bez interpunkcji, cyfry -> 0): {dup['unique_after_lowercase_punct_digit_normalisation']} unikalnych.",
        f"- Grupy bliskich duplikatów (cos char 3-5 >= 0,7): {dup['near_dup_groups']}; rekordy w grupach > 1: {dup['records_in_multi_groups_by_label']}.",
        "",
        "## Podział (seed 42, StratifiedGroupKFold, 70/15/15)",
        "",
        "| część | ham | spam | razem | % spam |",
        "|---|---:|---:|---:|---:|",
    ]
    for part, row in s["split"].items():
        lines.append(f"| {part} | {row['ham']} | {row['spam']} | {row['total']} | {pct(row['spam_share'])} |")
    lines += [
        "",
        "## Długość (train)",
        "",
        "| klasa | mediana znaków | Q1-Q3 | mediana słów | 120-160 znaków | > 160 znaków |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lab, row in s["lengths_train"].items():
        q1, q3 = row["chars_q1_q3"]
        lines.append(f"| {lab} | {row['chars_median']:.0f} | {q1:.0f}-{q3:.0f} | {row['words_median']:.0f} | {pct(row['share_120_to_160_chars'])} | {pct(row['share_over_160_chars'])} |")
    lines += ["", "## Znaczniki (train, % wiadomości w klasie)", "", "| znacznik | ham | spam |", "|---|---:|---:|"]
    for _, row in prev.iterrows():
        lines.append(f"| {row.marker} | {pct(row.ham)} | {pct(row.spam)} |")
    up = s["uppercase_heavy_train"]
    lines.append(f"| Przewaga wielkich liter (>= 50%) | {pct(up['ham'])} | {pct(up['spam'])} |")
    o = s["oov_train_to_val"]
    lines += [
        "",
        "## Słownictwo spoza treningu (słownik B1 z train, sprawdzany na val)",
        f"- Tokeny OOV: ham {pct(o['ham']['token_oov_rate'])}, spam {pct(o['spam']['token_oov_rate'])}.",
        f"- Tokeny liczbowe >= 5 cyfr w spamie (val): {o['spam']['long_number_tokens']}, z czego OOV {pct(o['spam']['long_number_oov_rate'])}.",
        f"- Spam w val z numerem telefonu: {pct(o['val_spam_with_phone'])}; spośród nich wszystkie numery nieznane z treningu: {pct(o['val_spam_with_phone_all_unseen_in_train'])}.",
        "",
        "## Najbardziej rozróżniające tokeny (train, tokenizacja B1, df >= 10)",
        "",
        "| spam | df spam/ham | ham | df spam/ham |",
        "|---|---:|---|---:|",
    ]
    for sp, hm in zip(s["discriminative_tokens"]["spam"], s["discriminative_tokens"]["ham"]):
        lines.append(f"| {sp['token']} | {sp['spam_df']}/{sp['ham_df']} | {hm['token']} | {hm['spam_df']}/{hm['ham_df']} |")
    (OUT / "eda.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_split()
    train, val = df[df.split == "train"], df[df.split == "val"]
    prev = marker_prevalence(train)
    summary = {
        "overview": overview(df),
        "duplicates": duplicates(df),
        "split": split_table(df),
        "lengths_train": lengths(train),
        "markers_train": prev.set_index("marker").to_dict(orient="index"),
        "uppercase_heavy_train": uppercase_heavy(train),
        "oov_train_to_val": oov(train, val),
        "discriminative_tokens": discriminative_tokens(train),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(summary, prev)
    plot_lengths(train)
    plot_markers(prev)
    print((OUT / "eda.md").read_text())


if __name__ == "__main__":
    main()
