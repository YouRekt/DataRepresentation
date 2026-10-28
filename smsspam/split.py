"""Group near-duplicates and freeze the 70/15/15 stratified group split.

Run: uv run python -m smsspam.split
"""

import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from smsspam.data import SPLIT_DIR, load_config, load_raw


def char_tfidf(texts: pd.Series, cfg: dict) -> sparse.csr_matrix:
    # Fitted on all texts without labels: this is split control, not a model feature.
    vec = TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=(cfg["ngram_min"], cfg["ngram_max"]),
        lowercase=True,
        sublinear_tf=True,
    )
    return vec.fit_transform(texts)


def near_duplicate_groups(X: sparse.csr_matrix, threshold: float) -> np.ndarray:
    sim = sparse.triu(X @ X.T, k=1).tocoo()
    keep = sim.data >= threshold
    n = X.shape[0]
    adj = sparse.coo_matrix((np.ones(keep.sum()), (sim.row[keep], sim.col[keep])), shape=(n, n))
    _, groups = connected_components(adj, directed=False)
    return groups


def assign_split(labels: pd.Series, groups: np.ndarray, cfg: dict) -> np.ndarray:
    n_folds = cfg["n_folds"]
    n_test = round(cfg["test"] * n_folds)
    n_val = round(cfg["val"] * n_folds)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])
    fold = np.empty(len(labels), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        fold[idx] = k
    return np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "val", "train"))


def max_sim_to_train(X: sparse.csr_matrix, split: np.ndarray) -> np.ndarray:
    """Highest cosine similarity of every record to any training record."""
    train = np.flatnonzero(split == "train")
    best = (X @ X[train].T).max(axis=1).toarray().ravel()
    best[train] = np.nan
    return best


def main() -> None:
    cfg = load_config()
    df = load_raw()
    X = char_tfidf(df.text, cfg["near_dup"])
    df["dup_group"] = near_duplicate_groups(X, cfg["near_dup"]["threshold"])
    df["split"] = assign_split(df.label, df.dup_group.to_numpy(), cfg["split"])
    df["text_sha1"] = df.text.map(lambda t: hashlib.sha1(t.encode()).hexdigest())

    # Leak checks: no near-duplicate group and no exact text may span two parts.
    assert (df.groupby("dup_group").split.nunique() == 1).all()
    assert (df.groupby("text").split.nunique() == 1).all()

    sim = max_sim_to_train(X, df.split.to_numpy())
    held_out = df.split != "train"
    leak_bands = {
        f">={th}": {s: int(((sim >= th) & (df.split == s)).sum()) for s in ("val", "test")}
        for th in (0.5, 0.6, 0.7)
    }
    assert leak_bands[">=0.7"] == {"val": 0, "test": 0}

    group_sizes = df.dup_group.value_counts()
    meta = {
        "source": cfg["source"],
        "split_config": cfg["split"],
        "near_dup_config": cfg["near_dup"],
        "n_records": len(df),
        "counts": df.groupby(["split", "label"]).size().unstack().to_dict(orient="index"),
        "spam_share": df.groupby("split").label.apply(lambda s: round((s == "spam").mean(), 4)).to_dict(),
        "n_groups": int(df.dup_group.nunique()),
        "records_in_multi_groups": int((df.dup_group.map(group_sizes) > 1).sum()),
        "largest_group": int(group_sizes.max()),
        "held_out_max_sim_to_train": {
            "median": round(float(np.nanmedian(sim[held_out])), 4),
            "counts": leak_bands,
        },
        "environment": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ("numpy", "pandas", "scipy", "scikit-learn")},
        },
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    seed = cfg["split"]["seed"]
    df[["id", "label", "split", "dup_group", "text_sha1"]].to_csv(SPLIT_DIR / f"split_seed{seed}.csv", index=False)
    (SPLIT_DIR / f"split_seed{seed}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: meta[k] for k in ("counts", "spam_share", "n_groups", "largest_group", "held_out_max_sim_to_train")}, indent=2))


if __name__ == "__main__":
    main()
