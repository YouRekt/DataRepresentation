"""Metrics, validation threshold choice and paired bootstrap for spam detection."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold with the highest spam F1 (predict spam when score >= threshold)."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "f1_spam": f1_score(y, pred, zero_division=0),
        "precision_spam": tp / (tp + fp) if tp + fp else 0.0,
        "recall_spam": tp / (tp + fn) if tp + fn else 0.0,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y, score),
        "FP": fp,
        "FN": fn,
    }


def paired_bootstrap(y, pred_a, pred_b, n: int = 2000, seed: int = 42, groups=None) -> dict:
    """95% CI of F1(b) - F1(a) for spam, both models scored on the same resampled test records.

    With `groups`, whole near-duplicate groups are resampled, so copies of one campaign
    move together (records inside a group are not independent).
    """
    y, pred_a, pred_b = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    units = np.arange(len(y)) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]
    k = units.max() + 1
    # Per-unit counts: true spam, predicted spam and true positives for each model.
    count = lambda v: np.bincount(units, weights=v, minlength=k)
    cols = np.stack([count(y), count(pred_a), count(pred_a * y), count(pred_b), count(pred_b * y)])
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.full(k, 1 / k), size=n)  # how often each unit is drawn
    pos, pa, tpa, pb, tpb = (w @ cols.T).T
    f1 = lambda tp, pp: np.divide(2 * tp, pp + pos, out=np.zeros_like(tp), where=(pp + pos) > 0)
    diff = f1(tpb, pb) - f1(tpa, pa)
    full = lambda p: 2 * (p * y).sum() / (p.sum() + y.sum())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"diff": float(full(pred_b) - full(pred_a)), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_0": float((diff <= 0).mean())}
