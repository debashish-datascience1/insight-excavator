"""
Statistical analysis library — the trust layer of Insight Excavator.

Every function takes a DataFrame + column args and returns:
  {test, stat, effect_size, effect_size_label, p_value, n, summary_numbers}

No LLM involved. These functions are the gatekeeper: an insight only survives
if it passes through here with p < 0.05 and a meaningful effect size.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import (
    pearsonr, spearmanr, shapiro,
    ttest_ind, mannwhitneyu,
    f_oneway, kruskal,
    chi2_contingency,
    linregress,
    binomtest,
)
from sklearn.ensemble import IsolationForest

AnalysisResult = dict  # {test, stat, effect_size, effect_size_label, p_value, n, summary_numbers}


def _is_normal(arr: np.ndarray, alpha: float = 0.05) -> bool:
    if len(arr) < 3:
        return False
    sample = arr[:5000]
    _, p = shapiro(sample)
    return bool(p >= alpha)


# ─────────────────────────────────────────────────────────────
# Correlation (numeric × numeric)
# ─────────────────────────────────────────────────────────────

def correlation(df: pd.DataFrame, col1: str, col2: str) -> Optional[AnalysisResult]:
    """Pearson or Spearman correlation between two numeric columns."""
    data = df[[col1, col2]].dropna()
    n = len(data)
    if n < 10:
        return None

    x = data[col1].values.astype(float)
    y = data[col2].values.astype(float)

    if _is_normal(x) and _is_normal(y):
        r, p = pearsonr(x, y)
        test = "pearson_correlation"
    else:
        r, p = spearmanr(x, y)
        test = "spearman_correlation"

    return {
        "test": test,
        "stat": float(r),
        "effect_size": float(abs(r)),
        "effect_size_label": "r",
        "p_value": float(p),
        "n": n,
        "summary_numbers": {
            "r": round(float(r), 4),
            f"mean_{col1}": round(float(x.mean()), 4),
            f"mean_{col2}": round(float(y.mean()), 4),
            f"std_{col1}": round(float(x.std()), 4),
            f"std_{col2}": round(float(y.std()), 4),
        },
    }


# ─────────────────────────────────────────────────────────────
# Group difference (numeric by categorical)
# ─────────────────────────────────────────────────────────────

def group_difference(df: pd.DataFrame, value_col: str, group_col: str) -> Optional[AnalysisResult]:
    """t-test / Mann-Whitney for 2 groups; ANOVA / Kruskal-Wallis for >2 groups."""
    groups: dict[str, np.ndarray] = {}
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].dropna().values.astype(float)
        if len(vals) >= 5:
            groups[str(g)] = vals

    if len(groups) < 2 or len(groups) > 30:
        return None

    n = sum(len(v) for v in groups.values())
    group_vals = list(groups.values())

    if len(groups) == 2:
        g1, g2 = group_vals
        if _is_normal(g1) and _is_normal(g2):
            t, p = ttest_ind(g1, g2)
            pooled_var = (
                (len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1)
            ) / (len(g1) + len(g2) - 2)
            pooled_std = float(np.sqrt(max(pooled_var, 1e-12)))
            d = (float(np.mean(g1)) - float(np.mean(g2))) / pooled_std
            effect_size, effect_label = float(abs(d)), "Cohen's d"
            test, stat = "independent_t_test", float(t)
        else:
            u, p = mannwhitneyu(g1, g2, alternative="two-sided")
            rbc = 1.0 - (2.0 * float(u)) / (len(g1) * len(g2))
            effect_size, effect_label = float(abs(rbc)), "rank-biserial r"
            test, stat = "mann_whitney_u", float(u)

        return {
            "test": test,
            "stat": stat,
            "effect_size": effect_size,
            "effect_size_label": effect_label,
            "p_value": float(p),
            "n": n,
            "summary_numbers": {
                "group_means": {k: round(float(np.mean(v)), 4) for k, v in groups.items()},
                "group_sizes": {k: len(v) for k, v in groups.items()},
            },
        }

    # >2 groups
    all_normal = all(_is_normal(g) for g in group_vals)
    if all_normal:
        f, p = f_oneway(*group_vals)
        grand_mean = float(np.concatenate(group_vals).mean())
        ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in group_vals)
        ss_total = sum(float(np.sum((g - grand_mean) ** 2)) for g in group_vals)
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        effect_size, effect_label, test, stat = float(eta2), "eta²", "one_way_anova", float(f)
    else:
        h, p = kruskal(*group_vals)
        eps2 = max(0.0, (float(h) - len(groups) + 1) / (n - len(groups)))
        effect_size, effect_label, test, stat = float(eps2), "epsilon²", "kruskal_wallis", float(h)

    return {
        "test": test,
        "stat": stat,
        "effect_size": effect_size,
        "effect_size_label": effect_label,
        "p_value": float(p),
        "n": n,
        "summary_numbers": {
            "group_means": {k: round(float(np.mean(v)), 4) for k, v in groups.items()},
            "group_sizes": {k: len(v) for k, v in groups.items()},
        },
    }


# ─────────────────────────────────────────────────────────────
# Association (categorical × categorical)
# ─────────────────────────────────────────────────────────────

def association(df: pd.DataFrame, col1: str, col2: str) -> Optional[AnalysisResult]:
    """Chi-square test of independence + Cramér's V."""
    data = df[[col1, col2]].dropna()
    # Limit cardinality
    top1 = data[col1].value_counts().nlargest(20).index
    top2 = data[col2].value_counts().nlargest(20).index
    data = data[data[col1].isin(top1) & data[col2].isin(top2)]
    n = len(data)
    if n < 10:
        return None

    ct = pd.crosstab(data[col1], data[col2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None

    chi2, p, dof, _ = chi2_contingency(ct)
    min_dim = min(ct.shape) - 1
    cramers_v = float(np.sqrt(float(chi2) / (n * max(min_dim, 1)))) if n > 0 else 0.0

    return {
        "test": "chi_square",
        "stat": float(chi2),
        "effect_size": cramers_v,
        "effect_size_label": "Cramér's V",
        "p_value": float(p),
        "n": n,
        "summary_numbers": {
            "cramers_v": round(cramers_v, 4),
            "dof": dof,
            "contingency_shape": list(ct.shape),
        },
    }


# ─────────────────────────────────────────────────────────────
# Trend (numeric over datetime)
# ─────────────────────────────────────────────────────────────

def trend(df: pd.DataFrame, value_col: str, datetime_col: str) -> Optional[AnalysisResult]:
    """Linear trend over time via OLS regression."""
    data = df[[datetime_col, value_col]].dropna().sort_values(datetime_col)
    n = len(data)
    if n < 10:
        return None

    y = data[value_col].values.astype(float)

    try:
        dt_col = pd.to_datetime(data[datetime_col], errors="coerce")
        t = (dt_col - dt_col.min()).dt.total_seconds().values.astype(float)
    except Exception:
        t = np.arange(n, dtype=float)

    if np.all(t == 0):
        t = np.arange(n, dtype=float)

    slope, intercept, r_value, p_value, _ = linregress(t, y)

    return {
        "test": "linear_regression_trend",
        "stat": float(slope),
        "effect_size": float(abs(r_value)),
        "effect_size_label": "r (trend strength)",
        "p_value": float(p_value),
        "n": n,
        "summary_numbers": {
            "slope": round(float(slope), 8),
            "r_squared": round(float(r_value ** 2), 4),
            "start_value": round(float(y[0]), 4),
            "end_value": round(float(y[-1]), 4),
            "direction": "increasing" if slope > 0 else "decreasing",
        },
    }


# ─────────────────────────────────────────────────────────────
# Anomaly (univariate IQR or multivariate IsolationForest)
# ─────────────────────────────────────────────────────────────

def anomaly(df: pd.DataFrame, cols: list[str]) -> Optional[AnalysisResult]:
    """IQR anomaly detection (univariate) or IsolationForest (multivariate)."""
    data = df[cols].dropna()
    n = len(data)
    if n < 10:
        return None

    if len(cols) == 1:
        col = cols[0]
        values = data[col].values.astype(float)
        q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        count = int(mask.sum())
        share = float(count / n)

        # Binomial test: expected ~0.7% outliers for normal data with 1.5*IQR
        btr = binomtest(count, n, p=0.007, alternative="greater")
        p_value = float(btr.pvalue) if count > 0 else 1.0

        return {
            "test": "iqr_anomaly",
            "stat": float(count),
            "effect_size": share,
            "effect_size_label": "anomaly share",
            "p_value": p_value,
            "n": n,
            "summary_numbers": {
                "anomaly_count": count,
                "anomaly_share": round(share, 4),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "iqr": round(iqr, 4),
            },
        }

    # Multivariate
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 2 or n < 30:
        return None

    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(numeric_data.values)
    count = int((preds == -1).sum())
    share = float(count / n)

    btr = binomtest(count, n, p=0.05, alternative="greater")
    p_value = float(btr.pvalue) if count > 0 else 1.0

    return {
        "test": "isolation_forest",
        "stat": float(count),
        "effect_size": share,
        "effect_size_label": "anomaly share",
        "p_value": p_value,
        "n": n,
        "summary_numbers": {
            "anomaly_count": count,
            "anomaly_share": round(share, 4),
            "columns_used": list(cols),
        },
    }


# ─────────────────────────────────────────────────────────────
# Missingness
# ─────────────────────────────────────────────────────────────

def missingness(df: pd.DataFrame, col: str) -> Optional[AnalysisResult]:
    """Check if missingness in a column correlates with other numeric columns."""
    if col not in df.columns:
        return None

    n = len(df)
    missing_mask = df[col].isna()
    miss_count = int(missing_mask.sum())
    if miss_count == 0:
        return None

    miss_share = float(miss_count / n)
    miss_indicator = missing_mask.astype(float)

    best_r, best_col, best_p = 0.0, None, 1.0
    for other_col in df.select_dtypes(include=[np.number]).columns:
        if other_col == col:
            continue
        pair = pd.DataFrame({"m": miss_indicator, "v": df[other_col]}).dropna()
        if len(pair) < 10:
            continue
        r, p = pearsonr(pair["m"].values, pair["v"].values)
        if abs(r) > abs(best_r):
            best_r, best_col, best_p = r, other_col, p

    effect_size = float(abs(best_r)) if best_col else miss_share
    p_value = float(best_p) if best_col else (0.01 if miss_share > 0.05 else 1.0)

    return {
        "test": "missingness_correlation",
        "stat": miss_share,
        "effect_size": effect_size,
        "effect_size_label": "r (missingness vs numeric col)" if best_col else "miss share",
        "p_value": p_value,
        "n": n,
        "summary_numbers": {
            "missing_count": miss_count,
            "missing_share": round(miss_share, 4),
            "most_correlated_with": best_col,
            "correlation_r": round(float(best_r), 4) if best_col else None,
        },
    }
