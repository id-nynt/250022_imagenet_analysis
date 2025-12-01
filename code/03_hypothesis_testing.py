"""
Hypothesis testing script for the report (H1–H3)

Uses existing artifacts to validate claims and export reproducible summaries.
- H1: Domain shift exists (per-feature KS/Wasserstein, KS p-values)
- H3: Importance–shift association (Pearson/Spearman)
- H2: Class difficulty varies (paired per-class comparison)

Outputs to results/hypothesis/ as CSV/JSON.
"""
from __future__ import annotations

import json
import math
import os # Import os module
from pathlib import Path
from typing import Dict, Any

import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr, ttest_rel, kstwo, ttest_ind, chi2
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Setup
BASE_DIR = Path("./")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "xgb_checkpoint.json"
META_PATH = OUTPUT_DIR / "xgb_meta.json"

SHIFT_CSV = OUTPUT_DIR / "feature_shift_analysis.csv"
IMP_CSV = OUTPUT_DIR / "feature_importances.csv"
CLASS_DELTA_CSV = OUTPUT_DIR / "class_accuracy_delta.csv"

# Define OUTDIR using OUTPUT_DIR
OUTDIR = Path(OUTPUT_DIR)


def summarize_h1_shift() -> Dict[str, Any]:
    """Summarize per-feature KS/Wasserstein and estimate KS p-values if SciPy is available."""
    shift_df = pd.read_csv(SHIFT_CSV)  # feature (e.g., 'f170'), ks_statistic, wasserstein_dist
    # Estimate KS p-values (two-sample) with reported sizes: Val=50,000; V2=10,000
    n1, n2 = 50_000, 10_000
    n_eff = (n1 * n2) / (n1 + n2)
    if HAVE_SCIPY:
        x = shift_df["ks_statistic"].astype(float) * math.sqrt(n_eff)
        # Asymptotic two-sample KS sf with dof ~ n_eff (rounded)
        ks_pvals = kstwo.sf(x, int(round(n_eff)))
        shift_df["ks_p_value"] = ks_pvals
    else:
        shift_df["ks_p_value"] = float("nan")

    # Save top-10 by KS with effect sizes
    top10 = shift_df.sort_values("ks_statistic", ascending=False).head(10).copy()
    top10.to_csv(OUTDIR / "H1_shift_top10.csv", index=False)

    # Aggregate summary
    summary = {
        "mean_ks": float(shift_df["ks_statistic"].mean()),
        "median_ks": float(shift_df["ks_statistic"].median()),
        "max_ks": float(shift_df["ks_statistic"].max()),
        "mean_wasserstein": float(shift_df["wasserstein_dist"].mean()),
    }
    if HAVE_SCIPY:
        for a in (1e-3, 1e-4, 1e-5):
            summary[f"ks_p_lt_{a}"] = int((shift_df["ks_p_value"] < a).sum())
    else:
        summary["ks_p_values"] = "SciPy not available"

    (OUTDIR / "H1_shift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize_h3_importance_vs_shift() -> Dict[str, Any]:
    """Compute correlation between feature importance and KS statistic."""
    shift_df = pd.read_csv(SHIFT_CSV)  # feature like 'f170'
    imp_df = pd.read_csv(IMP_CSV)      # has feature_index, feature_name, importance_score
    imp_df["feature"] = imp_df["feature_index"].apply(lambda i: f"f{i}")

    merged = pd.merge(shift_df, imp_df[["feature", "importance_score"]], on="feature", how="inner")
    ks = merged["ks_statistic"].astype(float)
    imp = merged["importance_score"].astype(float)

    out: Dict[str, Any] = {"merged_rows": int(len(merged))}

    if HAVE_SCIPY:
        r_p, p_p = pearsonr(ks, imp)
        r_s, p_s = spearmanr(ks, imp)
        out.update({
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_rho": float(r_s),
            "spearman_p": float(p_s),
        })
    else:
        # Fallback Pearson only (no p-value)
        def corr(a, b):
            a = a - a.mean(); b = b - b.mean()
            num = float((a * b).sum())
            den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
            return num / den if den else float("nan")
        out["pearson_r"] = float(corr(ks, imp))
        out["note"] = "SciPy not available for p-values"

    (OUTDIR / "H3_importance_shift_correlation.json").write_text(json.dumps(out, indent=2))
    return out


def summarize_h2_per_class() -> Dict[str, Any]:
    """Summarize per-class accuracy deltas and perform a paired test across classes."""
    if not CLASS_DELTA_CSV.exists():
        return {"error": f"Missing {CLASS_DELTA_CSV}"}
    df = pd.read_csv(CLASS_DELTA_CSV)
    # Expect columns: class, val_acc, test_acc, delta (robust handling)
    # Normalize possible column names
    cols = {c.lower(): c for c in df.columns}
    val_c = cols.get("val_acc") or cols.get("val_accuracy")
    test_c = cols.get("test_acc") or cols.get("test_accuracy")
    delta_c = cols.get("delta")

    out: Dict[str, Any] = {}
    if val_c and test_c:
        val = df[val_c].astype(float)
        tes = df[test_c].astype(float)
        deltas = val - tes
        out.update({
            "classes": int(len(df)),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(deltas.median()),
            "p90_delta": float(deltas.quantile(0.90)),
            "p10_delta": float(deltas.quantile(0.10)),
            "frac_delta_ge_0_2": float((deltas >= 0.2).mean()),
        })
        if HAVE_SCIPY:
            t_stat, p_val = ttest_rel(val, tes)
            out.update({"paired_t_stat": float(t_stat), "paired_p": float(p_val)})
        else:
            out["note"] = "SciPy not available for paired t-test"
    if delta_c:
        # Save top/bottom 10 by delta
        top10 = df.sort_values(cols["delta"], ascending=False).head(10)
        bot10 = df.sort_values(cols["delta"], ascending=True).head(10)
        top10.to_csv(OUTDIR / "H2_top10_val_minus_test.csv", index=False)
        bot10.to_csv(OUTDIR / "H2_top10_test_minus_val.csv", index=False)

    (OUTDIR / "H2_class_delta_summary.json").write_text(json.dumps(out, indent=2))
    return out


def _cohen_d_paired(x: pd.Series, y: pd.Series) -> float:
    """Cohen's d for paired samples using sd of differences (dz)."""
    diff = (x - y).astype(float)
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return 0.0
    return float(diff.mean()) / sd


def _cohen_d_independent(x: pd.Series, y: pd.Series) -> float:
    """Cohen's d for independent samples using pooled SD."""
    x = x.astype(float)
    y = y.astype(float)
    n1, n2 = len(x), len(y)
    s1, s2 = float(x.std(ddof=1)), float(y.std(ddof=1))
    if n1 + n2 - 2 <= 0:
        return float("nan")
    sp = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return float(x.mean() - y.mean()) / sp


def test_h1_overall_distribution_shift() -> Dict[str, Any]:
    """H1: Paired t-test on per-class accuracies (Val vs Test). One-tailed (Val > Test)."""
    if not CLASS_DELTA_CSV.exists():
        return {"error": f"Missing {CLASS_DELTA_CSV}"}
    df = pd.read_csv(CLASS_DELTA_CSV)
    cols = {c.lower(): c for c in df.columns}
    val_c = cols.get("val_acc") or cols.get("val_accuracy")
    test_c = cols.get("test_acc") or cols.get("test_accuracy")
    if not (val_c and test_c):
        return {"error": "Required columns val_acc/test_acc not found"}

    val = df[val_c].astype(float)
    tes = df[test_c].astype(float)
    diff = val - tes
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    n = len(diff)
    t_stat = mean_diff / (sd_diff / math.sqrt(n)) if sd_diff > 0 else float("inf")

    result = {
        "n_classes": int(n),
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "t_stat": t_stat,
        "tail": "one-sided (mean_diff > 0)",
        "cohen_d_paired_dz": _cohen_d_paired(val, tes),
    }
    if HAVE_SCIPY:
        t2, p2 = ttest_rel(val, tes)
        # Convert two-tailed p to one-tailed depending on direction
        if t2 >= 0:
            p1 = p2 / 2.0
        else:
            p1 = 1.0 - (p2 / 2.0)
        result.update({"t_stat_scipy": float(t2), "p_value_one_tailed": float(p1), "p_value_two_tailed": float(p2)})
    else:
        result["note"] = "SciPy not available for exact p-values"

    (OUTDIR / "H1_overall_ttest.json").write_text(json.dumps(result, indent=2))
    return result


def test_h2_class_specific_difficulty() -> Dict[str, Any]:
    """H2: Compare deltas for 'hard' vs 'easy' classes (split by median baseline/validation accuracy)."""
    if not CLASS_DELTA_CSV.exists():
        return {"error": f"Missing {CLASS_DELTA_CSV}"}
    df = pd.read_csv(CLASS_DELTA_CSV)
    cols = {c.lower(): c for c in df.columns}
    val_c = cols.get("val_acc") or cols.get("val_accuracy")
    test_c = cols.get("test_acc") or cols.get("test_accuracy")
    delta_c = cols.get("delta")
    if not (val_c and test_c):
        return {"error": "Required columns val_acc/test_acc not found"}

    val = df[val_c].astype(float)
    tes = df[test_c].astype(float)
    deltas = (val - tes)
    median_val = float(val.median())
    hard_mask = val <= median_val
    easy_mask = val > median_val
    hard = deltas[hard_mask]
    easy = deltas[easy_mask]

    res = {
        "median_val_baseline": median_val,
        "hard_n": int(hard.shape[0]),
        "easy_n": int(easy.shape[0]),
        "hard_mean_delta": float(hard.mean()),
        "easy_mean_delta": float(easy.mean()),
        "cohen_d": _cohen_d_independent(hard, easy),
    }
    if HAVE_SCIPY:
        t_stat, p_two = ttest_ind(hard, easy, equal_var=False)
        # One-sided: hard has larger negative deltas (i.e., greater drop). Since delta=val-test, larger positive means bigger drop.
        # If hypothesis is hard > easy (more positive), convert accordingly.
        if t_stat >= 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - (p_two / 2.0)
        res.update({"t_stat": float(t_stat), "p_value_two_tailed": float(p_two), "p_value_one_tailed": float(p_one)})
    else:
        res["note"] = "SciPy not available for p-values"

    (OUTDIR / "H2_hard_vs_easy.json").write_text(json.dumps(res, indent=2))
    return res


def test_h3_feature_level_instability(top_frac: float = 0.20) -> Dict[str, Any]:
    """H3: Compare KS statistics between high-importance (top_frac) and the rest."""
    shift_df = pd.read_csv(SHIFT_CSV)
    imp_df = pd.read_csv(IMP_CSV)
    imp_df["feature"] = imp_df["feature_index"].apply(lambda i: f"f{i}")
    merged = pd.merge(shift_df, imp_df[["feature", "importance_score"]], on="feature", how="inner")
    merged = merged.sort_values("importance_score", ascending=False).reset_index(drop=True)

    k = max(1, int(round(len(merged) * top_frac)))
    high = merged.iloc[:k]["ks_statistic"].astype(float)
    low = merged.iloc[k:]["ks_statistic"].astype(float)

    out = {
        "top_frac": top_frac,
        "high_n": int(len(high)),
        "low_n": int(len(low)),
        "high_mean_ks": float(high.mean()),
        "low_mean_ks": float(low.mean()),
        "cohen_d": _cohen_d_independent(high, low),
    }
    if HAVE_SCIPY:
        t_stat, p_two = ttest_ind(high, low, equal_var=False)
        # One-sided: high > low (greater divergence among high-importance features)
        if t_stat >= 0:
            p_one = p_two / 2.0
        else:
            p_one = 1.0 - (p_two / 2.0)
        out.update({"t_stat": float(t_stat), "p_value_two_tailed": float(p_two), "p_value_one_tailed": float(p_one)})
    else:
        out["note"] = "SciPy not available for p-values"

    (OUTDIR / "H3_feature_groups_ttest.json").write_text(json.dumps(out, indent=2))
    return out


# H4 removed per request


def test_mcnemar(baseline_vs_refined_csv: Path | None = None) -> Dict[str, Any]:
    """McNemar's test for paired predictions on the same samples.

    Expects CSV with two boolean/integer columns: baseline_correct, refined_correct (1 for correct, 0 for incorrect).
    Default path: results/mcnemar_inputs.csv
    """
    if baseline_vs_refined_csv is None:
        baseline_vs_refined_csv = OUTDIR / "mcnemar_inputs.csv"
    if not baseline_vs_refined_csv.exists():
        return {"error": f"Missing {baseline_vs_refined_csv}", "hint": "Create CSV with columns [baseline_correct, refined_correct] for Test set 2"}

    df = pd.read_csv(baseline_vs_refined_csv)
    cols = {c.lower(): c for c in df.columns}
    bcol = cols.get("baseline_correct")
    rcol = cols.get("refined_correct")
    if not (bcol and rcol):
        return {"error": "Columns baseline_correct/refined_correct not found"}
    b = df[bcol].astype(int)
    r = df[rcol].astype(int)
    # Contingency components
    a = int(((b == 1) & (r == 1)).sum())
    d = int(((b == 0) & (r == 0)).sum())
    b_only = int(((b == 1) & (r == 0)).sum())
    c_only = int(((b == 0) & (r == 1)).sum())

    # McNemar's chi-square (with and without continuity correction)
    if b_only + c_only == 0:
        chi2_nc = 0.0
        chi2_cc = 0.0
        p_nc = 1.0
        p_cc = 1.0
    else:
        chi2_nc = (b_only - c_only) ** 2 / (b_only + c_only)
        chi2_cc = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
        if HAVE_SCIPY:
            p_nc = float(chi2.sf(chi2_nc, 1))
            p_cc = float(chi2.sf(chi2_cc, 1))
        else:
            p_nc = float("nan")
            p_cc = float("nan")

    # Cohen's g for paired proportions
    g = (c_only - b_only) / (b_only + c_only) if (b_only + c_only) > 0 else 0.0

    out = {
        "a_both_correct": a,
        "b_baseline_only": b_only,
        "c_refined_only": c_only,
        "d_both_incorrect": d,
        "mcnemar_chi2_no_correction": float(chi2_nc),
        "mcnemar_chi2_continuity": float(chi2_cc),
        "p_value_no_correction": p_nc,
        "p_value_continuity": p_cc,
        "cohens_g": float(g),
    }
    (OUTDIR / "mcnemar_test.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    # H1: Domain shift exists (per-feature KS/Wasserstein, KS p-values)
    print("[H1] Domain shift summary (per-feature KS/Wasserstein)...")
    h1 = summarize_h1_shift()
    print(json.dumps(h1, indent=2))

    # H2: Class difficulty varies (paired per-class comparison and summary)
    print("\n[H2] Per-class accuracy deltas and paired test...")
    h2 = summarize_h2_per_class()
    print(json.dumps(h2, indent=2))

    # H3: Importance–shift association (Pearson/Spearman)
    print("\n[H3] Correlation between feature importance and KS shift...")
    h3 = summarize_h3_importance_vs_shift()
    print(json.dumps(h3, indent=2))

    print(f"\nArtifacts written to: {OUTDIR}")


if __name__ == "__main__":
    main()
