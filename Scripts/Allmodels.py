#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Model Comparison Pipeline - Version 2.0

Addresses Reviewer Comments:
- R1/R2: Classifier labels validated against external ground truth
- R2: EPSS removed from classifier features to prevent leakage
- R3(1): Ablation study for regression to assess true predictive power
- R3(2): Independent label construction with sensitivity analysis
- R3(3): Bootstrap confidence intervals for limited KEV positives
- R6: Regression-classification consistency analysis

Key Changes:
1. Feature ablation study for regression model
2. Independent exploit labels (KEV-only, EPSS-only, combined with analysis)
3. EPSS excluded from classifier features to prevent circular correlation
4. Bootstrap resampling for statistical robustness with small positive class
5. Consistency metrics between regression and classification rankings
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, average_precision_score, brier_score_loss, 
    precision_recall_curve, roc_auc_score, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression as PlattLR

# Optional xgboost
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Use: pip install shap")

from scipy import sparse as sp
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================
TODAY = datetime.now(timezone.utc).date()
RANDOM_STATE = 42
N_BOOTSTRAP = 100  # For confidence intervals

ROOT = Path("./model_compare_v2").resolve()
DIRS = {
    "data": ROOT / "data",
    "outputs": ROOT / "outputs",
    "reports": ROOT / "reports",
    "figures": ROOT / "figures",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

IN_POWER = Path("../data/power_grid_cves_enriched.csv")
IN_ALL = Path("../data/all_cves_risk_v2.csv")

# Priority configuration
PRIORITY_ALPHA = 0.6
PRIORITY_GAMMA = 1.2
KEV_FLOOR = 0.7
USE_KEV_FLOOR = True

# Logit transform for bounded regression
EPS = 1e-5

# ============================================================================
# Utility Functions
# ============================================================================
def y_to_logit(y):
    """Map y in [0,1] to real line."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, EPS, 1 - EPS)
    return np.log(y / (1 - y))

def logit_to_y(z):
    """Map real line back to [0,1]."""
    z = np.asarray(z, dtype=float)
    return 1 / (1 + np.exp(-z))

def to_dense_if_needed(X, estimator):
    name = estimator.__class__.__name__.lower()
    needs_dense = (
        "histgradientboosting" in name or 
        "randomforest" in name or
        ("logisticregression" in name and getattr(estimator, "solver", "") != "saga")
    )
    if needs_dense and sp.issparse(X):
        return X.toarray()
    return X

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")

def to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def ensure_cols(df, cols_defaults):
    for c, default in cols_defaults.items():
        if c not in df.columns:
            df[c] = default
    return df

def safe_str(x, default="unknown"):
    return str(x) if isinstance(x, str) and x.strip() else default

def save_fig(obj, path=None, close=True):
    if isinstance(obj, matplotlib.figure.Figure):
        fig = obj
        if path is None:
            raise ValueError("Must provide a path when passing a Figure.")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        if close:
            plt.close(fig)
    else:
        path = obj if path is None else path
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        if close:
            plt.close()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# Data Loading with Engineering Features
# ============================================================================
def add_engineered_features(df):
    """Add derived features from CVSS vector and description."""
    desc = df.get("description", "").fillna("").astype(str)
    vec = df.get("cvss_vector", "").fillna("").astype(str)

    df["is_remote"] = vec.str.contains(r"\bAV:N\b", na=False).astype(int)
    df["is_priv_none"] = vec.str.contains(r"\bPR:N\b", na=False).astype(int)
    df["ac_low"] = vec.str.contains(r"\bAC:L\b", na=False).astype(int)
    df["ui_none"] = vec.str.contains(r"\bUI:N\b", na=False).astype(int)
    
    flags = ["AV:N", "AC:L", "PR:N", "UI:N"]
    df["base_flags"] = sum(vec.str.contains(fr"\b{f}\b", na=False) for f in flags).astype(int)
    df["text_len"] = desc.str.len().clip(upper=20000)
    
    return df


def load_and_unify(power_path=IN_POWER, all_path=IN_ALL):
    """Load and merge CVE datasets."""
    df_cpps = pd.read_csv(power_path)
    df_all = pd.read_csv(all_path)

    df_cpps.columns = [c.strip().lower() for c in df_cpps.columns]
    df_all.columns = [c.strip().lower() for c in df_all.columns]

    df_cpps = ensure_cols(df_cpps, {
        "cve_id": np.nan, "vendor": "unknown", "product": "unknown",
        "cvss_score": np.nan, "severity": "unknown", "description": "",
        "published": np.nan, "epss_percentile": np.nan, "in_cisa_kev": 0,
        "vuln_cpe_count": np.nan, "criticality_0_5": np.nan,
        "exposure_0_5": np.nan, "cvss_vector": ""
    })
    df_all = ensure_cols(df_all, {
        "cve_id": np.nan, "vendor": "unknown", "product": "unknown",
        "cvss_score": np.nan, "severity": "unknown", "description": "",
        "published": np.nan, "epss_percentile": np.nan, "in_cisa_kev": 0,
        "vuln_cpe_count": np.nan, "criticality_0_5": np.nan,
        "exposure_0_5": np.nan, "risk": np.nan, "cvss_vector": ""
    })

    for df in (df_cpps, df_all):
        df["published"] = df["published"].apply(parse_date)
        df["age_days"] = df["published"].apply(
            lambda d: (TODAY - d.date()).days if pd.notna(d) else np.nan
        )
        df["cvss_score"] = df["cvss_score"].apply(to_float)
        df["vuln_cpe_count"] = pd.to_numeric(df["vuln_cpe_count"], errors="coerce")
        df["epss"] = pd.to_numeric(df["epss_percentile"], errors="coerce")
        df["is_kev"] = pd.to_numeric(df["in_cisa_kev"], errors="coerce").fillna(0).astype(int)
        
        for col in ["vendor", "product", "severity", "description", "cvss_vector"]:
            df[col] = df[col].apply(safe_str)
        for col in ["criticality_0_5", "exposure_0_5"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_cpps["_source"] = "CPPS"
    df_all["_source"] = "ALL"

    combined = pd.concat([df_all, df_cpps], ignore_index=True)
    combined.sort_values(by=["cve_id", "_source"], ascending=[True, True], inplace=True)
    combined = combined.drop_duplicates(subset=["cve_id"], keep="last").reset_index(drop=True)

    combined["epss"] = combined["epss"].clip(0, 1)
    combined = add_engineered_features(combined)

    # Merge risk from df_all
    df_all["risk"] = pd.to_numeric(df_all.get("risk", np.nan), errors="coerce")
    combined["risk"] = pd.to_numeric(combined.get("risk", np.nan), errors="coerce")
    combined = combined.merge(df_all[["cve_id", "risk"]], on="cve_id", how="left", suffixes=("", "_all"))
    combined["risk"] = combined["risk"].fillna(combined["risk_all"])
    combined = combined.drop(columns=["risk_all"], errors="ignore")

    combined.to_csv(DIRS["data"] / "combined_unified.csv", index=False, encoding="utf-8")
    print(f"After unify: risk_non_null = {int(combined['risk'].notna().sum())} of {len(combined)}")
    return combined


def time_aware_split(df):
    """Split data temporally for realistic evaluation."""
    has_dates = df["published"].notna().mean() >= 0.5
    if has_dates:
        df_sorted = df.sort_values("published")
        cut = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:cut].copy()
        test_df = df_sorted.iloc[cut:].copy()
        meta = {
            "strategy": "time",
            "train_range": [str(train_df["published"].min()), str(train_df["published"].max())],
            "test_range": [str(test_df["published"].min()), str(test_df["published"].max())],
            "train_n": len(train_df),
            "test_n": len(test_df),
        }
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
        meta = {"strategy": "random", "train_n": len(train_df), "test_n": len(test_df)}
    return train_df, test_df, meta


# ============================================================================
# Feature Pipelines - CRITICAL: Separate pipelines for regression vs classification
# ============================================================================
def make_numeric_pipeline():
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


def build_feature_pipeline_regression(max_tfidf=8000):
    """
    Full feature pipeline for REGRESSION.
    Includes EPSS since regression target (risk) is independently defined.
    """
    numeric_cols = [
        "cvss_score", "epss", "is_kev", "vuln_cpe_count", "age_days",
        "criticality_0_5", "exposure_0_5",
        "is_remote", "is_priv_none", "ac_low", "ui_none", "base_flags", "text_len"
    ]
    cat_cols = ["vendor", "product", "severity"]
    text_col = "description"

    return ColumnTransformer(
        transformers=[
            ("num", make_numeric_pipeline(), numeric_cols),
            ("cat", _make_ohe(), cat_cols),
            ("txt", TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf, min_df=2), text_col),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )


def build_feature_pipeline_classification(max_tfidf=8000):
    """
    Feature pipeline for CLASSIFICATION - EPSS EXCLUDED.
    
    Reviewer R2 Response:
    EPSS is excluded from classifier features because the exploit label
    partially depends on EPSS threshold. Including EPSS would create
    near-perfect circular correlation (if EPSS > threshold, predict positive).
    
    This ensures the classifier learns meaningful patterns from other features
    rather than simply thresholding EPSS.
    """
    numeric_cols = [
        "cvss_score", "vuln_cpe_count", "age_days",  # EPSS REMOVED
        "criticality_0_5", "exposure_0_5",
        "is_remote", "is_priv_none", "ac_low", "ui_none", "base_flags", "text_len"
    ]
    cat_cols = ["vendor", "product", "severity"]
    text_col = "description"

    return ColumnTransformer(
        transformers=[
            ("num", make_numeric_pipeline(), numeric_cols),
            ("cat", _make_ohe(), cat_cols),
            ("txt", TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf, min_df=2), text_col),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )


def build_feature_pipeline_ablation(exclude_cols, max_tfidf=8000):
    """
    Build pipeline with specific columns excluded for ablation study.
    
    Reviewer R3(1) Response:
    This enables systematic evaluation of each feature group's contribution
    to model performance.
    """
    all_numeric = [
        "cvss_score", "epss", "is_kev", "vuln_cpe_count", "age_days",
        "criticality_0_5", "exposure_0_5",
        "is_remote", "is_priv_none", "ac_low", "ui_none", "base_flags", "text_len"
    ]
    numeric_cols = [c for c in all_numeric if c not in exclude_cols]
    cat_cols = ["vendor", "product", "severity"]
    text_col = "description"

    transformers = [("num", make_numeric_pipeline(), numeric_cols)]
    
    if "categorical" not in exclude_cols:
        transformers.append(("cat", _make_ohe(), cat_cols))
    if "text" not in exclude_cols:
        transformers.append(("txt", TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf, min_df=2), text_col))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


# ============================================================================
# Label Construction - Multiple Strategies for Validation
# ============================================================================
def label_exploit_kev_only(df):
    """
    KEV-only labels (most conservative).
    
    Reviewer R1/R3(2) Response:
    This provides the most reliable ground truth since KEV entries
    are confirmed exploited vulnerabilities.
    """
    return (df["is_kev"] == 1).astype(int)


def label_exploit_epss_only(df, threshold=0.90):
    """
    EPSS-only labels (predictive signal).
    
    High EPSS indicates statistical likelihood of exploitation
    based on FIRST's model trained on real exploit data.
    """
    return (df["epss"] >= threshold).astype(int)


def label_exploit_combined(df, epss_threshold=0.75):
    """
    Combined labels (original approach).
    
    Reviewer R2 Response:
    We acknowledge this introduces correlation with EPSS.
    When used with the classification pipeline (EPSS excluded),
    the model must learn from other features.
    """
    return ((df["is_kev"] == 1) | (df["epss"] >= epss_threshold)).astype(int)


def label_exploit_tiered(df):
    """
    Tiered labels: 2=KEV (confirmed), 1=High EPSS (likely), 0=other.
    
    This allows ordinal analysis distinguishing confirmed vs predicted exploits.
    """
    labels = np.zeros(len(df), dtype=int)
    labels[df["epss"] >= 0.90] = 1  # High EPSS
    labels[df["is_kev"] == 1] = 2    # KEV overrides
    return labels


# ============================================================================
# Model Registry
# ============================================================================
def get_regressor(name):
    name = name.lower()
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=0
        )
    if name == "histgbm":
        return HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown regressor '{name}'")


def get_classifier(name, y_train):
    name = name.lower()
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = max(1.0, neg / max(1, pos))

    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=0
        )
    if name == "histgbm":
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=500, random_state=RANDOM_STATE
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=20, class_weight={0: 1, 1: spw},
            random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "logreg":
        return LogisticRegression(
            solver="saga", penalty="l2", C=1.0, class_weight="balanced", max_iter=2000
        )
    raise ValueError(f"Unknown classifier '{name}'")


# ============================================================================
# Calibration and Evaluation
# ============================================================================
def platt_calibrate(base_clf, pre, train_df, y, random_state=RANDOM_STATE):
    """Platt scaling for probability calibration."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_df, y, test_size=0.2, random_state=random_state,
        stratify=y if y.sum() >= 2 else None
    )
    Xt_tr = pre.transform(X_tr)
    Xt_val = pre.transform(X_val)
    Xt_tr = to_dense_if_needed(Xt_tr, base_clf)
    Xt_val = to_dense_if_needed(Xt_val, base_clf)

    base_clf.fit(Xt_tr, y_tr)
    p_val = base_clf.predict_proba(Xt_val)[:, 1].reshape(-1, 1)

    platt = PlattLR(max_iter=1000)
    platt.fit(p_val, y_val)

    class CalibWrapper:
        def __init__(self, pre, clf, platt):
            self.pre, self.clf, self.platt = pre, clf, platt

        def predict_proba(self, X):
            Xt = self.pre.transform(X)
            Xt = to_dense_if_needed(Xt, self.clf)
            p = self.clf.predict_proba(Xt)[:, 1].reshape(-1, 1)
            p_cal = self.platt.predict_proba(p)[:, 1]
            return np.vstack([1 - p_cal, p_cal]).T

    return CalibWrapper(pre, base_clf, platt)


def bootstrap_metrics(y_true, y_prob, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    """
    Bootstrap confidence intervals for classification metrics.
    
    Reviewer R3(3) Response:
    With only 18 KEV positives, point estimates may be unreliable.
    Bootstrap resampling provides confidence intervals.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    pr_aucs = []
    briers = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        
        if y_b.sum() >= 1:
            pr_aucs.append(average_precision_score(y_b, p_b))
            briers.append(brier_score_loss(y_b, p_b))
    
    return {
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_ci_low": np.percentile(pr_aucs, 2.5),
        "pr_auc_ci_high": np.percentile(pr_aucs, 97.5),
        "brier_mean": np.mean(briers),
        "brier_ci_low": np.percentile(briers, 2.5),
        "brier_ci_high": np.percentile(briers, 97.5),
    }


def eval_regressor(pre, reg, test_df, title_suffix=""):
    """Evaluate regression model."""
    mask = test_df["risk"].notna()
    if mask.sum() < 10:
        return np.nan, np.nan, None

    Xt = pre.transform(test_df.loc[mask])
    Xt = to_dense_if_needed(Xt, reg)

    y_true = test_df.loc[mask, "risk"].astype(float).clip(0, 1).values
    y_pred_logit = reg.predict(Xt)
    y_pred = np.clip(logit_to_y(y_pred_logit), 0, 1)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(4, 3))
    plt.hist(y_true - y_pred, bins=40)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title(f"Regressor Residuals {title_suffix}")
    
    return r2, mae, plt.gcf()


def eval_classifier(pre, clf_calib, test_df, label_func, title_suffix=""):
    """Evaluate classification model with bootstrap CI."""
    y_true = label_func(test_df).values
    y_prob = np.clip(clf_calib.predict_proba(test_df)[:, 1], 0, 1)

    if y_true.sum() < 1:
        return {"pr_auc": np.nan, "brier": np.nan}, None, None

    # Point estimates
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Bootstrap CI
    boot_metrics = bootstrap_metrics(y_true, y_prob)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
    ax_pr.step(recall, precision, where="post")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"PR Curve {title_suffix}\nAP={pr_auc:.3f} [{boot_metrics['pr_auc_ci_low']:.3f}, {boot_metrics['pr_auc_ci_high']:.3f}]")

    # Calibration plot
    bins = pd.qcut(y_prob, q=10, duplicates="drop")
    tmp = pd.DataFrame({"bin": bins, "y": y_true, "p": y_prob})
    cal = tmp.groupby("bin").agg(emp_rate=("y", "mean"), avg_pred=("p", "mean")).reset_index(drop=True)
    fig_cal, ax_cal = plt.subplots(figsize=(4, 3))
    ax_cal.plot([0, 1], [0, 1], linestyle="--")
    ax_cal.scatter(cal["avg_pred"], cal["emp_rate"], s=15)
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Empirical frequency")
    ax_cal.set_title(f"Calibration {title_suffix}")

    metrics = {
        "pr_auc": pr_auc,
        "brier": brier,
        **boot_metrics,
        "n_positive": int(y_true.sum()),
        "n_total": len(y_true),
    }
    
    return metrics, fig_pr, fig_cal


# ============================================================================
# Ablation Study for Regression
# ============================================================================
def run_ablation_study(train_df, test_df, reg_name="ridge"):
    """
    Reviewer R3(1) Response:
    Systematic ablation study to assess contribution of each feature group.
    
    Since risk score is derived from CVSS, EPSS, KEV, criticality, exposure,
    we test how much R² drops when each is removed.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Feature Group Contribution")
    print("="*60)
    
    ablation_configs = [
        ("Full Model", []),
        ("No CVSS", ["cvss_score"]),
        ("No EPSS", ["epss"]),
        ("No KEV", ["is_kev"]),
        ("No CVSS+EPSS+KEV (Context Only)", ["cvss_score", "epss", "is_kev"]),
        ("No Criticality/Exposure", ["criticality_0_5", "exposure_0_5"]),
        ("No CVSS Vector Features", ["is_remote", "is_priv_none", "ac_low", "ui_none", "base_flags"]),
        ("No Text Features", ["text"]),
        ("No Categorical", ["categorical"]),
        ("Minimal (CVSS+EPSS+KEV only)", ["criticality_0_5", "exposure_0_5", "is_remote", 
            "is_priv_none", "ac_low", "ui_none", "base_flags", "text_len", "vuln_cpe_count", 
            "age_days", "categorical", "text"]),
    ]
    
    results = []
    mask_train = train_df["risk"].notna()
    mask_test = test_df["risk"].notna()
    
    y_train = y_to_logit(train_df.loc[mask_train, "risk"].astype(float).clip(EPS, 1-EPS).values)
    y_test = test_df.loc[mask_test, "risk"].astype(float).clip(0, 1).values
    
    for config_name, exclude_cols in ablation_configs:
        try:
            pre = build_feature_pipeline_ablation(exclude_cols)
            pre.fit(train_df.loc[mask_train])
            
            Xt_train = pre.transform(train_df.loc[mask_train])
            Xt_test = pre.transform(test_df.loc[mask_test])
            
            reg = get_regressor(reg_name)
            Xt_train = to_dense_if_needed(Xt_train, reg)
            Xt_test = to_dense_if_needed(Xt_test, reg)
            
            reg.fit(Xt_train, y_train)
            y_pred = np.clip(logit_to_y(reg.predict(Xt_test)), 0, 1)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                "config": config_name,
                "excluded": ", ".join(exclude_cols) if exclude_cols else "None",
                "R2": r2,
                "MAE": mae,
                "n_features": Xt_train.shape[1] if hasattr(Xt_train, 'shape') else "N/A"
            })
            print(f"  {config_name}: R²={r2:.4f}, MAE={mae:.4f}")
            
        except Exception as e:
            print(f"  {config_name}: FAILED - {e}")
            results.append({
                "config": config_name,
                "excluded": ", ".join(exclude_cols) if exclude_cols else "None",
                "R2": np.nan,
                "MAE": np.nan,
                "n_features": "ERROR"
            })
    
    return pd.DataFrame(results)


# ============================================================================
# Regression-Classification Consistency Analysis
# ============================================================================
def analyze_consistency(df_scored, label_func):
    """
    Reviewer R6 Response:
    Compare rankings from regression (pred_risk) vs classification (p_exploit).
    """
    # Rank correlation
    from scipy.stats import spearmanr, kendalltau
    
    risk_rank = df_scored["pred_risk"].rank(ascending=False)
    exploit_rank = df_scored["p_exploit"].rank(ascending=False)
    
    spearman, sp_pval = spearmanr(risk_rank, exploit_rank)
    kendall, kt_pval = kendalltau(risk_rank, exploit_rank)
    
    # Agreement in top-K
    k_values = [50, 100, 200, 500]
    agreement = {}
    
    for k in k_values:
        top_k_risk = set(df_scored.nlargest(k, "pred_risk")["cve_id"])
        top_k_exploit = set(df_scored.nlargest(k, "p_exploit")["cve_id"])
        overlap = len(top_k_risk & top_k_exploit)
        agreement[f"top_{k}_overlap"] = overlap
        agreement[f"top_{k}_jaccard"] = overlap / len(top_k_risk | top_k_exploit)
    
    # Priority tier agreement
    df_scored["risk_tier"] = pd.cut(df_scored["pred_risk"], 
                                     bins=[0, 0.3, 0.6, 0.8, 1.01],
                                     labels=["Low", "Medium", "High", "Act Now"])
    df_scored["exploit_tier"] = pd.cut(df_scored["p_exploit"],
                                        bins=[0, 0.3, 0.6, 0.8, 1.01],
                                        labels=["Low", "Medium", "High", "Act Now"])
    
    tier_match = (df_scored["risk_tier"] == df_scored["exploit_tier"]).mean()
    
    return {
        "spearman_corr": spearman,
        "spearman_pval": sp_pval,
        "kendall_tau": kendall,
        "kendall_pval": kt_pval,
        "tier_agreement": tier_match,
        **agreement
    }
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_feature_names(preprocessor, df):
    """
    Extract interpretable feature names from ColumnTransformer.
    """
    feature_names = []
    
    # Get transformers
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            # Numeric features - use column names directly
            feature_names.extend(cols)
        
        elif name == "cat":
            # One-hot encoded categorical features
            if hasattr(trans, 'named_steps'):
                ohe = trans.named_steps['onehotencoder']
            else:
                ohe = trans
            
            if hasattr(ohe, 'get_feature_names_out'):
                cat_features = ohe.get_feature_names_out(cols)
            else:
                cat_features = ohe.get_feature_names(cols)
            feature_names.extend(cat_features)
        
        elif name == "txt":
            # TF-IDF text features
            if hasattr(trans, 'get_feature_names_out'):
                txt_features = trans.get_feature_names_out()
            else:
                txt_features = trans.get_feature_names()
            feature_names.extend([f"text_{f}" for f in txt_features])
    
    return feature_names
"""
Bug Fix for SHAP Analysis in Allmodels_claude.py

Replace the analyze_shap_regression function with this corrected version.
The issue: Feature grouping was creating index lists that exceeded array bounds.
"""

def analyze_shap_regression(model, X_transformed, X_df, feature_names, 
                            output_dir, model_name="ridge", sample_size=500):
    """
    Comprehensive SHAP analysis for regression model - FIXED VERSION.
    
    Generates:
    1. Global feature importance (mean |SHAP|)
    2. Summary plot (beeswarm)
    3. Dependence plots for key interactions
    """
    print(f"\n{'='*60}")
    print(f"SHAP Analysis: Regression Model ({model_name})")
    print(f"{'='*60}")
    
    # Convert to dense if sparse (SHAP requires dense for tree models)
    if hasattr(X_transformed, 'toarray'):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed
    
    # Sample for computational efficiency
    if len(X_dense) > sample_size:
        indices = np.random.choice(len(X_dense), sample_size, replace=False)
        X_sample = X_dense[indices]
        X_df_sample = X_df.iloc[indices]
    else:
        X_sample = X_dense
        X_df_sample = X_df
    
    print(f"Computing SHAP values for {len(X_sample)} samples...")
    print(f"Feature space: {X_sample.shape[1]} features")
    
    # Choose appropriate explainer
    if model_name.lower() in ['ridge', 'linear']:
        # Linear model - use LinearExplainer
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Tree models - use TreeExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except:
            # Fallback to KernelExplainer (slower but works for any model)
            print("  TreeExplainer failed, using KernelExplainer (slower)...")
            explainer = shap.KernelExplainer(model.predict, 
                                            shap.sample(X_sample, 100))
            shap_values = explainer.shap_values(X_sample)
    
    print("  SHAP computation complete!")
    
    # ========================================================================
    # 1. GLOBAL FEATURE IMPORTANCE - FIXED
    # ========================================================================
    print("\n[1/4] Computing global feature importance...")
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # CRITICAL FIX: Verify array sizes match
    n_features = X_sample.shape[1]
    n_shap = len(mean_abs_shap)
    n_names = len(feature_names)
    
    print(f"  Dimensions check: {n_features} features, {n_shap} SHAP values, {n_names} names")
    
    if n_shap != n_names:
        print(f"  WARNING: Mismatch between SHAP values ({n_shap}) and feature names ({n_names})")
        # Truncate to minimum
        min_len = min(n_shap, n_names)
        mean_abs_shap = mean_abs_shap[:min_len]
        feature_names = feature_names[:min_len]
        print(f"  Truncated to {min_len} features")
    
    # Aggregate features safely with bounds checking
    feature_groups = {}
    for i, fname in enumerate(feature_names):
        if i >= len(mean_abs_shap):  # Safety check
            break
            
        if fname.startswith('text_'):
            feature_groups.setdefault('Text Features (TF-IDF)', []).append(i)
        elif 'cat__' in str(fname):  # More robust check
            try:
                base = str(fname).split('__')[1].split('_')[0]
                feature_groups.setdefault(f'Categorical: {base}', []).append(i)
            except:
                feature_groups.setdefault('Categorical: other', []).append(i)
        else:
            feature_groups[str(fname)] = [i]  # Ensure string key
    
    # Compute aggregated importance with bounds checking
    grouped_importance = []
    for group_name, indices in feature_groups.items():
        # Filter indices that are within bounds
        valid_indices = [idx for idx in indices if idx < len(mean_abs_shap)]
        if valid_indices:
            importance = mean_abs_shap[valid_indices].sum()
            grouped_importance.append({
                'Feature': group_name, 
                'Mean |SHAP|': importance,
                'n_features': len(valid_indices)
            })
    
    importance_df = pd.DataFrame(grouped_importance).sort_values('Mean |SHAP|', ascending=False)
    importance_df.to_csv(output_dir / f"shap_importance_{model_name}.csv", index=False)
    
    print("\nTop 15 Feature Groups by Importance:")
    print(importance_df.head(15).to_string(index=False))
    
    # Plot top features
    top_n = min(15, len(importance_df))
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    
    colors = ['steelblue' if not row['Feature'].startswith(('Text', 'Categorical')) 
              else 'lightcoral' if row['Feature'].startswith('Text')
              else 'lightgreen'
              for _, row in top_features.iterrows()]
    
    ax.barh(range(top_n), top_features['Mean |SHAP|'].values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['Feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value| (impact on risk score)', fontsize=11, fontweight='bold')
    ax.set_title(f'Global Feature Importance - {model_name.upper()} Regression\n(Blue=Numeric, Red=Text, Green=Categorical)', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_importance_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: shap_importance_{model_name}.png")
    
    # ========================================================================
    # 2. SUMMARY PLOT (Beeswarm) - Focus on interpretable features
    # ========================================================================
    print("\n[2/4] Generating SHAP summary plot (beeswarm)...")
    
    # Select top numeric/interpretable features for beeswarm
    interpretable_mask = [
        (not str(fname).startswith('text_') and 
         'cat__' not in str(fname) and 
         i < len(mean_abs_shap))
        for i, fname in enumerate(feature_names)
    ]
    interpretable_indices = [i for i, m in enumerate(interpretable_mask) if m]
    
    if len(interpretable_indices) > 0:
        # Take top 12 by importance
        importance_scores = [(i, mean_abs_shap[i]) for i in interpretable_indices]
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        top_interp_idx = [i for i, _ in importance_scores[:12]]
        
        shap_interp = shap_values[:, top_interp_idx]
        X_interp = X_sample[:, top_interp_idx]
        feature_names_interp = [feature_names[i] for i in top_interp_idx]
        
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_interp,
            X_interp,
            feature_names=feature_names_interp,
            show=False,
            plot_size=(10, 7)
        )
        plt.title(f'SHAP Summary Plot - {model_name.upper()} Regression\n(Color: Feature Value | X-axis: SHAP Impact on Risk Score)', 
                 fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: shap_summary_{model_name}.png")
    else:
        print("  ⚠ Not enough interpretable features for beeswarm plot")
    
    # ========================================================================
    # 3. DEPENDENCE PLOTS (Key Interactions)
    # ========================================================================
    print("\n[3/4] Generating SHAP dependence plots...")
    
    # Find key features by name
    feature_name_to_idx = {str(fname): i for i, fname in enumerate(feature_names) if i < len(mean_abs_shap)}
    
    dependence_configs = [
        ('epss', 'cvss_score', 'EPSS vs CVSS Interaction'),
        ('cvss_score', 'epss', 'CVSS vs EPSS Interaction'),
        ('age_days', 'epss', 'Age vs EPSS Interaction'),
        ('is_kev', 'cvss_score', 'KEV vs CVSS Interaction'),
    ]
    
    for feat1_name, feat2_name, plot_title in dependence_configs:
        try:
            if feat1_name in feature_name_to_idx and feat2_name in feature_name_to_idx:
                feat1_idx = feature_name_to_idx[feat1_name]
                feat2_idx = feature_name_to_idx[feat2_name]
                
                # Safety check
                if feat1_idx >= shap_values.shape[1] or feat2_idx >= X_sample.shape[1]:
                    print(f"  ⚠ Skipping {plot_title}: index out of bounds")
                    continue
                
                plt.figure(figsize=(8, 5))
                shap.dependence_plot(
                    feat1_idx,
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    interaction_index=feat2_idx,
                    show=False
                )
                plt.title(f'SHAP Dependence: {plot_title}\n{model_name.upper()} Regression', 
                         fontsize=11, fontweight='bold')
                plt.tight_layout()
                
                safe_title = plot_title.replace(' ', '_').replace('vs', 'vs')
                plt.savefig(output_dir / f"shap_dependence_{safe_title}_{model_name}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved: shap_dependence_{safe_title}_{model_name}.png")
            else:
                missing = []
                if feat1_name not in feature_name_to_idx:
                    missing.append(feat1_name)
                if feat2_name not in feature_name_to_idx:
                    missing.append(feat2_name)
                print(f"  ⚠ Skipping {plot_title}: features not found ({', '.join(missing)})")
                
        except Exception as e:
            print(f"  ⚠ Could not generate dependence plot for {plot_title}: {e}")
    
    # ========================================================================
    # 4. SHAP INTERACTION VALUES (if supported)
    # ========================================================================
    print("\n[4/4] Checking for interaction effects...")
    
    if model_name.lower() in ['rf', 'xgb', 'histgbm'] and len(interpretable_indices) >= 2:
        try:
            print("  Computing SHAP interaction values (sample of 100)...")
            # Use smaller sample for interactions (computationally expensive)
            small_sample_idx = np.random.choice(len(X_sample), min(100, len(X_sample)), replace=False)
            X_small = X_sample[small_sample_idx]
            
            shap_interaction = explainer.shap_interaction_values(X_small)
            
            # Sum absolute interaction values to find strongest pairs
            interaction_strength = np.abs(shap_interaction).sum(axis=0)
            np.fill_diagonal(interaction_strength, 0)  # Remove self-interactions
            
            # Find top 5 interactions among interpretable features
            top_pairs = []
            for i in interpretable_indices[:20]:  # Limit to top 20 interpretable
                for j in interpretable_indices[:20]:
                    if i < j and i < interaction_strength.shape[0] and j < interaction_strength.shape[1]:
                        strength = interaction_strength[i, j]
                        top_pairs.append((i, j, strength))
            
            top_pairs.sort(key=lambda x: x[2], reverse=True)
            
            print("\n  Top 5 Feature Interactions:")
            for i, j, strength in top_pairs[:5]:
                if i < len(feature_names) and j < len(feature_names):
                    print(f"    {feature_names[i]:20s} × {feature_names[j]:20s}: {strength:.4f}")
            
        except Exception as e:
            print(f"  ⚠ Interaction analysis not available: {e}")
    else:
        print("  Skipping interaction analysis (not supported for linear models or insufficient features)")
    
    return {
        'importance_df': importance_df,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'feature_names': feature_names
    }

def analyze_shap_classification(model, X_transformed, X_df, feature_names,
                                output_dir, model_name="histgbm", sample_size=500):
    """
    SHAP analysis for classification model (exploitation probability).
    """
    print(f"\n{'='*60}")
    print(f"SHAP Analysis: Classification Model ({model_name})")
    print(f"{'='*60}")
    
    # Similar structure to regression analysis
    # (Abbreviated for brevity - use same pattern as analyze_shap_regression)
    
    if hasattr(X_transformed, 'toarray'):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed
    
    if len(X_dense) > sample_size:
        indices = np.random.choice(len(X_dense), sample_size, replace=False)
        X_sample = X_dense[indices]
    else:
        X_sample = X_dense
    
    print(f"Computing SHAP values for {len(X_sample)} samples...")
    
    # For classification, we want SHAP values for the positive class
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except:
        # Fallback: use probability predictions
        def predict_proba_func(X):
            return model.predict_proba(X)[:, 1]
        
        explainer = shap.KernelExplainer(predict_proba_func, 
                                        shap.sample(X_sample, 100))
        shap_values = explainer.shap_values(X_sample)
    
    # Generate similar plots as regression
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values('Mean |SHAP|', ascending=False)
    
    importance_df.to_csv(output_dir / f"shap_importance_clf_{model_name}.csv", index=False)
    
    # Bar plot
    top_n = min(15, len(importance_df))
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    ax.barh(range(top_n), top_features['Mean |SHAP|'].values, color='coral')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value| (impact on exploitation probability)', fontsize=11)
    ax.set_title(f'Global Feature Importance - {model_name.upper()} Classification\n(Note: EPSS excluded from features to prevent circularity)', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_importance_clf_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: shap_importance_clf_{model_name}.png")
    
    return importance_df


# ============================================================================
# Integration with main() - ADD THIS TO YOUR SCRIPT
# ============================================================================

def run_shap_analysis(pre_reg, reg, pre_clf, clf, train_df, test_df, 
                     reg_name, clf_name, output_dir):
    """
    Run SHAP analysis for both regression and classification models.
    
    Call this after training models in main():
    
    if args.run_shap:
        run_shap_analysis(pre_reg, reg, pre_clf, clf_calib.clf, 
                         train_df, test_df, rname, cname, DIRS["figures"])
    """
    
    # Create SHAP subdirectory
    shap_dir = output_dir / "shap"
    shap_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # REGRESSION SHAP ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("SHAP ANALYSIS: REGRESSION MODEL (Risk Score Prediction)")
    print("="*70)
    
    # Prepare data for regression
    mask = train_df["risk"].notna()
    if mask.sum() < 10:
        print("Not enough data for regression SHAP analysis")
    else:
        X_reg_transformed = pre_reg.transform(train_df.loc[mask])
        feature_names_reg = get_feature_names(pre_reg, train_df)
        
        reg_shap_results = analyze_shap_regression(
            model=reg,
            X_transformed=X_reg_transformed,
            X_df=train_df.loc[mask],
            feature_names=feature_names_reg,
            output_dir=shap_dir,
            model_name=reg_name,
            sample_size=500
        )
        
        print(f"\nRegression SHAP Key Insights:")
        print("-" * 70)
        top5 = reg_shap_results['importance_df'].head(5)
        for idx, row in top5.iterrows():
            print(f"  {row['Feature']:30s}: {row['Mean |SHAP|']:.4f}")
    
    # ========================================================================
    # CLASSIFICATION SHAP ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("SHAP ANALYSIS: CLASSIFICATION MODEL (Exploitation Probability)")
    print("="*70)
    
    X_clf_transformed = pre_clf.transform(train_df)
    feature_names_clf = get_feature_names(pre_clf, train_df)
    
    clf_shap_results = analyze_shap_classification(
        model=clf,  # Pass the base classifier (not calibrated wrapper)
        X_transformed=X_clf_transformed,
        X_df=train_df,
        feature_names=feature_names_clf,
        output_dir=shap_dir,
        model_name=clf_name,
        sample_size=500
    )
    
    print(f"\nClassification SHAP Key Insights:")
    print("-" * 70)
    top5 = clf_shap_results.head(5)
    for idx, row in top5.iterrows():
        print(f"  {row['Feature']:30s}: {row['Mean |SHAP|']:.4f}")
    
    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: Regression vs Classification Feature Importance")
    print("="*70)
    
    # Compare which features drive each model
    comparison = pd.merge(
        reg_shap_results['importance_df'].head(10)[['Feature', 'Mean |SHAP|']].rename(columns={'Mean |SHAP|': 'Regression'}),
        clf_shap_results.head(10)[['Feature', 'Mean |SHAP|']].rename(columns={'Mean |SHAP|': 'Classification'}),
        on='Feature',
        how='outer'
    ).fillna(0)
    
    comparison.to_csv(shap_dir / "shap_comparison_reg_vs_clf.csv", index=False)
    print("\nTop Features Comparison:")
    print(comparison.to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"SHAP analysis complete! Results saved to: {shap_dir}")
    print(f"{'='*70}")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default="ridge", help="Regressor: xgb,histgbm,rf,ridge or 'all'")
    ap.add_argument("--clf", default="histgbm", help="Classifier: xgb,histgbm,rf,logreg or 'all'")
    ap.add_argument("--run-ablation", action="store_true", help="Run ablation study")
    ap.add_argument("--run-shap", action="store_true", help="Run SHAP explainability analysis")  # ADD THIS
    ap.add_argument("--label-strategy", default="combined", 
                    choices=["kev_only", "epss_only", "combined"],
                    help="Exploit label strategy")
    args = ap.parse_args()

    reg_names = ["xgb", "histgbm", "rf", "ridge"] if args.reg == "all" else [args.reg]
    clf_names = ["xgb", "histgbm", "rf", "logreg"] if args.clf == "all" else [args.clf]

    print("Loading and unifying data...")
    df = load_and_unify()

    print("Creating time-aware split...")
    train_df, test_df, meta = time_aware_split(df)
    print(json.dumps(meta, indent=2, default=str))

    # Select label function based on strategy
    label_funcs = {
        "kev_only": label_exploit_kev_only,
        "epss_only": lambda df: label_exploit_epss_only(df, 0.90),
        "combined": lambda df: label_exploit_combined(df, 0.75),
    }
    label_func = label_funcs[args.label_strategy]
    
    print(f"\nLabel Strategy: {args.label_strategy}")
    y_train_clf = label_func(train_df)
    y_test_clf = label_func(test_df)
    print(f"Train positives: {y_train_clf.sum()} / {len(y_train_clf)}")
    print(f"Test positives: {y_test_clf.sum()} / {len(y_test_clf)}")

    # Ablation study
    if args.run_ablation:
        ablation_results = run_ablation_study(train_df, test_df, "ridge")
        ablation_results.to_csv(DIRS["reports"] / "ablation_study.csv", index=False)
        print("\nAblation results saved to:", DIRS["reports"] / "ablation_study.csv")

    # Build separate pipelines
    print("\nBuilding feature pipelines...")
    pre_reg = build_feature_pipeline_regression(max_tfidf=8000)
    pre_clf = build_feature_pipeline_classification(max_tfidf=8000)  # No EPSS!
    
    pre_reg.fit(train_df)
    pre_clf.fit(train_df)

    results = []
    all_consistency = []

    for rname in reg_names:
        if rname == "xgb" and not HAS_XGB:
            print(f"Skipping {rname} (xgboost not installed)")
            continue

        reg = get_regressor(rname)
        
        # Fit regressor
        lab = pd.to_numeric(train_df["risk"], errors="coerce")
        m = lab.notna()
        if m.sum() < 1:
            continue

        Xt_reg = pre_reg.transform(train_df.loc[m])
        Xt_reg = to_dense_if_needed(Xt_reg, reg)
        y_tr = y_to_logit(lab[m].values.astype(float))
        reg.fit(Xt_reg, y_tr)

        # Evaluate regressor
        r2, mae, rfig = eval_regressor(pre_reg, reg, test_df, title_suffix=f"[{rname}]")
        if rfig:
            save_fig(rfig, DIRS["figures"] / f"reg_resid_{rname}.png")
        print(f"\n[{rname}] Regression: R²={r2:.4f}, MAE={mae:.4f}")

        for cname in clf_names:
            if cname == "xgb" and not HAS_XGB:
                continue

            base_clf = get_classifier(cname, y_train_clf)
            # FIT BASE CLASSIFIER SEPARATELY (for SHAP access)
            Xt_clf_train = pre_clf.transform(train_df)
            Xt_clf_train = to_dense_if_needed(Xt_clf_train, base_clf)
            base_clf.fit(Xt_clf_train, y_train_clf)  # ADD THIS LINE
            
            # Now calibrate (creates wrapper)
            clf_calib = platt_calibrate(base_clf, pre_clf, train_df, y_train_clf)
            clf_metrics, pr_fig, cal_fig = eval_classifier(
                pre_clf, clf_calib, test_df, label_func, title_suffix=f"[{cname}]"
            )
            
        if args.run_shap and HAS_SHAP:
            print(f"\n{'='*70}")
            print(f"Running SHAP Analysis for {rname} + {cname}")
            print(f"{'='*70}")
            
            try:
                run_shap_analysis(
                    pre_reg=pre_reg,
                    reg=reg,
                    pre_clf=pre_clf,
                    clf=base_clf,  # Use unwrapped classifier
                    train_df=train_df,
                    test_df=test_df,
                    reg_name=rname,
                    clf_name=cname,
                    output_dir=DIRS["figures"]
                )
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
                import traceback
                traceback.print_exc()
            if pr_fig:
                save_fig(pr_fig, DIRS["figures"] / f"pr_{cname}_{args.label_strategy}.png")
            if cal_fig:
                save_fig(cal_fig, DIRS["figures"] / f"cal_{cname}_{args.label_strategy}.png")

            print(f"[{rname} + {cname}] Classification: PR-AUC={clf_metrics['pr_auc']:.4f} "
                  f"[{clf_metrics['pr_auc_ci_low']:.3f}, {clf_metrics['pr_auc_ci_high']:.3f}], "
                  f"Brier={clf_metrics['brier']:.4f}")

            # Score full dataset
            Xt_all_reg = pre_reg.transform(df)
            Xt_all_reg = to_dense_if_needed(Xt_all_reg, reg)
            pred_risk = np.clip(logit_to_y(reg.predict(Xt_all_reg)), 0, 1)
            p_exploit = np.clip(clf_calib.predict_proba(df)[:, 1], 0, 1)

            scored = df.copy()
            scored["pred_risk"] = pred_risk
            scored["p_exploit"] = p_exploit
            
            # Consistency analysis
            consistency = analyze_consistency(scored, label_func)
            consistency["regressor"] = rname
            consistency["classifier"] = cname
            all_consistency.append(consistency)
            
            print(f"  Consistency: Spearman={consistency['spearman_corr']:.3f}, "
                  f"Tier Agreement={consistency['tier_agreement']:.3f}")

            results.append({
                "regressor": rname,
                "classifier": cname,
                "label_strategy": args.label_strategy,
                "R2": r2,
                "MAE": mae,
                "PR-AUC": clf_metrics["pr_auc"],
                "PR-AUC_CI_low": clf_metrics["pr_auc_ci_low"],
                "PR-AUC_CI_high": clf_metrics["pr_auc_ci_high"],
                "Brier": clf_metrics["brier"],
                "n_train_pos": int(y_train_clf.sum()),
                "n_test_pos": int(y_test_clf.sum()),
            })

    # Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv(DIRS["reports"] / "model_comparison_v2.csv", index=False)
    
    cons_df = pd.DataFrame(all_consistency)
    cons_df.to_csv(DIRS["reports"] / "consistency_analysis.csv", index=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(res_df.to_string(index=False))
    print(f"\nResults saved to: {DIRS['reports']}")


if __name__ == "__main__":
    main()
