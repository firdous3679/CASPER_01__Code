#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model comparison harness for Week-4 pipeline.

What this does:
- load & unify data (same as your script)
- time-aware split
- build the same feature pipeline (numeric + one-hot cats + TF-IDF)
- train multiple REGRESSORS and CLASSIFIERS
- Platt-calibrate classifier probabilities
- evaluate on the time-based test set:
    * Regressor: R^2
    * Classifier: PR-AUC, Brier, PR curve, calibration plot
- compute priority and write scored outputs per (reg, clf) pair
- write a results CSV summarizing model metrics

Run:
  python compare_models.py --reg all --clf all
  python compare_models.py --reg xgb,histgbm --clf xgb,logreg
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
# add import at top
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Regressors
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Calibration
from sklearn.linear_model import LogisticRegression as PlattLR

# Optional xgboost (will be skipped if not installed)
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from scipy import sparse as sp

def to_dense_if_needed(X, estimator):
    name = estimator.__class__.__name__.lower()
    needs_dense = ("histgradientboosting" in name) or ("randomforest" in name) \
                  or ("logisticregression" in name and getattr(estimator, "solver", "") != "saga")
    if needs_dense and sp.issparse(X):
        return X.toarray()
    return X
# Bounded target transform for regression on [0,1]
EPS = 1e-5  # avoid infinities at 0 or 1

def y_to_logit(y):
    """Map y in [0,1] to real line."""
    y = np.asarray(y, dtype=float)
    y = np.clip(y, EPS, 1 - EPS)
    return np.log(y / (1 - y))

def logit_to_y(z):
    """Map real line back to [0,1]."""
    z = np.asarray(z, dtype=float)
    return 1 / (1 + np.exp(-z))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------
# Config / paths
# ------------------------------
TODAY = datetime.now(timezone.utc).date()
RANDOM_STATE = 42

ROOT = Path("./model_compare").resolve()
DIRS = {
    "data": ROOT / "data",
    "outputs": ROOT / "outputs",
    "reports": ROOT / "reports",
    "figures": ROOT / "figures",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

IN_POWER = Path("../data/power_grid_cves_enriched.csv")
IN_ALL   = Path("../data/all_cves_risk_v2.csv")

# Priority config (matches your improved version)
PRIORITY_ALPHA = 0.6   # weight for criticality vs exposure (1-alpha)
PRIORITY_GAMMA = 1.2   # >1 emphasizes high impact without crushing mids
KEV_FLOOR      = 0.7
USE_KEV_FLOOR  = True

# ------------------------------
# Utilities
# ------------------------------
def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)   # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)          # sklearn <1.2

def parse_date(s):
    if pd.isna(s): return pd.NaT
    return pd.to_datetime(s, utc=True, errors="coerce")

def to_float(s):
    try: return float(s)
    except Exception: return np.nan

def ensure_cols(df, cols_defaults):
    for c, default in cols_defaults.items():
        if c not in df.columns:
            df[c] = default
    return df

def percentile_cut(series, p):
    return float(np.nanpercentile(series, p)) if len(series) else np.nan

def safe_str(x, default="unknown"):
    return str(x) if isinstance(x, str) and x.strip() else default

def save_fig(obj, path: Path = None, close=True):
    """
    Save a matplotlib Figure.

    Usage:
      save_fig(plt.gcf(), DIRS["figures"] / "name.png")
      save_fig(DIRS["figures"] / "name.png")  # saves current figure
    """
    if isinstance(obj, matplotlib.figure.Figure):
        fig = obj
        if path is None:
            raise ValueError("Must provide a path when passing a Figure.")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        if close:
            plt.close(fig)
    else:
        # obj is a path; save the current figure
        path = obj if path is None else path
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        if close:
            plt.close()


def short_explain(row):
    reasons = []
    if row.get("is_kev", 0) == 1: reasons.append("KEV")
    if row.get("epss", 0) >= 0.8: reasons.append("HighEPSS")
    vec = str(row.get("cvss_vector", "") or "")
    if "AV:N" in vec: reasons.append("AV:N")
    if "AC:L" in vec: reasons.append("AC:L")
    if "PR:N" in vec: reasons.append("PR:N")
    if "UI:N" in vec: reasons.append("UI:N")
    if row.get("criticality_0_5", 0) >= 4: reasons.append("HighCriticality")
    if row.get("exposure_0_5", 0) >= 4: reasons.append("HighExposure")
    if row.get("cvss_score", 0) >= 9: reasons.append("CVSS≥9")
    return ",".join(reasons[:5]) if reasons else "MixedSignals"

# ------------------------------
# Data loading / unify (same schema as your script)
# ------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Safe strings to avoid NaN issues
    desc = df.get("description", "").fillna("").astype(str)
    vec  = df.get("cvss_vector", "").fillna("").astype(str)

    # 1) Remote reachability (AV:N)
    df["is_remote"] = vec.str.contains(r"\bAV:N\b", na=False).astype(int)

    # 2) No privileges required (PR:N)
    df["is_priv_none"] = vec.str.contains(r"\bPR:N\b", na=False).astype(int)

    # 3) Base exploitability flags present
    flags = ["AV:N", "AC:L", "PR:N", "UI:N"]
    # count how many of the flags appear in the vector
    df["base_flags"] = sum(vec.str.contains(fr"\b{f}\b", na=False) for f in flags).astype(int)

    # 4) Description length (proxy for detail/complexity)
    df["text_len"] = desc.str.len().clip(upper=20000)

    return df

def load_and_unify(power_path=IN_POWER, all_path=IN_ALL) -> pd.DataFrame:
    df_cpps = pd.read_csv(power_path)
    df_all  = pd.read_csv(all_path)

    df_cpps.columns = [c.strip().lower() for c in df_cpps.columns]
    df_all.columns  = [c.strip().lower() for c in df_all.columns]

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
        df["age_days"] = df["published"].apply(lambda d: (TODAY - d.date()).days if pd.notna(d) else np.nan)
        df["cvss_score"] = df["cvss_score"].apply(to_float)
        df["vuln_cpe_count"] = pd.to_numeric(df["vuln_cpe_count"], errors="coerce")
        df["epss"] = pd.to_numeric(df["epss_percentile"], errors="coerce")
        df["is_kev"] = pd.to_numeric(df["in_cisa_kev"], errors="coerce").fillna(0).astype(int)
        for col in ["vendor","product","severity","description","cvss_vector"]:
            df[col] = df[col].apply(safe_str)
        for col in ["criticality_0_5","exposure_0_5"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_cpps["_source"] = "CPPS"
    df_all["_source"]  = "ALL"

    combined = pd.concat([df_all, df_cpps], ignore_index=True)
    combined.sort_values(by=["cve_id","_source"], ascending=[True, True], inplace=True)
    combined = combined.drop_duplicates(subset=["cve_id"], keep="last").reset_index(drop=True)

    combined["epss"] = combined["epss"].clip(0,1)
    combined = add_engineered_features(combined)

    # coalesce risk from ALL into combined
    df_all["risk"] = pd.to_numeric(df_all.get("risk", np.nan), errors="coerce")
    combined["risk"] = pd.to_numeric(combined.get("risk", np.nan), errors="coerce")
    combined = combined.merge(df_all[["cve_id","risk"]], on="cve_id", how="left", suffixes=("", "_all"))
    combined["risk"] = combined["risk"].fillna(combined["risk_all"])
    combined = combined.drop(columns=["risk_all"])

    combined.to_csv(DIRS["data"] / "combined_unified.csv", index=False, encoding="utf-8")
    print("After unify: risk_non_null =", int(combined["risk"].notna().sum()), "of", len(combined))
    return combined

def time_aware_split(df: pd.DataFrame):
    has_dates = df["published"].notna().mean() >= 0.5
    if has_dates:
        df_sorted = df.sort_values("published")
        cut = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:cut].copy()
        test_df  = df_sorted.iloc[cut:].copy()
        meta = {
            "strategy":"time",
            "train_range":[str(train_df["published"].min()), str(train_df["published"].max())],
            "test_range":[str(test_df["published"].min()),  str(test_df["published"].max())],
            "train_n":len(train_df),"test_n":len(test_df),
        }
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
        meta={"strategy":"random","train_n":len(train_df),"test_n":len(test_df)}
    return train_df, test_df, meta

# ------------------------------
# Features
# ------------------------------
def build_feature_pipeline(max_tfidf=8000):
    numeric_cols = [
        "cvss_score","epss","is_kev","vuln_cpe_count","age_days",
        "criticality_0_5","exposure_0_5",
        # new engineered numerics
        "is_remote","is_priv_none","base_flags","text_len"
    ]
    cat_cols = ["vendor","product","severity"]
    text_col = "description"

    num_tf = ("num", make_numeric_pipeline(), numeric_cols)
    cat_tf = ("cat", _make_ohe(), cat_cols)
    txt_tf = ("txt", TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf, min_df=2), text_col)

    return ColumnTransformer(
        transformers=[num_tf, cat_tf, txt_tf],
        remainder="drop",
        sparse_threshold=0.3
    )


# put this helper near your other utils
from sklearn.pipeline import Pipeline
def make_numeric_pipeline():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
        # no scaler; boosted trees don’t need it, and we want apples-to-apples with XGB
    ])

# ------------------------------
# Exploit label and calibration
# ------------------------------
def label_exploit(df):
    # you set this to 0.75 recently; keep it here for consistency
    return ((df["is_kev"] == 1) | (df["epss"] >= 0.75)).astype(int)

def platt_calibrate(base_clf, pre, train_df, y, random_state=RANDOM_STATE):
    # held-out slice for calibration
    X_tr, X_val, y_tr, y_val = train_test_split(
        train_df, y, test_size=0.2, random_state=random_state,
        stratify=y if y.sum() >= 2 else None
    )
    Xt_tr  = pre.transform(X_tr)
    Xt_val = pre.transform(X_val)
    Xt_tr  = to_dense_if_needed(Xt_tr, base_clf)
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
            p  = self.clf.predict_proba(Xt)[:, 1].reshape(-1, 1)
            p_cal = self.platt.predict_proba(p)[:, 1]
            return np.vstack([1 - p_cal, p_cal]).T

    return CalibWrapper(pre, base_clf, platt)

# ------------------------------
# Priority math
# ------------------------------
def _impact_score(rows: pd.DataFrame) -> np.ndarray:
    crit = rows["criticality_0_5"].astype(float).to_numpy() / 5.0
    expo = rows["exposure_0_5"].astype(float).to_numpy() / 5.0
    I = PRIORITY_ALPHA * crit + (1.0 - PRIORITY_ALPHA) * expo
    I = np.clip(I, 0.0, 1.0) ** PRIORITY_GAMMA
    if USE_KEV_FLOOR and "is_kev" in rows.columns:
        kev = rows["is_kev"].astype(int).to_numpy()
        I = np.maximum(I, KEV_FLOOR * (kev > 0))
    return I

def score_and_prioritize(df, pre, reg, clf_calib):
    Xt_all = pre.transform(df)
    Xt_reg = to_dense_if_needed(Xt_all, reg)

    # predict in logit space, then invert to [0,1]
    pred_risk_logit = reg.predict(Xt_reg)
    pred_risk = np.clip(logit_to_y(pred_risk_logit), 0, 1)


    # classifier path: if your CalibWrapper already densifies inside predict_proba, keep this;
    # otherwise do the same dance for clf_calib.clf.
    p_exploit = np.clip(clf_calib.predict_proba(df)[:, 1], 0, 1)

    I = _impact_score(df)
    priority_decomp = p_exploit * I
    priority = np.maximum(pred_risk, priority_decomp)

    out = df.copy()
    out["pred_risk"] = pred_risk
    out["p_exploit"] = p_exploit
    out["impact_I"] = I
    out["priority_decomp"] = priority_decomp
    out["priority"] = priority
    out["top_reasons"] = [short_explain(r) for _, r in df.iterrows()]
    return out

# ------------------------------
# Models registry
# ------------------------------
def get_regressor(name: str):
    name = name.lower()
    if name == "xgb":
        if not HAS_XGB: raise RuntimeError("xgboost not installed")
        return XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=0
        )
    if name == "histgbm":
        return HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, max_iter=300,
            random_state=RANDOM_STATE
        )
    if name == "rf":
        return RandomForestRegressor(
             n_estimators=300, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
            )
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown regressor '{name}'")

def get_classifier(name: str, y_train: np.ndarray):
    name = name.lower()
    # class imbalance weight
    pos = int(y_train.sum()); neg = int(len(y_train) - pos)
    spw = max(1.0, neg / max(1, pos))

    if name == "xgb":
        if not HAS_XGB: raise RuntimeError("xgboost not installed")
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=0
        )
    if name == "histgbm":
        return HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=500,
            random_state=RANDOM_STATE
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=20, class_weight={0:1, 1:spw},
            random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "logreg":
        return LogisticRegression(
        solver="saga",            # supports sparse
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        n_jobs=None               # not used by saga; harmless
    )
    raise ValueError(f"Unknown classifier '{name}'")

# ------------------------------
# Evaluation helpers
# ------------------------------
def dump_top_importances(pre, reg, k=30, path=None):
    """
    Print and optionally save top-k feature importances for tree/boosting models.
    Uses ColumnTransformer.get_feature_names_out() to guarantee length match.
    Skips models without feature_importances_.
    """
    # 1) Only tree/boosting models have feature_importances_
    imp = getattr(reg, "feature_importances_", None)
    if imp is None:
        print("Regressor has no feature_importances_. Skipping.")
        return

    # 2) Get names directly from the fitted ColumnTransformer
    try:
        names = pre.get_feature_names_out()
    except Exception:
        # Fallback: at least keep lengths aligned if names fail
        names = np.array([f"f_{i}" for i in range(len(imp))])

    # 3) If lengths still disagree, align safely and warn
    if len(names) != len(imp):
        print(f"Warning: names({len(names)}) != importances({len(imp)}). Aligning to min length.")
        L = min(len(names), len(imp))
        names = names[:L]
        imp = imp[:L]

    top = (pd.DataFrame({"feature": names, "importance": imp})
             .sort_values("importance", ascending=False)
             .head(k))

    print("\nTop features:\n", top.to_string(index=False))
    if path is not None:
        top.to_csv(path, index=False, encoding="utf-8")


def eval_regressor(pre, reg, test_df):
    mask = test_df["risk"].notna()
    if mask.sum() < 10:
        return np.nan, None

    Xt = pre.transform(test_df.loc[mask])
    Xt = to_dense_if_needed(Xt, reg)

    y_true = test_df.loc[mask, "risk"].astype(float).clip(0, 1)

    # predict in logit space, then invert
    y_pred_logit = reg.predict(Xt)
    y_pred = np.clip(logit_to_y(y_pred_logit), 0, 1)
    mae = mean_absolute_error(y_true, y_pred)  # in eval_regressor after y_pred
    print(f"Regressor MAE: {mae:.3f}")
    r2 = r2_score(y_true, y_pred)

    # residuals figure (true - pred in original [0,1] space)
    plt.figure(figsize=(4, 3))
    plt.hist(y_true - y_pred, bins=40)
    plt.xlabel("Residual (true - pred)")
    plt.ylabel("Count")
    plt.title("Regressor Residuals")
    return r2, plt.gcf()



def eval_classifier(pre, clf_calib, test_df, title_suffix=""):
    y_true = label_exploit(test_df)

    # wrapper handles densify + calibration
    y_prob = np.clip(clf_calib.predict_proba(test_df)[:, 1], 0, 1)

    pr_auc = average_precision_score(y_true, y_prob) if y_true.sum() >= 1 else np.nan
    brier  = brier_score_loss(y_true, y_prob) if len(y_true) else np.nan

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(4, 3))
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve {title_suffix}\nAP={pr_auc:.3f}")
    pr_fig = plt.gcf()

    # 10-bin calibration plot
    bins = pd.qcut(y_prob, q=10, duplicates="drop")
    tmp = pd.DataFrame({"bin": bins, "y": y_true, "p": y_prob})
    cal = tmp.groupby("bin").agg(emp_rate=("y","mean"), avg_pred=("p","mean")).reset_index(drop=True)
    plt.figure(figsize=(4, 3))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(cal["avg_pred"], cal["emp_rate"], s=15)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Empirical frequency")
    plt.title(f"Calibration (10-bin) {title_suffix}")
    cal_fig = plt.gcf()

    return pr_auc, brier, pr_fig, cal_fig



# ------------------------------
# Main compare
# ------------------------------

# ------------------------------
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default="all", help="comma list: xgb,histgbm,rf,ridge or 'all'")
    ap.add_argument("--clf", default="all", help="comma list: xgb,histgbm,rf,logreg or 'all'")
    args = ap.parse_args()

    reg_names = ["xgb", "histgbm", "rf", "ridge"] if args.reg == "all" else [s.strip().lower() for s in args.reg.split(",")]
    clf_names = ["xgb", "histgbm", "rf", "logreg"] if args.clf == "all" else [s.strip().lower() for s in args.clf.split(",")]

    print("Loading and unifying data...")
    df = load_and_unify()  # ensure add_engineered_features() is called inside this

    print("Creating time-aware split...")
    train_df, test_df, meta = time_aware_split(df)
    print(json.dumps(meta, indent=2, default=str))
    print("Train risk_non_null =", int(train_df["risk"].notna().sum()),
          "| Test risk_non_null =", int(test_df["risk"].notna().sum()))

    # Labels for classifier class weights, etc.
    y_train_clf = label_exploit(train_df)

    # === Diagnostics: coverage + imputed correlations (run once) ===
    for c in ["criticality_0_5", "exposure_0_5"]:
        n = train_df[c].notna().sum() if c in train_df.columns else 0
        print(f"{c}: {n} non-null of {len(train_df)}")

    requested_num_cols = [
        "cvss_score","epss","is_kev","vuln_cpe_count","age_days",
        "criticality_0_5","exposure_0_5","is_remote","is_priv_none",
        "base_flags","text_len"
    ]
    num_cols_diag = [c for c in requested_num_cols if c in train_df.columns]
    missing = sorted(set(requested_num_cols) - set(num_cols_diag))
    if missing:
        print("Missing numeric columns (skipped in diag):", missing)

    X_sel = train_df[num_cols_diag].apply(pd.to_numeric, errors="coerce")
    all_nan_cols = list(X_sel.columns[X_sel.isna().all()])
    if all_nan_cols:
        print("All-NaN numeric columns (dropped in diag):", all_nan_cols)
        X_sel = X_sel.drop(columns=all_nan_cols)

    imp_diag = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    Xn_arr = imp_diag.fit_transform(X_sel)
    Xn = pd.DataFrame(Xn_arr, columns=X_sel.columns, index=train_df.index)
    print("Imputed corr with risk (numeric block):")
    print(Xn.corrwith(train_df["risk"]).sort_values(ascending=False))

    # Dummy R² baseline and drift by year
    mask = test_df["risk"].notna()
    if mask.sum() > 0:
        y_true_test = test_df.loc[mask, "risk"].astype(float).clip(0, 1)
        dum = DummyRegressor(strategy="mean").fit(np.zeros((mask.sum(), 1)), y_true_test)
        r2_dummy = dum.score(np.zeros((mask.sum(), 1)), y_true_test)
        print(f"Dummy R² (predict mean): {r2_dummy:.3f}")
    else:
        print("Dummy R²: no labeled risk rows in test set.")

    print("Risk mean/var by year (train):")
    print(train_df.groupby(train_df["published"].dt.year)["risk"].agg(["count", "mean", "std"]))

    # === Single feature pipeline build (fit on train) ===
    print("Building feature pipeline...")
    pre = build_feature_pipeline(max_tfidf=8000)  # increase if RAM allows
    pre.fit(train_df)

    results = []
    for rname in list(reg_names):
        if rname == "xgb" and not HAS_XGB:
            print("Skipping regressor xgb (xgboost not installed).")
            continue

        reg = get_regressor(rname)

        # Fit regressor on rows with risk using logit target
        lab = pd.to_numeric(train_df["risk"], errors="coerce")
        m = lab.notna()
        if m.sum() < 1:
            print(f"[{rname}] No labeled risk rows; skipping regressor.")
            continue

        Xt_reg = pre.transform(train_df.loc[m])
        Xt_reg = to_dense_if_needed(Xt_reg, reg)
        y_tr = y_to_logit(lab[m].values.astype(float))
        reg.fit(Xt_reg, y_tr)

        # Top features (tree/boosting models); safe no-op for linear models
        dump_top_importances(pre, reg, k=30, path=DIRS["reports"] / f"reg_top_feats_{rname}.csv")

        # Evaluate regressor
        r2, rfig = eval_regressor(pre, reg, test_df)
        if rfig is not None:
            rfig_path = DIRS["figures"] / f"reg_resid_{rname}.png"
            save_fig(rfig, rfig_path)

        # Classifier loop
        for cname in list(clf_names):
            if cname == "xgb" and not HAS_XGB:
                print("Skipping classifier xgb (xgboost not installed).")
                continue

            base_clf = get_classifier(cname, y_train_clf)
            clf_calib = platt_calibrate(base_clf, pre, train_df, y_train_clf, RANDOM_STATE)

            pr_auc, brier, pr_fig, cal_fig = eval_classifier(pre, clf_calib, test_df, title_suffix=f"[{cname}]")
            pr_path  = DIRS["figures"] / f"pr_{cname}.png"
            cal_path = DIRS["figures"] / f"cal_{cname}.png"
            save_fig(pr_fig, pr_path)
            save_fig(cal_fig, cal_path)

            # Score full corpus and write outputs for this pair
            scored = score_and_prioritize(df, pre, reg, clf_calib)
            out_prefix = f"{rname}__{cname}"
            keep_cols = [
                "cve_id","vendor","product","cvss_score","epss","is_kev",
                "criticality_0_5","exposure_0_5",
                "pred_risk","p_exploit","priority","priority_decomp","impact_I",
                "published","severity","cvss_vector","_source","top_reasons"
            ]
            scored[keep_cols].to_csv(
                    DIRS["outputs"] / f"cve_scored__{out_prefix}.csv",
                    index=False, encoding="utf-8"
                )

            grp = (scored.groupby(["vendor","product"], dropna=False)["priority"]
                        .agg(max_priority="max", mean_priority="mean", n_cves="size")
                        .reset_index()
                        .sort_values(["max_priority","mean_priority"], ascending=[False, False]))
            grp.to_csv(
                DIRS["outputs"] / f"product_ranked__{out_prefix}.csv",
                index=False, encoding="utf-8"
            )

            results.append({
                "regressor": rname,
                "classifier": cname,
                "R2": r2,
                "PR-AUC": pr_auc,
                "Brier": brier,
                "train_n": meta.get("train_n"),
                "test_n": meta.get("test_n"),
                "split": meta.get("strategy")
            })
            print(f"[{rname} + {cname}] R2={r2:.3f} | PR-AUC={pr_auc:.3f} | Brier={brier:.3f}")

    # Summary table
    res_df = pd.DataFrame(results).sort_values(["PR-AUC","R2"], ascending=[False, False])
    res_path = DIRS["reports"] / "model_comparison.csv"
    res_df.to_csv(res_path, index=False, encoding="utf-8")
    print("\n=== Summary ===")
    print(res_df.to_string(index=False))
    print(f"\nSaved results to: {res_path}")
    print(f"Figures dir: {DIRS['figures']}")
    print(f"Outputs dir: {DIRS['outputs']}")
# ------------------------------

if __name__ == "__main__":
    main()
