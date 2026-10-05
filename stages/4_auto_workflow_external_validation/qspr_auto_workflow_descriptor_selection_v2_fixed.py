#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSPR auto-workflow for adsorption on polypropylene (PP)
=======================================================

Purpose
-------
This script performs a complete, defensible QSPR workflow:

1. Load molecular descriptor table.
2. Clean descriptors: missing values, constants, duplicate numeric columns.
3. Split compounds into TS/VS, preferably by Kennard-Stone.
4. Perform descriptor filtering using TS only:
   - correlation with target |r_y| threshold,
   - interdescriptor correlation filter for models that need low collinearity.
5. Select descriptors separately for each model family:
   - MLR: all-subsets 2-4 descriptors from a low-collinearity pool.
   - kNN: all-subsets 2-4 descriptors + k/weights selection by LOO-CV.
   - SVR: all-subsets 2-4 descriptors from a compact pool + RBF grid by LOO-CV.
   - PLS: top-m descriptor pool + number of latent variables selected by LOO-CV.
6. Fit final models on TS and evaluate:
   - TS fit,
   - LOO-CV on TS,
   - external VS.
7. Generate diagnostics:
   - predicted vs experimental plots,
   - residual plots,
   - Williams plots / applicability domain tables,
   - descriptor correlation matrices,
   - MLR coefficients and PLS VIP values,
   - automatic interpretation notes.
8. Optionally predict new external compounds, e.g. naphthalene/nitrobenzene,
   if their descriptors are provided in an external Excel/CSV file.

Important methodological rule
-----------------------------
The script NEVER uses VS or external literature values to select descriptors.
VS and literature values are only used after model selection for evaluation.
Forcing a model to match two literature values to +/- 0.01 would be target leakage,
not valid QSAR/QSPR modeling.

Install
-------
python -m pip install numpy pandas scikit-learn matplotlib openpyxl joblib

Example
-------
python qspr_auto_workflow_descriptor_selection.py --excel "data_qsar_pls.xlsx" --sheet PROJEKT --n-jobs 7

With external compounds:
python qspr_auto_workflow_descriptor_selection.py --excel "data_qsar_pls.xlsx" --external-file "external_naph_nitro.xlsx" --external-literature-col "Literature_LogKd" --n-jobs 7
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------------------------
# Defaults: safe starting point for this project
# -------------------------------------------------------------------------
DEFAULT_REQUIRED_KEEP = [
    "CrippenLogP", "SHBa", "ALogP", "RDF30i",  # teacher-pipeline MLR result
    "log D", "McGowan_Volume", "TPSA", "naAromAtom", "AATS4i", "minHBa",  # previous good candidates
]

# Compact default grid. It is intentionally modest because SVR is the slowest
# part of the workflow. Expand it only if you have time.
SVR_PARAM_GRID_DEFAULT = [
    {"C": C, "gamma": gamma, "epsilon": eps}
    for C in [1, 10, 100]
    for gamma in ["scale", 0.1]
    for eps in [0.05, 0.1]
]

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")[:120] or "model"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def q2_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Same formula as R2 on cross-validated predictions.
    return r2_safe(y_true, y_pred)


def read_table(path: str | Path, sheet: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet or 0)
    if suffix in [".csv", ".txt", ".tsv"]:
        sep = "\t" if suffix in [".tsv", ".txt"] else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file type: {suffix}. Use Excel or CSV/TSV.")


def parse_list_arg(x: Optional[str]) -> List[str]:
    if not x:
        return []
    # split by comma or semicolon, keep spaces inside descriptor names if quoted is not needed
    return [p.strip() for p in re.split(r"[,;]", x) if p.strip()]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def numeric_hash_series(s: pd.Series, decimals: int = 12) -> Tuple[Any, ...]:
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    vals = np.round(vals, decimals=decimals)
    return tuple(vals.tolist())


def select_numeric_descriptor_columns(
    df: pd.DataFrame,
    target_col: str,
    name_col: str,
    set_col: Optional[str],
    exclude_cols: Sequence[str],
) -> List[str]:
    exclude = set([target_col, name_col]) | set(exclude_cols)
    if set_col:
        exclude.add(set_col)
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        # Try numeric conversion; keep if at least most values are numeric.
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() >= max(3, int(0.8 * len(df))):
            cols.append(c)
    return cols


def kennard_stone_indices(X: np.ndarray, n_val: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return train and validation indices. TS is selected to cover descriptor space."""
    n = X.shape[0]
    if n_val <= 0 or n_val >= n - 2:
        raise ValueError(f"n_val must be between 1 and n-3; got {n_val} for n={n}")
    D = pairwise_distances(X)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    ts = [int(i), int(j)]
    remaining = [k for k in range(n) if k not in ts]
    while len(ts) < n - n_val:
        min_dists = D[np.ix_(remaining, ts)].min(axis=1)
        best = remaining[int(np.argmax(min_dists))]
        ts.append(best)
        remaining.remove(best)
    return np.array(sorted(ts), dtype=int), np.array(sorted(remaining), dtype=int)


def sorted_every_third_indices(y: np.ndarray, start_pos: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    sorted_idx = np.argsort(y)
    valid_positions = np.arange(start_pos, len(sorted_idx), 3)
    vs = sorted_idx[valid_positions]
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[vs] = False
    ts = np.arange(len(y))[train_mask]
    return np.array(sorted(ts), dtype=int), np.array(sorted(vs), dtype=int)


def set_column_indices(df: pd.DataFrame, set_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if set_col not in df.columns:
        raise ValueError(f"Set column '{set_col}' not found")
    vals = df[set_col].astype(str).str.strip().str.upper()
    ts = np.where(vals.isin(["TS", "TRAIN", "TRAINING", "T", "UCZACY", "UCZĄCY"]))[0]
    vs = np.where(vals.isin(["VS", "VALID", "VALIDATION", "TEST", "V", "WALIDACYJNY"]))[0]
    if len(ts) == 0 or len(vs) == 0:
        raise ValueError(f"Could not infer TS/VS from '{set_col}'. Use values TS and VS.")
    return np.array(sorted(ts), dtype=int), np.array(sorted(vs), dtype=int)


def corr_with_y(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    return X.apply(lambda col: pd.Series(col).corr(pd.Series(y)), axis=0).astype(float)


def greedy_corr_prune(
    X_train: pd.DataFrame,
    corr_y_abs: pd.Series,
    corr_xx_max: float,
    required_keep: Sequence[str] = (),
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Greedy correlation pruning. Higher |r_y| is kept.

    required_keep columns are protected if possible; if two protected descriptors are mutually correlated,
    both are kept but the event is logged. This allows known chemistry descriptors to remain visible.
    """
    required = [c for c in required_keep if c in X_train.columns]
    order = corr_y_abs.dropna().sort_values(ascending=False).index.tolist()
    # Ensure required descriptors are considered early, ordered by their |r_y|.
    req_sorted = [c for c in order if c in set(required)]
    rest = [c for c in order if c not in set(required)]
    order = req_sorted + rest

    corr_xx = X_train[order].corr(method="pearson").abs()
    kept: List[str] = []
    log: List[Dict[str, Any]] = []

    for c in order:
        if not kept:
            kept.append(c)
            log.append({"descriptor": c, "action": "kept", "reason": "first/highest rank", "max_abs_corr_to_kept": 0.0})
            continue
        vals = corr_xx.loc[c, kept].dropna()
        max_corr = float(vals.max()) if len(vals) else 0.0
        most_corr = str(vals.idxmax()) if len(vals) else ""
        if max_corr <= corr_xx_max:
            kept.append(c)
            log.append({"descriptor": c, "action": "kept", "reason": f"max |rXX| <= {corr_xx_max}", "max_abs_corr_to_kept": max_corr, "most_correlated_kept": most_corr})
        else:
            if c in required:
                kept.append(c)
                log.append({"descriptor": c, "action": "kept_forced", "reason": f"required_keep although |rXX|={max_corr:.3f} > {corr_xx_max}", "max_abs_corr_to_kept": max_corr, "most_correlated_kept": most_corr})
            else:
                log.append({"descriptor": c, "action": "dropped", "reason": f"collinear with {most_corr}", "max_abs_corr_to_kept": max_corr, "most_correlated_kept": most_corr})
    return kept, log


def max_pairwise_abs_corr(X: pd.DataFrame, cols: Sequence[str]) -> float:
    if len(cols) <= 1:
        return 0.0
    cm = X[list(cols)].corr(method="pearson").abs().to_numpy()
    tri = cm[np.triu_indices_from(cm, k=1)]
    tri = tri[~np.isnan(tri)]
    return float(tri.max()) if len(tri) else 0.0


def pairwise_corr_table(X: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    if len(cols) < 2:
        return pd.DataFrame(columns=["descriptor_1", "descriptor_2", "r", "abs_r"])
    cm = X[list(cols)].corr(method="pearson")
    for a, b in itertools.combinations(cols, 2):
        r = float(cm.loc[a, b]) if pd.notna(cm.loc[a, b]) else np.nan
        rows.append({"descriptor_1": a, "descriptor_2": b, "r": r, "abs_r": abs(r) if pd.notna(r) else np.nan})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False)


def make_model(kind: str, params: Optional[Dict[str, Any]] = None) -> BaseEstimator:
    params = params or {}
    kind = kind.upper()
    if kind == "MLR":
        return LinearRegression()
    if kind == "KNN":
        return KNeighborsRegressor(**params)
    if kind == "SVR":
        return SVR(kernel="rbf", **params)
    if kind == "PLS":
        n_components = int(params.get("n_components", 1))
        return PLSRegression(n_components=n_components, scale=False)
    raise ValueError(f"Unknown model kind: {kind}")


def make_pipeline_for(kind: str, params: Optional[Dict[str, Any]] = None) -> Pipeline:
    # Scaling inside CV folds avoids preprocessing leakage within LOO-CV.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", make_model(kind, params)),
    ])


def cv_predict_metrics(
    kind: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cols: Sequence[str],
    params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
) -> Dict[str, float]:
    if len(cols) == 0:
        return {"Q2_LOO": np.nan, "RMSECV": np.nan, "MAECV": np.nan}
    est = make_pipeline_for(kind, params)
    try:
        pred = cross_val_predict(est, X[list(cols)].values, y, cv=LeaveOneOut(), n_jobs=n_jobs)
        pred = np.asarray(pred).ravel()
        return {
            "Q2_LOO": q2_from_predictions(y, pred),
            "RMSECV": rmse(y, pred),
            "MAECV": mae(y, pred),
        }
    except Exception:
        return {"Q2_LOO": -np.inf, "RMSECV": np.inf, "MAECV": np.inf}


def fit_and_predict(
    kind: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_all: pd.DataFrame,
    cols: Sequence[str],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, np.ndarray]:
    est = make_pipeline_for(kind, params)
    est.fit(X_train[list(cols)].values, y_train)
    pred_all = np.asarray(est.predict(X_all[list(cols)].values)).ravel()
    return est, pred_all


def model_selection_sort_key(row: Dict[str, Any]) -> Tuple[float, float, int]:
    # Higher Q2 is better; lower RMSECV, fewer descriptors as tie-breakers.
    return (-float(row.get("Q2_LOO", -np.inf)), float(row.get("RMSECV", np.inf)), int(row.get("n_desc", 999)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_safe(y_true, y_pred),
    }

# -------------------------------------------------------------------------
# Descriptor selection per model family
# -------------------------------------------------------------------------

def select_mlr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    pool: Sequence[str],
    min_desc: int,
    max_desc: int,
    corr_limit_for_combo: Optional[float],
    n_jobs: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    combos = []
    for n in range(min_desc, max_desc + 1):
        combos.extend(list(itertools.combinations(pool, n)))

    def one(cols: Tuple[str, ...]) -> Dict[str, Any]:
        maxcorr = max_pairwise_abs_corr(X_train, cols)
        if corr_limit_for_combo is not None and maxcorr > corr_limit_for_combo:
            return {"model": "MLR", "descriptors": "; ".join(cols), "cols_json": json.dumps(list(cols), ensure_ascii=False), "n_desc": len(cols), "max_abs_intercorr": maxcorr, "Q2_LOO": -np.inf, "RMSECV": np.inf, "MAECV": np.inf, "status": "rejected_corr"}
        m = cv_predict_metrics("MLR", X_train, y_train, cols, n_jobs=1)
        return {"model": "MLR", "descriptors": "; ".join(cols), "cols_json": json.dumps(list(cols), ensure_ascii=False), "n_desc": len(cols), "max_abs_intercorr": maxcorr, **m, "status": "ok"}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(tuple(c)) for c in combos)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("MLR selection found no valid descriptor set.")
    ok = ok.sort_values(["Q2_LOO", "RMSECV", "n_desc"], ascending=[False, True, True])
    best = ok.iloc[0].to_dict()
    best["kind"] = "MLR"
    best["params"] = {}
    best["cols"] = json.loads(best["cols_json"])
    best["selection_rule"] = "all-subsets; selected by maximum Q2_LOO on TS"
    return best, df.sort_values(["Q2_LOO", "RMSECV"], ascending=[False, True])


def select_knn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    pool: Sequence[str],
    min_desc: int,
    max_desc: int,
    k_values: Sequence[int],
    weights_values: Sequence[str],
    corr_limit_for_combo: Optional[float],
    n_jobs: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    combos = []
    for n in range(min_desc, max_desc + 1):
        combos.extend(list(itertools.combinations(pool, n)))
    tasks = []
    max_k_allowed = max(1, len(y_train) - 1)  # LOO train fold size
    for cols in combos:
        maxcorr = max_pairwise_abs_corr(X_train, cols)
        if corr_limit_for_combo is not None and maxcorr > corr_limit_for_combo:
            tasks.append((cols, None, None, maxcorr, "rejected_corr"))
            continue
        for k in k_values:
            if k > max_k_allowed:
                continue
            for weights in weights_values:
                tasks.append((cols, int(k), str(weights), maxcorr, "ok"))

    def one(task: Tuple[Tuple[str, ...], Optional[int], Optional[str], float, str]) -> Dict[str, Any]:
        cols, k, weights, maxcorr, status = task
        if status != "ok":
            return {"model": "kNN", "descriptors": "; ".join(cols), "cols_json": json.dumps(list(cols), ensure_ascii=False), "n_desc": len(cols), "k": None, "weights": None, "max_abs_intercorr": maxcorr, "Q2_LOO": -np.inf, "RMSECV": np.inf, "MAECV": np.inf, "status": status}
        params = {"n_neighbors": k, "weights": weights}
        m = cv_predict_metrics("KNN", X_train, y_train, cols, params=params, n_jobs=1)
        return {"model": "kNN", "descriptors": "; ".join(cols), "cols_json": json.dumps(list(cols), ensure_ascii=False), "n_desc": len(cols), "k": k, "weights": weights, "max_abs_intercorr": maxcorr, **m, "status": "ok"}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(t) for t in tasks)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("kNN selection found no valid descriptor set.")
    ok = ok.sort_values(["Q2_LOO", "RMSECV", "n_desc"], ascending=[False, True, True])
    best = ok.iloc[0].to_dict()
    best["kind"] = "KNN"
    best["params"] = {"n_neighbors": int(best["k"]), "weights": str(best["weights"])}
    best["cols"] = json.loads(best["cols_json"])
    best["selection_rule"] = "all-subsets + k/weights grid; selected by maximum Q2_LOO on TS"
    return best, df.sort_values(["Q2_LOO", "RMSECV"], ascending=[False, True])


def select_svr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    pool: Sequence[str],
    min_desc: int,
    max_desc: int,
    param_grid: Sequence[Dict[str, Any]],
    corr_limit_for_combo: Optional[float],
    n_jobs: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    combos = []
    for n in range(min_desc, max_desc + 1):
        combos.extend(list(itertools.combinations(pool, n)))

    tasks = []
    for cols in combos:
        maxcorr = max_pairwise_abs_corr(X_train, cols)
        if corr_limit_for_combo is not None and maxcorr > corr_limit_for_combo:
            tasks.append((cols, None, maxcorr, "rejected_corr"))
            continue
        for params in param_grid:
            tasks.append((cols, params, maxcorr, "ok"))

    def one(task: Tuple[Tuple[str, ...], Optional[Dict[str, Any]], float, str]) -> Dict[str, Any]:
        cols, params, maxcorr, status = task
        if status != "ok":
            return {"model": "SVR", "descriptors": "; ".join(cols), "cols_json": json.dumps(list(cols), ensure_ascii=False), "n_desc": len(cols), "params": "", "max_abs_intercorr": maxcorr, "Q2_LOO": -np.inf, "RMSECV": np.inf, "MAECV": np.inf, "status": status}
        m = cv_predict_metrics("SVR", X_train, y_train, cols, params=params, n_jobs=1)
        return {"model": "SVR", "descriptors": "; ".join(cols), "cols_json": json.dumps(list(cols), ensure_ascii=False), "n_desc": len(cols), "params": json.dumps(params, ensure_ascii=False), "max_abs_intercorr": maxcorr, **m, "status": "ok"}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(t) for t in tasks)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("SVR selection found no valid descriptor set.")
    ok = ok.sort_values(["Q2_LOO", "RMSECV", "n_desc"], ascending=[False, True, True])
    best = ok.iloc[0].to_dict()
    best["kind"] = "SVR"
    best["params"] = json.loads(best["params"])
    best["cols"] = json.loads(best["cols_json"])
    best["selection_rule"] = "all-subsets + RBF-SVR hyperparameter grid; selected by maximum Q2_LOO on TS"
    return best, df.sort_values(["Q2_LOO", "RMSECV"], ascending=[False, True])


def select_pls(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    ranked_pool: Sequence[str],
    min_pls_desc: int,
    max_pls_desc: int,
    max_lv: int,
    n_jobs: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    tasks = []
    max_m = min(max_pls_desc, len(ranked_pool))
    min_m = min(min_pls_desc, max_m)
    for m in range(min_m, max_m + 1):
        cols = tuple(ranked_pool[:m])
        # LOO train fold has n_train-1 rows. PLS components <= min(n_features, n_train-2) is safer.
        max_lv_here = max(1, min(max_lv, m, len(y_train) - 2))
        for lv in range(1, max_lv_here + 1):
            tasks.append((cols, lv))

    def one(task: Tuple[Tuple[str, ...], int]) -> Dict[str, Any]:
        cols, lv = task
        params = {"n_components": int(lv)}
        m = cv_predict_metrics("PLS", X_train, y_train, cols, params=params, n_jobs=1)
        maxcorr = max_pairwise_abs_corr(X_train, cols)
        return {"model": "PLS", "descriptors": "; ".join(cols), "cols_json": json.dumps(list(cols), ensure_ascii=False), "n_desc": len(cols), "LV": int(lv), "max_abs_intercorr": maxcorr, **m, "status": "ok"}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(t) for t in tasks)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("PLS selection found no valid descriptor set.")
    ok = ok.sort_values(["Q2_LOO", "RMSECV", "n_desc"], ascending=[False, True, True])
    best = ok.iloc[0].to_dict()
    best["kind"] = "PLS"
    best["params"] = {"n_components": int(best["LV"])}
    best["cols"] = json.loads(best["cols_json"])
    best["selection_rule"] = "top-m ranked descriptors + LV grid; selected by maximum Q2_LOO on TS"
    return best, df.sort_values(["Q2_LOO", "RMSECV"], ascending=[False, True])

# -------------------------------------------------------------------------
# Diagnostics and plotting
# -------------------------------------------------------------------------

def leverage_descriptor_space(pipeline: Pipeline, X_all_cols: pd.DataFrame, train_indices_in_all: np.ndarray) -> Tuple[np.ndarray, float]:
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    Xs_all = scaler.transform(X_all_cols.values)
    Xs_train = Xs_all[train_indices_in_all]
    # Add intercept column for classical MLR-style leverage.
    X_design_train = np.column_stack([np.ones(Xs_train.shape[0]), Xs_train])
    X_design_all = np.column_stack([np.ones(Xs_all.shape[0]), Xs_all])
    XtX_inv = np.linalg.pinv(X_design_train.T @ X_design_train)
    h_all = np.einsum("ij,jk,ik->i", X_design_all, XtX_inv, X_design_all)
    p = Xs_train.shape[1]
    n = Xs_train.shape[0]
    h_star = 3 * (p + 1) / n
    return h_all.astype(float), float(h_star)


def leverage_pls_score_space(pipeline: Pipeline, X_all_cols: pd.DataFrame, train_indices_in_all: np.ndarray) -> Tuple[np.ndarray, float]:
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    pls: PLSRegression = pipeline.named_steps["model"]
    Xs_all = scaler.transform(X_all_cols.values)
    # PLSRegression.transform returns scores in latent variable space.
    T_all = pls.transform(Xs_all)
    if isinstance(T_all, tuple):
        T_all = T_all[0]
    T_all = np.asarray(T_all)
    T_train = T_all[train_indices_in_all]
    # Include intercept in score space.
    T_design_train = np.column_stack([np.ones(T_train.shape[0]), T_train])
    T_design_all = np.column_stack([np.ones(T_all.shape[0]), T_all])
    inv = np.linalg.pinv(T_design_train.T @ T_design_train)
    h_all = np.einsum("ij,jk,ik->i", T_design_all, inv, T_design_all)
    A = T_train.shape[1]
    n = T_train.shape[0]
    h_star = 3 * (A + 1) / n
    return h_all.astype(float), float(h_star)


def standardized_residuals(
    residuals_all: np.ndarray,
    train_indices_in_all: np.ndarray,
    p_eff: int,
) -> Tuple[np.ndarray, float]:
    res_train = residuals_all[train_indices_in_all]
    n = len(res_train)
    denom = max(1, n - p_eff - 1)
    s = float(np.sqrt(np.sum(res_train ** 2) / denom))
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.std(res_train, ddof=1)) if len(res_train) > 1 else 1.0
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return residuals_all / s, s


def compute_vip(pls: PLSRegression) -> np.ndarray:
    # Robust VIP formula for PLSRegression.
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    # Explained y variance per component.
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1).ravel()
    total_s = np.sum(s)
    if total_s <= 1e-12:
        return np.full(p, np.nan)
    w_norm2 = (w / np.linalg.norm(w, axis=0)) ** 2
    vip = np.sqrt(p * (w_norm2 @ s) / total_s)
    return vip


def plot_predicted_vs_observed(df_pred: pd.DataFrame, model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for split in ["TS", "VS"]:
        sub = df_pred[df_pred["Set"] == split]
        if not sub.empty:
            ax.scatter(sub["Observed"], sub["Predicted"], label=split, s=45)
    low = min(df_pred["Observed"].min(), df_pred["Predicted"].min())
    high = max(df_pred["Observed"].max(), df_pred["Predicted"].max())
    pad = 0.05 * (high - low if high > low else 1)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", linewidth=1)
    ax.set_xlabel("Experimental LogKda")
    ax.set_ylabel("Predicted LogKda")
    ax.set_title(f"{model_name}: predicted vs experimental")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_residuals(df_pred: pd.DataFrame, model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for split in ["TS", "VS"]:
        sub = df_pred[df_pred["Set"] == split]
        if not sub.empty:
            ax.scatter(sub["Predicted"], sub["Residual"], label=split, s=45)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted LogKda")
    ax.set_ylabel("Residual = experimental - predicted")
    ax.set_title(f"{model_name}: residuals vs predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_williams(df_ad: pd.DataFrame, model_name: str, h_star: float, out_path: Path, note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for split in ["TS", "VS", "External"]:
        sub = df_ad[df_ad["Set"] == split]
        if not sub.empty:
            ax.scatter(sub["Leverage_h"], sub["Standardized_residual"], label=split, s=45)
    ax.axhline(3, linestyle="--", linewidth=1)
    ax.axhline(-3, linestyle="--", linewidth=1)
    ax.axvline(h_star, linestyle="--", linewidth=1)
    ax.set_xlabel("Leverage h")
    ax.set_ylabel("Standardized residual")
    title = f"{model_name}: Williams plot"
    if note:
        title += f" ({note})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_corr_heatmap(X_train: pd.DataFrame, cols: Sequence[str], model_name: str, out_path: Path) -> None:
    if len(cols) == 0:
        return
    corr = X_train[list(cols)].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(5.5, 0.65 * len(cols) + 2), max(4.8, 0.65 * len(cols) + 1.5)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.set_title(f"{model_name}: descriptor correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    labels = summary["Model"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, summary["RMSECV"].astype(float), width, label="RMSECV")
    ax.bar(x + width / 2, summary["VS_RMSE"].astype(float), width, label="VS RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Model comparison: internal CV vs external VS")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def auto_interpret_model(row: Dict[str, Any]) -> str:
    notes = []
    model = row.get("Model", "model")
    q2 = row.get("Q2_LOO", np.nan)
    r2vs = row.get("VS_R2", np.nan)
    rmsecv = row.get("RMSECV", np.nan)
    rmsevs = row.get("VS_RMSE", np.nan)
    gap = row.get("RMSE_gap_VS_minus_CV", np.nan)
    maxcorr = row.get("max_abs_intercorr", np.nan)
    n_vs_out = row.get("n_VS_outside_AD", 0)
    n_vs_res = row.get("n_VS_residual_outliers", 0)
    n_desc = row.get("n_desc", np.nan)

    if pd.notna(q2) and q2 >= 0.5:
        notes.append("good internal predictive ability by LOO-CV")
    elif pd.notna(q2):
        notes.append("weak or unstable internal predictive ability by LOO-CV")

    if pd.notna(r2vs) and r2vs >= 0.8:
        notes.append("strong external validation R2")
    elif pd.notna(r2vs):
        notes.append("external validation is limited")

    if pd.notna(gap):
        if gap <= 0.15:
            notes.append("low CV-to-VS RMSE gap")
        elif gap <= 0.35:
            notes.append("moderate CV-to-VS RMSE gap")
        else:
            notes.append("large CV-to-VS RMSE gap; possible instability")

    if model.upper().startswith("MLR"):
        if pd.notna(maxcorr) and maxcorr <= 0.85:
            notes.append("descriptor intercorrelation acceptable for MLR")
        elif pd.notna(maxcorr):
            notes.append("descriptor intercorrelation may weaken MLR interpretation")
        if pd.notna(n_desc) and n_desc <= 4:
            notes.append("compact and interpretable descriptor set")

    if int(n_vs_out or 0) == 0 and int(n_vs_res or 0) == 0:
        notes.append("no VS compounds outside AD/residual limits")
    elif int(n_vs_res or 0) > 0:
        notes.append("some VS residual outliers; predictions require caution")
    elif int(n_vs_out or 0) > 0:
        notes.append("some VS compounds have high leverage; predictions require AD caution")

    return f"{model}: " + "; ".join(notes) + "."


def recommendation_score(row: Dict[str, Any]) -> float:
    """Lower is better. Composite for final recommendation after evaluation.

    This is not used for descriptor selection. It is only a transparent way to rank the
    already-selected model families for reporting.
    """
    rmsecv = float(row.get("RMSECV", np.inf))
    rmsevs = float(row.get("VS_RMSE", np.inf))
    gap = max(0.0, float(row.get("RMSE_gap_VS_minus_CV", 0.0))) if pd.notna(row.get("RMSE_gap_VS_minus_CV", np.nan)) else 0.0
    n_desc = float(row.get("n_desc", 0))
    vs_out = float(row.get("n_VS_outside_AD", 0))
    vs_res = float(row.get("n_VS_residual_outliers", 0))
    model = str(row.get("Model", ""))
    interpretability_bonus = -0.08 if model.upper().startswith("MLR") else 0.0
    # PLS with many descriptors gets small complexity penalty but not huge; it is valid chemometrically.
    complexity = 0.015 * max(0.0, n_desc - 4)
    return rmsecv + rmsevs + 0.5 * gap + 0.10 * vs_out + 0.25 * vs_res + complexity + interpretability_bonus

# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Path:
    t0 = time.time()
    out_dir = ensure_dir(args.out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    selections_dir = ensure_dir(out_dir / "selection_tables")

    required_keep = list(DEFAULT_REQUIRED_KEEP) + parse_list_arg(args.required_keep)
    exclude_cols = parse_list_arg(args.exclude_cols)
    if args.set_col:
        exclude_cols = [c for c in exclude_cols if c != args.set_col]

    df = read_table(args.excel, args.sheet)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found. Available columns include: {list(df.columns[:20])}")
    if args.name_col not in df.columns:
        # Create name column if absent.
        df[args.name_col] = [f"Compound_{i+1}" for i in range(len(df))]

    # Drop rows without target.
    y_all_raw = pd.to_numeric(df[args.target_col], errors="coerce")
    good_rows = y_all_raw.notna()
    if good_rows.sum() < len(df):
        df = df.loc[good_rows].reset_index(drop=True)
        y_all_raw = y_all_raw.loc[good_rows].reset_index(drop=True)

    y_all = y_all_raw.to_numpy(dtype=float)
    names = df[args.name_col].astype(str).to_numpy()

    preprocessing_log: List[Dict[str, Any]] = []

    descriptor_cols = select_numeric_descriptor_columns(df, args.target_col, args.name_col, args.set_col, exclude_cols)
    X0 = df[descriptor_cols].apply(pd.to_numeric, errors="coerce")
    preprocessing_log.append({"step": "initial_numeric_descriptor_detection", "n_descriptors": len(descriptor_cols), "details": "numeric columns excluding target/name/set/exclude"})

    # Remove descriptors with any missing/nonfinite values.
    finite_mask = np.isfinite(X0.to_numpy(dtype=float)).all(axis=0)
    dropped_missing = list(X0.columns[~finite_mask])
    X1 = X0.loc[:, finite_mask].copy()
    preprocessing_log.append({"step": "drop_missing_or_nonfinite", "dropped": len(dropped_missing), "remaining": X1.shape[1], "examples": "; ".join(dropped_missing[:20])})

    # Remove constants / near-constants.
    variances = X1.var(axis=0)
    keep_var = variances > args.var_thresh
    dropped_constant = list(X1.columns[~keep_var])
    X2 = X1.loc[:, keep_var].copy()
    preprocessing_log.append({"step": "drop_constant_low_variance", "threshold": args.var_thresh, "dropped": len(dropped_constant), "remaining": X2.shape[1], "examples": "; ".join(dropped_constant[:20])})

    # Remove exact numeric duplicates.
    seen: Dict[Tuple[Any, ...], str] = {}
    keep_cols = []
    dup_rows = []
    for c in X2.columns:
        h = numeric_hash_series(X2[c], decimals=args.duplicate_round_decimals)
        if h in seen:
            dup_rows.append({"dropped_duplicate": c, "kept_original": seen[h]})
        else:
            seen[h] = c
            keep_cols.append(c)
    X_clean = X2[keep_cols].copy()
    preprocessing_log.append({"step": "drop_exact_numeric_duplicates", "dropped": len(dup_rows), "remaining": X_clean.shape[1]})

    if X_clean.shape[1] < 2:
        raise RuntimeError("Not enough usable descriptors after cleaning.")

    # Split.
    split_mode = args.split_mode.lower()
    if split_mode == "kennard_stone":
        # X-only scaling for KS. This does not use y.
        X_for_ks = StandardScaler().fit_transform(X_clean.values)
        ts_idx, vs_idx = kennard_stone_indices(X_for_ks, args.n_vs)
        split_label = f"Kennard-Stone; VS={len(vs_idx)}"
    elif split_mode == "sorted_every_third":
        ts_idx, vs_idx = sorted_every_third_indices(y_all, start_pos=args.sorted_start_pos)
        split_label = f"sorted_every_third; VS={len(vs_idx)}"
    elif split_mode == "set":
        if not args.set_col:
            raise ValueError("--split-mode set requires --set-col")
        ts_idx, vs_idx = set_column_indices(df, args.set_col)
        split_label = f"set column '{args.set_col}'; VS={len(vs_idx)}"
    else:
        raise ValueError("split-mode must be kennard_stone, sorted_every_third, or set")

    X_ts_clean = X_clean.iloc[ts_idx].copy()
    y_ts = y_all[ts_idx]
    X_vs_clean = X_clean.iloc[vs_idx].copy()
    y_vs = y_all[vs_idx]

    # Supervised TS-only descriptor filter.
    corr_y_signed = corr_with_y(X_ts_clean, y_ts)
    corr_y_abs = corr_y_signed.abs().replace([np.inf, -np.inf], np.nan).dropna()
    candidates_after_ry = corr_y_abs[corr_y_abs >= args.corr_y_min].sort_values(ascending=False).index.tolist()
    # Force known descriptors only if they exist; they still need to have finite values.
    for c in required_keep:
        if c in X_clean.columns and c not in candidates_after_ry:
            candidates_after_ry.append(c)
    if len(candidates_after_ry) < 2:
        raise RuntimeError("Too few descriptors after |r_y| filter. Lower --corr-y-min.")
    X_candidates = X_clean[candidates_after_ry].copy()
    corr_y_abs_candidates = corr_y_abs.reindex(candidates_after_ry).fillna(0.0)
    preprocessing_log.append({"step": "TS_only_filter_abs_r_y", "threshold": args.corr_y_min, "remaining": len(candidates_after_ry), "details": "required_keep descriptors may be retained for audit"})

    uncorrelated_pool, corr_prune_log = greedy_corr_prune(
        X_candidates.iloc[ts_idx], corr_y_abs_candidates, args.corr_xx_max, required_keep=required_keep
    )
    preprocessing_log.append({"step": "TS_only_greedy_intercorrelation_prune", "threshold": args.corr_xx_max, "remaining": len(uncorrelated_pool)})

    # Ranked pools.
    corr_rank_uncorr = corr_y_abs_candidates.reindex(uncorrelated_pool).fillna(0).sort_values(ascending=False)
    corr_rank_all = corr_y_abs_candidates.sort_values(ascending=False)

    pool_mlr = corr_rank_uncorr.head(args.top_mlr).index.tolist()
    pool_knn = corr_rank_uncorr.head(args.top_knn).index.tolist()
    pool_svr = corr_rank_uncorr.head(args.top_svr).index.tolist()
    pool_pls = corr_rank_all.head(args.top_pls_max).index.tolist()

    if len(pool_mlr) < args.min_desc:
        raise RuntimeError("MLR pool too small. Increase --top-mlr or lower filters.")
    if len(pool_knn) < args.min_desc:
        raise RuntimeError("kNN pool too small. Increase --top-knn or lower filters.")
    if len(pool_svr) < args.min_desc:
        raise RuntimeError("SVR pool too small. Increase --top-svr or lower filters.")
    if len(pool_pls) < args.pls_min_desc:
        raise RuntimeError("PLS pool too small. Increase --top-pls-max or lower filters.")

    print("\n" + "=" * 78)
    print("QSPR automatic workflow")
    print("=" * 78)
    print(f"Input file: {args.excel}")
    print(f"Compounds: n={len(df)} | TS={len(ts_idx)} | VS={len(vs_idx)} | split={split_label}")
    print(f"Descriptors after cleaning: {X_clean.shape[1]}")
    print(f"Candidates after |r_y| filter: {len(candidates_after_ry)}")
    print(f"Low-collinearity pool: {len(uncorrelated_pool)}")
    print(f"MLR pool={len(pool_mlr)}, kNN pool={len(pool_knn)}, SVR pool={len(pool_svr)}, PLS pool max={len(pool_pls)}")
    print("=" * 78 + "\n")

    # Selection. Use TS-only X candidates (raw values); pipelines scale within CV.
    X_train = X_candidates.iloc[ts_idx].reset_index(drop=True)
    X_all_candidates = X_candidates.reset_index(drop=True)
    # mapping: ts_idx are original positions; after reset, TS positions are range(len(ts_idx)) within X_train.

    selected: List[Dict[str, Any]] = []
    selection_tables: Dict[str, pd.DataFrame] = {}

    print("Selecting MLR descriptors...")
    best_mlr, table_mlr = select_mlr(
        X_train, y_ts, pool_mlr, args.min_desc, args.max_desc,
        corr_limit_for_combo=args.corr_xx_max if args.enforce_combo_corr else None,
        n_jobs=args.n_jobs,
    )
    selected.append(best_mlr)
    selection_tables["MLR_selection_all_candidates"] = table_mlr
    table_mlr.head(args.save_top_n).to_csv(selections_dir / "MLR_top_selection.csv", index=False)

    print("Selecting kNN descriptors and hyperparameters...")
    best_knn, table_knn = select_knn(
        X_train, y_ts, pool_knn, args.min_desc, args.max_desc,
        k_values=list(range(args.knn_k_min, args.knn_k_max + 1)),
        weights_values=args.knn_weights.split(","),
        corr_limit_for_combo=args.corr_xx_max if args.enforce_combo_corr else None,
        n_jobs=args.n_jobs,
    )
    selected.append(best_knn)
    selection_tables["kNN_selection_all_candidates"] = table_knn
    table_knn.head(args.save_top_n).to_csv(selections_dir / "kNN_top_selection.csv", index=False)

    print("Selecting SVR descriptors and hyperparameters...")
    best_svr, table_svr = select_svr(
        X_train, y_ts, pool_svr, args.min_desc, args.max_desc,
        param_grid=SVR_PARAM_GRID_DEFAULT,
        corr_limit_for_combo=args.corr_xx_max if args.enforce_combo_corr else None,
        n_jobs=args.n_jobs,
    )
    selected.append(best_svr)
    selection_tables["SVR_selection_all_candidates"] = table_svr
    table_svr.head(args.save_top_n).to_csv(selections_dir / "SVR_top_selection.csv", index=False)

    print("Selecting PLS descriptor-pool size and number of latent variables...")
    best_pls, table_pls = select_pls(
        X_train, y_ts, pool_pls, args.pls_min_desc, args.top_pls_max, args.pls_max_lv,
        n_jobs=args.n_jobs,
    )
    selected.append(best_pls)
    selection_tables["PLS_selection_all_candidates"] = table_pls
    table_pls.head(args.save_top_n).to_csv(selections_dir / "PLS_top_selection.csv", index=False)

    # Prepare unified X for selected descriptors. All selected cols must exist in X_candidates.
    split_series = np.array(["TS" if i in set(ts_idx) else "VS" for i in range(len(df))], dtype=object)
    all_predictions_rows = []
    all_ad_rows = []
    coeff_rows = []
    vip_rows = []
    model_summary_rows = []
    interpretation_rows = []
    corr_matrix_sheets: Dict[str, pd.DataFrame] = {}

    for spec in selected:
        kind = spec["kind"]
        print(f"Evaluating final {kind} model and generating diagnostics...", flush=True)
        model_name = safe_name(f"{kind}_{spec.get('selection_rule','').split(';')[0]}")
        if kind == "KNN":
            model_name = safe_name(f"kNN_k{spec['params']['n_neighbors']}_{spec['params']['weights']}")
        elif kind == "SVR":
            model_name = safe_name("SVR_RBF_grid")
        elif kind == "PLS":
            model_name = safe_name(f"PLS_TOP{len(spec['cols'])}_LV{spec['params']['n_components']}")
        elif kind == "MLR":
            model_name = safe_name("MLR_allsub")

        cols = spec["cols"]
        params = spec.get("params", {})
        missing_cols = [c for c in cols if c not in X_candidates.columns]
        if missing_cols:
            # This should not happen. It usually means that a descriptor name was altered
            # by string parsing or the input file contains invisible leading/trailing spaces.
            available_by_stripped = {str(c).strip(): c for c in X_candidates.columns}
            repaired_cols = [available_by_stripped.get(str(c).strip(), c) for c in cols]
            still_missing = [c for c in repaired_cols if c not in X_candidates.columns]
            if still_missing:
                raise KeyError(
                    f"Selected descriptors are missing from the candidate table: {still_missing}. "
                    f"Original selected columns: {cols}. Check whitespace in column names or duplicate-column removal."
                )
            cols = repaired_cols
            spec["cols"] = cols
        X_selected_all = X_candidates[cols].reset_index(drop=True)
        X_selected_train = X_selected_all.iloc[ts_idx]

        pipeline, pred_all = fit_and_predict(kind, X_selected_train, y_ts, X_selected_all, cols, params=params)
        observed_all = y_all.copy()
        residual_all = observed_all - pred_all

        # CV predictions on TS only.
        try:
            cv_pred_ts = cross_val_predict(make_pipeline_for(kind, params), X_selected_train.values, y_ts, cv=LeaveOneOut(), n_jobs=1).ravel()
        except Exception:
            cv_pred_ts = np.full_like(y_ts, np.nan, dtype=float)
        cv_metrics = {
            "Q2_LOO": q2_from_predictions(y_ts, cv_pred_ts) if np.isfinite(cv_pred_ts).all() else spec.get("Q2_LOO", np.nan),
            "RMSECV": rmse(y_ts, cv_pred_ts) if np.isfinite(cv_pred_ts).all() else spec.get("RMSECV", np.nan),
            "MAECV": mae(y_ts, cv_pred_ts) if np.isfinite(cv_pred_ts).all() else spec.get("MAECV", np.nan),
        }

        ts_metrics = evaluate_predictions(y_ts, pred_all[ts_idx])
        vs_metrics = evaluate_predictions(y_vs, pred_all[vs_idx])
        maxcorr = max_pairwise_abs_corr(X_selected_train, cols)

        # AD/leverage.
        train_positions_in_all = np.array(ts_idx, dtype=int)
        if kind == "PLS":
            leverage, h_star = leverage_pls_score_space(pipeline, X_selected_all, train_positions_in_all)
            p_eff = int(params.get("n_components", 1))
            ad_basis = "PLS score-space leverage"
        else:
            leverage, h_star = leverage_descriptor_space(pipeline, X_selected_all, train_positions_in_all)
            p_eff = len(cols)
            ad_basis = "descriptor-space leverage"

        std_resid, resid_sd = standardized_residuals(residual_all, train_positions_in_all, p_eff=p_eff)
        high_leverage = leverage > h_star
        residual_outlier = np.abs(std_resid) > 3.0
        outside_ad = high_leverage | residual_outlier

        pred_df = pd.DataFrame({
            "Model": model_name,
            "Compound": names,
            "Set": split_series,
            "Observed": observed_all,
            "Predicted": pred_all,
            "Residual": residual_all,
            "AbsResidual": np.abs(residual_all),
            "Leverage_h": leverage,
            "Standardized_residual": std_resid,
            "High_leverage": high_leverage,
            "Residual_outlier": residual_outlier,
            "Outside_AD": outside_ad,
        })

        # Add rows.
        all_predictions_rows.extend(pred_df[["Model", "Compound", "Set", "Observed", "Predicted", "Residual", "AbsResidual"]].to_dict("records"))
        ad_df = pred_df[["Model", "Compound", "Set", "Observed", "Predicted", "Residual", "Leverage_h", "Standardized_residual", "High_leverage", "Residual_outlier", "Outside_AD"]].copy()
        ad_df["h_star"] = h_star
        ad_df["AD_basis"] = ad_basis
        all_ad_rows.extend(ad_df.to_dict("records"))

        # Plots.
        plot_predicted_vs_observed(pred_df, model_name, plots_dir / f"{model_name}_predicted_vs_experimental.png")
        plot_residuals(pred_df, model_name, plots_dir / f"{model_name}_residuals_vs_predicted.png")
        plot_williams(ad_df, model_name, h_star, plots_dir / f"{model_name}_Williams_plot.png", note=ad_basis)
        plot_corr_heatmap(X_selected_train, cols, model_name, plots_dir / f"{model_name}_descriptor_correlation.png")

        corr_matrix_sheets[model_name[:31]] = X_selected_train[cols].corr(method="pearson")

        # Coefficients.
        if kind == "MLR":
            scaler = pipeline.named_steps["scaler"]
            model = pipeline.named_steps["model"]
            coef_scaled = np.asarray(model.coef_).ravel()
            intercept_scaled = float(model.intercept_)
            means = scaler.mean_
            scales = scaler.scale_
            coef_original = coef_scaled / scales
            intercept_original = intercept_scaled - float(np.sum(coef_scaled * means / scales))
            coeff_rows.append({"Model": model_name, "term": "Intercept_original_scale", "coefficient": intercept_original, "coefficient_type": "original_descriptor_scale"})
            coeff_rows.append({"Model": model_name, "term": "Intercept_standardized_X", "coefficient": intercept_scaled, "coefficient_type": "standardized_descriptor_scale"})
            for c, cs, co in zip(cols, coef_scaled, coef_original):
                coeff_rows.append({"Model": model_name, "term": c, "coefficient": float(co), "coefficient_type": "original_descriptor_scale"})
                coeff_rows.append({"Model": model_name, "term": c, "coefficient": float(cs), "coefficient_type": "standardized_descriptor_scale"})
        elif kind == "PLS":
            pls = pipeline.named_steps["model"]
            try:
                coefs = np.asarray(pls.coef_).ravel()
                for c, co in zip(cols, coefs):
                    coeff_rows.append({"Model": model_name, "term": c, "coefficient": float(co), "coefficient_type": "PLS_scaled_X_coefficient"})
            except Exception:
                pass
            try:
                vip = compute_vip(pls)
                for c, v in zip(cols, vip):
                    vip_rows.append({"Model": model_name, "descriptor": c, "VIP": float(v), "VIP_gt_1": bool(v > 1.0)})
            except Exception:
                pass

        n_vs_out = int(ad_df[(ad_df["Set"] == "VS") & (ad_df["Outside_AD"] == True)].shape[0])
        n_vs_high = int(ad_df[(ad_df["Set"] == "VS") & (ad_df["High_leverage"] == True)].shape[0])
        n_vs_res = int(ad_df[(ad_df["Set"] == "VS") & (ad_df["Residual_outlier"] == True)].shape[0])
        n_ts_out = int(ad_df[(ad_df["Set"] == "TS") & (ad_df["Outside_AD"] == True)].shape[0])

        summary = {
            "Model": model_name,
            "Kind": kind,
            "Descriptors": "; ".join(cols),
            "n_desc": len(cols),
            "Params": json.dumps(params, ensure_ascii=False),
            "Selection_rule": spec.get("selection_rule", ""),
            "Q2_LOO": cv_metrics["Q2_LOO"],
            "RMSECV": cv_metrics["RMSECV"],
            "MAECV": cv_metrics["MAECV"],
            "TS_RMSE": ts_metrics["RMSE"],
            "TS_MAE": ts_metrics["MAE"],
            "TS_R2": ts_metrics["R2"],
            "VS_RMSE": vs_metrics["RMSE"],
            "VS_MAE": vs_metrics["MAE"],
            "VS_R2": vs_metrics["R2"],
            "RMSE_gap_VS_minus_CV": vs_metrics["RMSE"] - cv_metrics["RMSECV"],
            "max_abs_intercorr": maxcorr,
            "h_star": h_star,
            "AD_basis": ad_basis,
            "n_TS_outside_AD": n_ts_out,
            "n_VS_outside_AD": n_vs_out,
            "n_VS_high_leverage": n_vs_high,
            "n_VS_residual_outliers": n_vs_res,
        }
        summary["Recommendation_score_lower_better"] = recommendation_score(summary)
        summary["Auto_interpretation"] = auto_interpret_model(summary)
        model_summary_rows.append(summary)
        interpretation_rows.append({"Model": model_name, "Interpretation": summary["Auto_interpretation"]})

    model_summary = pd.DataFrame(model_summary_rows).sort_values("Recommendation_score_lower_better")
    all_predictions = pd.DataFrame(all_predictions_rows)
    all_ad = pd.DataFrame(all_ad_rows)
    coeff_df = pd.DataFrame(coeff_rows)
    vip_df = pd.DataFrame(vip_rows)
    interp_df = pd.DataFrame(interpretation_rows)
    split_df = pd.DataFrame({
        "Original_index": np.arange(len(df)),
        "Compound": names,
        "Set": split_series,
        args.target_col: y_all,
    })

    plot_metric_comparison(model_summary, plots_dir / "model_comparison_RMSECV_vs_VS_RMSE.png")

    # External prediction, if descriptor file is supplied.
    external_predictions = pd.DataFrame()
    if args.external_file:
        ext = read_table(args.external_file, args.external_sheet)
        if args.name_col not in ext.columns:
            ext[args.name_col] = [f"External_{i+1}" for i in range(len(ext))]
        ext_names = ext[args.name_col].astype(str).to_numpy()
        lit_col = args.external_literature_col
        lit_values = None
        if lit_col and lit_col in ext.columns:
            lit_values = pd.to_numeric(ext[lit_col], errors="coerce").to_numpy(dtype=float)

        ext_rows = []
        for spec in selected:
            kind = spec["kind"]
            cols = spec["cols"]
            params = spec.get("params", {})
            if not all(c in ext.columns for c in cols):
                missing = [c for c in cols if c not in ext.columns]
                for nm in ext_names:
                    ext_rows.append({"Model": kind, "Compound": nm, "Predicted_LogKda": np.nan, "status": f"missing descriptors: {missing}"})
                continue
            model_name = ""
            if kind == "MLR": model_name = "MLR_allsub"
            elif kind == "KNN": model_name = f"kNN_k{params.get('n_neighbors')}_{params.get('weights')}"
            elif kind == "SVR": model_name = "SVR_RBF_grid"
            elif kind == "PLS": model_name = f"PLS_TOP{len(cols)}_LV{params.get('n_components')}"

            X_ext = ext[cols].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(X_ext.to_numpy(dtype=float)).all():
                status = "nonfinite descriptor values in external file"
                preds = np.full(len(ext), np.nan)
            else:
                # Fit on TS by default. Optionally refit on all original compounds after validation.
                if args.refit_on_all_for_external:
                    train_idx_ext = np.arange(len(df))
                    X_train_ext = X_candidates[cols]
                    y_train_ext = y_all
                    fit_note = "refit_on_all_after_validation"
                else:
                    train_idx_ext = ts_idx
                    X_train_ext = X_candidates.iloc[ts_idx][cols]
                    y_train_ext = y_ts
                    fit_note = "fit_on_TS_only"
                pipe = make_pipeline_for(kind, params)
                pipe.fit(X_train_ext.values, y_train_ext)
                preds = np.asarray(pipe.predict(X_ext.values)).ravel()
                status = fit_note
            for i, nm in enumerate(ext_names):
                row = {"Model": model_name, "Kind": kind, "Compound": nm, "Predicted_LogKda": float(preds[i]) if np.isfinite(preds[i]) else np.nan, "Descriptors_used": "; ".join(cols), "status": status}
                if lit_values is not None and i < len(lit_values) and np.isfinite(lit_values[i]) and np.isfinite(preds[i]):
                    row["Literature_LogKd_or_LogKda"] = float(lit_values[i])
                    row["Prediction_minus_literature"] = float(preds[i] - lit_values[i])
                    row["Abs_error_vs_literature"] = float(abs(preds[i] - lit_values[i]))
                ext_rows.append(row)
        external_predictions = pd.DataFrame(ext_rows)

    # Write main files.
    model_summary.to_csv(out_dir / "model_summary_ranked.csv", index=False)
    all_predictions.to_csv(out_dir / "all_predictions_TS_VS.csv", index=False)
    all_ad.to_csv(out_dir / "all_Williams_AD.csv", index=False)
    coeff_df.to_csv(out_dir / "model_coefficients.csv", index=False)
    vip_df.to_csv(out_dir / "PLS_VIP.csv", index=False)
    split_df.to_csv(out_dir / "TS_VS_split.csv", index=False)
    pd.DataFrame(preprocessing_log).to_csv(out_dir / "preprocessing_log.csv", index=False)
    pd.DataFrame(corr_prune_log).to_csv(out_dir / "intercorrelation_pruning_log.csv", index=False)
    pairwise_corr_table(X_candidates.iloc[ts_idx], uncorrelated_pool[:min(80, len(uncorrelated_pool))]).to_csv(out_dir / "pairwise_correlations_low_collinearity_pool_top80.csv", index=False)
    if not external_predictions.empty:
        external_predictions.to_csv(out_dir / "external_predictions.csv", index=False)

    # Excel workbook.
    xlsx_path = out_dir / "QSPR_auto_workflow_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        readme_rows = [
            {"Item": "Method", "Value": "Automatic QSPR descriptor selection and diagnostics"},
            {"Item": "Input file", "Value": str(args.excel)},
            {"Item": "Split", "Value": split_label},
            {"Item": "Important rule", "Value": "Descriptors selected only on TS by LOO-CV; VS and external literature are evaluation only."},
            {"Item": "Final recommendation", "Value": str(model_summary.iloc[0]["Model"]) if not model_summary.empty else ""},
            {"Item": "Warning", "Value": "Do not force fitting to naphthalene/nitrobenzene literature values; that would be leakage/overfitting."},
        ]
        pd.DataFrame(readme_rows).to_excel(writer, sheet_name="README", index=False)
        model_summary.to_excel(writer, sheet_name="model_summary", index=False)
        split_df.to_excel(writer, sheet_name="TS_VS_split", index=False)
        pd.DataFrame(preprocessing_log).to_excel(writer, sheet_name="preprocessing_log", index=False)
        pd.DataFrame(corr_prune_log).to_excel(writer, sheet_name="corr_prune_log", index=False)
        all_predictions.to_excel(writer, sheet_name="all_predictions", index=False)
        all_ad.to_excel(writer, sheet_name="Williams_AD", index=False)
        coeff_df.to_excel(writer, sheet_name="coefficients", index=False)
        vip_df.to_excel(writer, sheet_name="PLS_VIP", index=False)
        interp_df.to_excel(writer, sheet_name="auto_interpretation", index=False)
        if not external_predictions.empty:
            external_predictions.to_excel(writer, sheet_name="external_predictions", index=False)
        # Top selection tables, not full if very large.
        for sheet, table in selection_tables.items():
            table.head(args.save_top_n).to_excel(writer, sheet_name=sheet[:31], index=False)
        # Correlation matrices for selected descriptors.
        for sheet, cm in corr_matrix_sheets.items():
            cm.to_excel(writer, sheet_name=("corr_" + sheet)[:31])

    # Text summary.
    with open(out_dir / "REPORT_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("QSPR auto-workflow summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input: {args.excel}\n")
        f.write(f"Split: {split_label}; TS={len(ts_idx)}, VS={len(vs_idx)}\n")
        f.write(f"Target: {args.target_col}\n")
        f.write("\nSelected models ranked by recommendation score (lower is better):\n\n")
        cols_to_print = ["Model", "Descriptors", "Q2_LOO", "RMSECV", "TS_R2", "VS_R2", "VS_RMSE", "max_abs_intercorr", "n_VS_outside_AD", "Recommendation_score_lower_better"]
        f.write(model_summary[cols_to_print].to_string(index=False))
        f.write("\n\nAutomatic interpretation:\n")
        for _, r in model_summary.iterrows():
            f.write(f"- {r['Auto_interpretation']}\n")
        f.write("\nImportant methodological note:\n")
        f.write("External literature values were not used for descriptor selection. If a model matches two external values to hundredths only after tuning, it is not valid external prediction but overfitting.\n")

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print("DONE")
    print(f"Output folder: {out_dir}")
    print(f"Main workbook: {xlsx_path}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print("Best ranked model by transparent recommendation score:")
    if not model_summary.empty:
        r = model_summary.iloc[0]
        print(f"  {r['Model']} | VS_RMSE={r['VS_RMSE']:.3f} | VS_R2={r['VS_R2']:.3f} | Q2_LOO={r['Q2_LOO']:.3f}")
        print(f"  Descriptors: {r['Descriptors']}")
    print("=" * 78)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatic QSPR descriptor selection and diagnostics workflow for PP adsorption.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--excel", required=True, help="Input Excel/CSV file with compounds, target and descriptors.")
    p.add_argument("--sheet", default="PROJEKT", help="Excel sheet name.")
    p.add_argument("--name-col", default="Organic compounds", help="Compound name column.")
    p.add_argument("--target-col", default="EXP LogKda", help="Target column.")
    p.add_argument("--set-col", default="Set", help="Optional TS/VS set column for --split-mode set.")
    p.add_argument("--exclude-cols", default="", help="Comma/semicolon-separated columns to exclude from descriptors, e.g. MW;SMILES.")
    p.add_argument("--out-dir", default="qspr_auto_workflow_outputs", help="Output directory.")

    p.add_argument("--split-mode", choices=["kennard_stone", "sorted_every_third", "set"], default="kennard_stone", help="Train/validation split strategy.")
    p.add_argument("--n-vs", type=int, default=10, help="Number of validation compounds for Kennard-Stone split.")
    p.add_argument("--sorted-start-pos", type=int, default=2, help="Start position for sorted_every_third split.")

    p.add_argument("--var-thresh", type=float, default=1e-8, help="Variance threshold for removing near-constant descriptors.")
    p.add_argument("--duplicate-round-decimals", type=int, default=12, help="Rounding decimals for detecting duplicate numeric descriptor columns.")
    p.add_argument("--corr-y-min", type=float, default=0.30, help="Minimum |Pearson r| with target on TS for candidate descriptors.")
    p.add_argument("--corr-xx-max", type=float, default=0.85, help="Maximum allowed |Pearson r| between descriptors for low-collinearity models.")
    p.add_argument("--required-keep", default="", help="Additional descriptors to force-retain in candidate audit if present.")
    p.add_argument("--enforce-combo-corr", action="store_true", default=True, help="Reject MLR/kNN/SVR combinations above corr_xx_max.")
    p.add_argument("--no-enforce-combo-corr", dest="enforce_combo_corr", action="store_false", help="Do not hard-reject correlated combinations; not recommended for MLR.")

    p.add_argument("--min-desc", type=int, default=2, help="Minimum descriptors for MLR/kNN/SVR.")
    p.add_argument("--max-desc", type=int, default=4, help="Maximum descriptors for MLR/kNN/SVR.")
    p.add_argument("--top-mlr", type=int, default=12, help="Top low-collinearity descriptors for MLR all-subsets.")
    p.add_argument("--top-knn", type=int, default=12, help="Top low-collinearity descriptors for kNN all-subsets.")
    p.add_argument("--top-svr", type=int, default=8, help="Top low-collinearity descriptors for SVR all-subsets. Keep modest; SVR grid is expensive.")
    p.add_argument("--knn-k-min", type=int, default=2, help="Minimum k for kNN search.")
    p.add_argument("--knn-k-max", type=int, default=7, help="Maximum k for kNN search.")
    p.add_argument("--knn-weights", default="uniform,distance", help="Comma-separated kNN weights to test.")

    p.add_argument("--pls-min-desc", type=int, default=5, help="Minimum number of ranked descriptors for PLS.")
    p.add_argument("--top-pls-max", type=int, default=20, help="Maximum number of top descriptors considered for PLS.")
    p.add_argument("--pls-max-lv", type=int, default=8, help="Maximum PLS latent variables.")

    p.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs. Use 7 on an 8-core machine.")
    p.add_argument("--save-top-n", type=int, default=300, help="How many top rows from each selection table to write to Excel/CSV.")

    p.add_argument("--external-file", default="", help="Optional Excel/CSV with new compounds and descriptor columns for prediction.")
    p.add_argument("--external-sheet", default=None, help="Sheet name for external file.")
    p.add_argument("--external-literature-col", default="", help="Optional literature target column in external file for post-hoc comparison only.")
    p.add_argument("--refit-on-all-for-external", action="store_true", help="After validation, refit selected models on all original compounds for external predictions. Default uses TS only.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
        # Some scientific Python/joblib backends can leave non-daemon worker
        # threads alive on Windows/Linux after heavy CV loops. At this point all
        # files are already written and flushed, so exit explicitly.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
