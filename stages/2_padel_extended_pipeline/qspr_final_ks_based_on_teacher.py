#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSPR final pipeline based on qspr_v2.py from the lecturer
==========================================================

What this script does:
1. Loads QSAR/QSPR Excel data.
2. Removes descriptors with missing values and near-zero variance.
3. Splits compounds into TS/VS using Kennard-Stone in standardized descriptor space.
4. Applies descriptor pre-filtering on TS only:
   - |r_y| >= threshold
   - pairwise |r_XX| <= threshold, keeping the descriptor with higher |r_y|
5. Scales filtered descriptors using TS parameters and applies the same scaling to VS.
6. Selects and evaluates four model families, following the lecturer's script logic:
   - MLR: all-subsets 2-4 descriptors from TOP-10 |r_y|, max Q2_LOO
   - kNN: all-subsets 2-4 descriptors from TOP-10 at k=3, then tune k=2..7
   - PLS: TOP-15 descriptors by |r_y|, choose LV count by min RMSECV/Q2_LOO
   - SVR: RFE ranking using linear SVR, then RBF SVR grid-search with LOO
7. Saves diagnostics:
   - summary tables
   - TS/VS split
   - predictions and residuals per compound
   - Williams plot / applicability domain tables
   - descriptor correlations
   - PLS VIP
   - predicted vs experimental, residual, Williams, and correlation plots

Required packages:
    pip install numpy pandas scikit-learn matplotlib openpyxl

Example:
    python qspr_final_ks_based_on_teacher.py --excel "MÓJ PROJEKT do Orange - Dane do PROJEKTU qsar.xlsx"

Notes:
- This script intentionally uses Kennard-Stone split, not sorted_every_third.
- MLR, kNN and SVR are restricted to 2-4 descriptors by default.
- PLS follows the lecturer's design: TOP-15 descriptors and latent variables (LVs).
  Therefore PLS is a chemometric comparison model, not a 2-4 raw-descriptor model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, pairwise_distances, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================
# Defaults matching the lecturer's script
# ============================================================
DEFAULT_SHEET = "PROJEKT"
DEFAULT_NAME_COL = "Organic compounds"
DEFAULT_TARGET_COL = "EXP LogKda"
DEFAULT_MW_COL = "MW"
DEFAULT_SET_COL = "Set"
DEFAULT_OUTPUT_DIR = "qspr_final_ks_teacher_outputs"

VAR_THRESH = 1e-8
CORR_Y_MIN = 0.30
CORR_XX_MAX = 0.85
N_VS = 10
MIN_DESC = 2
MAX_DESC = 4
TOP_ALLSUB = 10
TOP_FOR_PLS = 15
TOP_FOR_SVR_RFE = 20
STD_RESIDUAL_LIMIT = 3.0


# ============================================================
# Data classes
# ============================================================
@dataclass
class SelectedModel:
    name: str
    family: str
    descriptors: List[str]
    estimator: object
    selection_Q2_LOO: float
    selection_RMSECV: float
    notes: str
    params: Dict[str, object]
    pls_components: Optional[int] = None


# ============================================================
# Utility functions
# ============================================================
def clean_column_name(col) -> str:
    return str(col).replace("\xa0", " ").strip()


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return text.strip("_") or "model"


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
    }


def effective_rank(X: np.ndarray, tol: float = 1e-10) -> int:
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return 0
    return int(np.linalg.matrix_rank(X, tol=tol))


def max_abs_intercorrelation(X: pd.DataFrame) -> float:
    if X.shape[1] <= 1:
        return 0.0
    corr = X.corr(method="pearson").to_numpy(dtype=float)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else 0.0


def flatten_pred(pred) -> np.ndarray:
    return np.asarray(pred, dtype=float).ravel()


def q2_loo(estimator, X, y) -> Tuple[float, float, np.ndarray]:
    """Return Q2_LOO, RMSECV and LOO predictions.

    Uses explicit LOO rather than cross_val_predict so that invalid PLS variants
    can be caught safely.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    loo = LeaveOneOut()
    pred = np.empty_like(y, dtype=float)

    try:
        for train_idx, test_idx in loo.split(X):
            model = clone(estimator)
            with warnings.catch_warnings():
                # Treat invalid numerical PLS cases as failed candidates.
                warnings.simplefilter("error", RuntimeWarning)
                model.fit(X[train_idx], y[train_idx])
                p = flatten_pred(model.predict(X[test_idx]))
            pred[test_idx[0]] = p[0]
        if not np.all(np.isfinite(pred)):
            return -np.inf, np.inf, pred
        return float(r2_score(y, pred)), rmse(y, pred), pred
    except Exception:
        return -np.inf, np.inf, np.full_like(y, np.nan, dtype=float)


# ============================================================
# Kennard-Stone split
# ============================================================
def kennard_stone(X_scaled: np.ndarray, n_val: int) -> Tuple[List[int], List[int]]:
    """Kennard-Stone split.

    Selects a representative training set that covers descriptor space.
    The remaining objects are used as validation set.
    """
    X_scaled = np.asarray(X_scaled, dtype=float)
    n = X_scaled.shape[0]
    if n_val <= 0 or n_val >= n:
        raise ValueError(f"n_val must be between 1 and n-1. Got n_val={n_val}, n={n}.")

    D = pairwise_distances(X_scaled)
    i, j = np.unravel_index(D.argmax(), D.shape)
    ts = [int(i), int(j)]
    remaining = [k for k in range(n) if k not in ts]

    while len(ts) < (n - n_val):
        min_dists = D[remaining][:, ts].min(axis=1)
        best = remaining[int(min_dists.argmax())]
        ts.append(best)
        remaining.remove(best)

    return sorted(ts), sorted(remaining)


# ============================================================
# Loading + preprocessing
# ============================================================
def load_excel(path: Path, sheet: str, target_col: str, name_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [clean_column_name(c) for c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns include: {df.columns[:30].tolist()}")
    if name_col not in df.columns:
        print(f"WARNING: name column '{name_col}' not found. Generic compound names will be used.")
        df[name_col] = [f"compound_{i+1}" for i in range(len(df))]
    return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    name_col: str,
    mw_col: str,
    set_col: str,
    n_vs: int,
    var_thresh: float,
    corr_y_min: float,
    corr_xx_max: float,
    exclude_cols: Iterable[str],
    output_dir: Path,
) -> Dict[str, object]:
    print("=" * 72)
    print("PHASE A: DATA LOADING + PREPROCESSING + KENNARD-STONE SPLIT")
    print("=" * 72)

    y_all = pd.to_numeric(df[target_col], errors="coerce")
    if y_all.isna().any():
        bad = df.loc[y_all.isna(), [name_col, target_col]]
        raise ValueError(f"Target has missing/non-numeric values:\n{bad}")
    y_all_np = y_all.to_numpy(dtype=float)
    names = df[name_col].astype(str).to_numpy()

    excluded = {target_col, name_col, set_col, *[clean_column_name(c) for c in exclude_cols if str(c).strip()]}
    dcols = [c for c in df.columns if c not in excluded]

    X_raw = df[dcols].apply(pd.to_numeric, errors="coerce")

    # Drop all descriptor columns containing any NaN, matching the lecturer's script.
    cols_with_na = X_raw.columns[X_raw.isna().any()].tolist()
    X_raw = X_raw.drop(columns=cols_with_na)
    print(f"[A1] Removed descriptors with missing/non-numeric values: {len(cols_with_na)}")

    # Near-zero variance filter.
    variances = X_raw.var(axis=0)
    constant_cols = variances[variances <= var_thresh].index.tolist()
    X_raw = X_raw.drop(columns=constant_cols)
    print(f"[A2] Removed constant/near-constant descriptors: {len(constant_cols)}")

    if X_raw.shape[1] < 2:
        raise ValueError("Too few descriptors remain after missing-value and variance filtering.")

    # Kennard-Stone split in full standardized descriptor space.
    sc_full = StandardScaler().fit(X_raw)
    Xs_full = sc_full.transform(X_raw)
    ts_idx, vs_idx = kennard_stone(Xs_full, n_vs)
    print(f"[A3] Kennard-Stone split: TS={len(ts_idx)}, VS={len(vs_idx)}")

    mw_available = mw_col in df.columns
    if mw_available:
        mw = pd.to_numeric(df[mw_col], errors="coerce").to_numpy(dtype=float)
        print(f"     TS MW={np.nanmin(mw[ts_idx]):.1f}-{np.nanmax(mw[ts_idx]):.1f}  "
              f"LogKda={np.nanmin(y_all_np[ts_idx]):.2f}-{np.nanmax(y_all_np[ts_idx]):.2f}")
        print(f"     VS MW={np.nanmin(mw[vs_idx]):.1f}-{np.nanmax(mw[vs_idx]):.1f}  "
              f"LogKda={np.nanmin(y_all_np[vs_idx]):.2f}-{np.nanmax(y_all_np[vs_idx]):.2f}")
    else:
        mw = np.full(len(df), np.nan)
        print(f"     TS LogKda={np.nanmin(y_all_np[ts_idx]):.2f}-{np.nanmax(y_all_np[ts_idx]):.2f}")
        print(f"     VS LogKda={np.nanmin(y_all_np[vs_idx]):.2f}-{np.nanmax(y_all_np[vs_idx]):.2f}")

    print("     VS compounds:")
    for i in vs_idx:
        if mw_available:
            print(f"       {names[i]:45s} MW={mw[i]:8.2f}  LogKda={y_all_np[i]:.3f}")
        else:
            print(f"       {names[i]:45s} LogKda={y_all_np[i]:.3f}")

    # Filters below are TS-only.
    Xdf = pd.DataFrame(X_raw.to_numpy(dtype=float), columns=X_raw.columns)
    y_s = pd.Series(y_all_np)

    # Filter by descriptor-target correlation on TS.
    corr_y = Xdf.iloc[ts_idx].corrwith(y_s.iloc[ts_idx]).abs()
    weak = corr_y[corr_y < corr_y_min].index.tolist()
    Xdf = Xdf.drop(columns=weak)
    print(f"[A4] Filter |r_y| >= {corr_y_min}: removed {len(weak)}, remaining {Xdf.shape[1]}")

    if Xdf.shape[1] < 2:
        raise ValueError("Too few descriptors remain after |r_y| filtering.")

    # Intercorrelation filter on TS: from a highly correlated pair, keep higher |r_y|.
    corr_y2 = Xdf.iloc[ts_idx].corrwith(y_s.iloc[ts_idx]).abs()
    cm = Xdf.iloc[ts_idx].corr(method="pearson").abs()
    order = corr_y2.sort_values(ascending=False).index.tolist()
    drop = set()
    for ii, a in enumerate(order):
        if a in drop:
            continue
        for b in order[ii + 1:]:
            if b in drop:
                continue
            if cm.loc[a, b] > corr_xx_max:
                drop.add(b)
    Xdf = Xdf.drop(columns=sorted(drop))
    print(f"[A5] Filter |r_XX| <= {corr_xx_max}: removed {len(drop)}, remaining {Xdf.shape[1]}")

    if Xdf.shape[1] < 2:
        raise ValueError("Too few descriptors remain after intercorrelation filtering.")

    # Autoscaling: fit on TS only, transform all rows.
    scaler = StandardScaler().fit(Xdf.iloc[ts_idx])
    Xs = pd.DataFrame(scaler.transform(Xdf), columns=Xdf.columns, index=df.index)

    # Ranking by |r_y| on scaled TS.
    corr_rank = Xs.iloc[ts_idx].corrwith(y_s.iloc[ts_idx]).abs().sort_values(ascending=False)

    print("\nTOP 10 candidates by |r_y| on TS:")
    for nm, rv in corr_rank.head(10).items():
        print(f"       {nm:30s} |r|={rv:.3f}")
    print()

    split = pd.Series("TS", index=df.index, name="Set_used")
    split.iloc[vs_idx] = "VS"

    split_table = pd.DataFrame({
        "index_original": np.arange(len(df)),
        name_col: names,
        target_col: y_all_np,
        "Set_used": split.values,
        mw_col: mw,
    })
    split_table.to_csv(output_dir / "TS_VS_split_Kennard_Stone.csv", index=False, encoding="utf-8-sig")

    candidate_table = pd.DataFrame({
        "descriptor": corr_rank.index,
        "abs_r_y_TS": corr_rank.values,
    })
    candidate_table.to_csv(output_dir / "candidate_ranking_after_filters.csv", index=False, encoding="utf-8-sig")

    preprocessing_log = pd.DataFrame([
        {"step": "removed_missing_or_non_numeric", "n_removed": len(cols_with_na), "columns": "; ".join(cols_with_na[:200])},
        {"step": "removed_constant_or_near_constant", "n_removed": len(constant_cols), "columns": "; ".join(constant_cols[:200])},
        {"step": f"removed_abs_r_y_lt_{corr_y_min}", "n_removed": len(weak), "columns": "; ".join(weak[:200])},
        {"step": f"removed_intercorrelated_abs_r_xx_gt_{corr_xx_max}", "n_removed": len(drop), "columns": "; ".join(sorted(drop)[:200])},
        {"step": "remaining_descriptors", "n_removed": 0, "columns": str(Xdf.shape[1])},
    ])
    preprocessing_log.to_csv(output_dir / "preprocessing_log.csv", index=False, encoding="utf-8-sig")

    return {
        "df": df,
        "X_scaled": Xs,
        "X_unscaled_filtered": Xdf,
        "y": y_all_np,
        "names": names,
        "mw": mw,
        "ts_idx": ts_idx,
        "vs_idx": vs_idx,
        "split": split,
        "corr_rank": corr_rank,
        "scaler": scaler,
        "preprocessing_log": preprocessing_log,
    }


# ============================================================
# VIP
# ============================================================
def calculate_vip(pls: PLSRegression) -> np.ndarray:
    """Vectorized VIP calculation.

    VIP > 1 is often interpreted as above-average variable importance.
    """
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).ravel()
    denom = float(np.sum(s))
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.full(p, np.nan)
    w_norm = w / np.linalg.norm(w, axis=0)
    return np.sqrt(p * ((w_norm ** 2) @ s) / denom)


# ============================================================
# Model selection functions
# ============================================================
def all_subsets(
    Xts: pd.DataFrame,
    yts: np.ndarray,
    estimator_factory: Callable[[], object],
    pool: List[str],
    min_d: int,
    max_d: int,
) -> Tuple[List[str], float, float, List[Dict[str, object]]]:
    """All-subsets descriptor selection by maximum Q2_LOO."""
    best_cols: Optional[List[str]] = None
    best_q2 = -np.inf
    best_rmsecv = np.inf
    rows: List[Dict[str, object]] = []

    max_d = min(max_d, len(pool))
    for n_desc in range(min_d, max_d + 1):
        for cols_tuple in itertools.combinations(pool, n_desc):
            cols = list(cols_tuple)
            est = estimator_factory()
            q2, rv, _ = q2_loo(est, Xts[cols].to_numpy(dtype=float), yts)
            rows.append({
                "n_desc": n_desc,
                "descriptors": "; ".join(cols),
                "Q2_LOO": q2,
                "RMSECV": rv,
            })
            if np.isfinite(q2) and q2 > best_q2:
                best_cols = cols
                best_q2 = q2
                best_rmsecv = rv

    if best_cols is None:
        raise RuntimeError("All-subsets selection failed for every candidate.")
    return best_cols, best_q2, best_rmsecv, rows


def select_models(
    Xs: pd.DataFrame,
    y: np.ndarray,
    ts_idx: List[int],
    corr_rank: pd.Series,
    args,
    output_dir: Path,
) -> Tuple[List[SelectedModel], pd.DataFrame, pd.DataFrame]:
    print("=" * 72)
    print("PHASE B: DESCRIPTOR SELECTION")
    print("=" * 72)

    Xts = Xs.iloc[ts_idx]
    yts = y[ts_idx]
    top10 = corr_rank.head(args.top_allsub).index.tolist()
    top15 = corr_rank.head(args.top_pls).index.tolist()
    top_svr = corr_rank.head(args.top_svr_rfe).index.tolist()

    selection_rows: List[Dict[str, object]] = []
    selected: List[SelectedModel] = []

    # MLR
    print(f"\n--- MLR all-subsets {args.min_desc}-{args.max_desc} descriptors from TOP-{args.top_allsub} ---")
    desc_mlr, q2_mlr, rv_mlr, mlr_rows = all_subsets(
        Xts, yts, lambda: LinearRegression(), top10, args.min_desc, args.max_desc
    )
    for row in mlr_rows:
        row["family"] = "MLR"
    selection_rows.extend(mlr_rows)
    print(f"  >> MLR: {len(desc_mlr)} desc | Q2_LOO={q2_mlr:.3f} RMSECV={rv_mlr:.3f}\n     {desc_mlr}")
    selected.append(SelectedModel(
        name="MLR_allsub_TOP10_Q2LOO",
        family="MLR",
        descriptors=desc_mlr,
        estimator=LinearRegression(),
        selection_Q2_LOO=q2_mlr,
        selection_RMSECV=rv_mlr,
        notes="all-subsets 2-4 descriptors from TOP-10 |r_y|; criterion=max Q2_LOO",
        params={},
    ))

    # kNN
    print(f"\n--- kNN all-subsets {args.min_desc}-{args.max_desc} descriptors from TOP-{args.top_allsub}, then k tuning ---")
    desc_knn, q2_knn_start, rv_knn_start, knn_rows = all_subsets(
        Xts, yts, lambda: KNeighborsRegressor(n_neighbors=3), top10, args.min_desc, args.max_desc
    )
    for row in knn_rows:
        row["family"] = "kNN_initial_k3"
    selection_rows.extend(knn_rows)

    best_k = None
    best_q2_knn = -np.inf
    best_rv_knn = np.inf
    knn_k_rows = []
    for k in range(args.knn_k_min, args.knn_k_max + 1):
        est = KNeighborsRegressor(n_neighbors=k)
        q2, rv, _ = q2_loo(est, Xts[desc_knn].to_numpy(dtype=float), yts)
        knn_k_rows.append({"family": "kNN_k_tuning", "k": k, "n_desc": len(desc_knn), "descriptors": "; ".join(desc_knn), "Q2_LOO": q2, "RMSECV": rv})
        if np.isfinite(q2) and q2 > best_q2_knn:
            best_k = k
            best_q2_knn = q2
            best_rv_knn = rv
    selection_rows.extend(knn_k_rows)
    if best_k is None:
        raise RuntimeError("kNN k-tuning failed for every k.")
    print(f"  >> kNN: k={best_k}, {len(desc_knn)} desc | Q2_LOO={best_q2_knn:.3f} RMSECV={best_rv_knn:.3f}\n     {desc_knn}")
    selected.append(SelectedModel(
        name=f"kNN_allsub_TOP10_k{best_k}",
        family="kNN",
        descriptors=desc_knn,
        estimator=KNeighborsRegressor(n_neighbors=best_k),
        selection_Q2_LOO=best_q2_knn,
        selection_RMSECV=best_rv_knn,
        notes="all-subsets 2-4 descriptors at k=3, then k=2..7 tuning by Q2_LOO",
        params={"n_neighbors": best_k, "weights": "uniform"},
    ))

    # PLS TOP-15
    print(f"\n--- PLS TOP-{args.top_pls} descriptors by |r_y|, LV by min RMSECV/max Q2_LOO ---")
    print(f"  Pool: {top15}")
    rank_top15 = effective_rank(Xts[top15].to_numpy(dtype=float))
    max_lv = max(1, min(args.pls_max_lv, len(top15), len(Xts) - 1, rank_top15))
    best_lv = None
    best_q2_pls = -np.inf
    best_rv_pls = np.inf
    pls_lv_rows = []
    for n_lv in range(1, max_lv + 1):
        est = PLSRegression(n_components=n_lv)
        q2, rv, _ = q2_loo(est, Xts[top15].to_numpy(dtype=float), yts)
        pls_lv_rows.append({"family": "PLS_LV_tuning", "n_components": n_lv, "n_desc": len(top15), "descriptors": "; ".join(top15), "Q2_LOO": q2, "RMSECV": rv})
        print(f"    LV={n_lv}: Q2_LOO={q2:.3f} RMSECV={rv:.3f}")
        if np.isfinite(q2) and q2 > best_q2_pls:
            best_lv = n_lv
            best_q2_pls = q2
            best_rv_pls = rv
    selection_rows.extend(pls_lv_rows)
    if best_lv is None:
        raise RuntimeError("PLS LV selection failed for every LV count.")

    pls_fit = PLSRegression(n_components=best_lv).fit(Xts[top15].to_numpy(dtype=float), yts)
    vip = calculate_vip(pls_fit)
    vip_ser = pd.Series(vip, index=top15).sort_values(ascending=False)
    vip_df = pd.DataFrame({"descriptor": vip_ser.index, "VIP": vip_ser.values, "VIP_gt_1": vip_ser.values > 1.0})
    vip_df.to_csv(output_dir / "PLS_VIP_ranking.csv", index=False, encoding="utf-8-sig")
    desc_pls_vip = vip_ser[vip_ser > 1.0].index.tolist()
    print(f"  >> PLS: LV={best_lv} | Q2_LOO={best_q2_pls:.3f} RMSECV={best_rv_pls:.3f}")
    print(f"     VIP>1 ({len(desc_pls_vip)}): {desc_pls_vip}")

    selected.append(SelectedModel(
        name=f"PLS_TOP{args.top_pls}_LV{best_lv}",
        family="PLS",
        descriptors=top15,
        estimator=PLSRegression(n_components=best_lv),
        selection_Q2_LOO=best_q2_pls,
        selection_RMSECV=best_rv_pls,
        notes="TOP-15 descriptors by |r_y|; LV selected by Q2_LOO/RMSECV; PLS model may use correlated predictors",
        params={"n_components": best_lv, "n_raw_descriptors": len(top15)},
        pls_components=best_lv,
    ))

    if args.evaluate_pls_vip and len(desc_pls_vip) >= 1:
        max_vip_lv = max(1, min(args.pls_max_lv, len(desc_pls_vip), len(Xts) - 1, effective_rank(Xts[desc_pls_vip].to_numpy(dtype=float))))
        best_vip_lv = None
        best_vip_q2 = -np.inf
        best_vip_rv = np.inf
        for lv in range(1, max_vip_lv + 1):
            est = PLSRegression(n_components=lv)
            q2, rv, _ = q2_loo(est, Xts[desc_pls_vip].to_numpy(dtype=float), yts)
            selection_rows.append({"family": "PLS_VIP_gt_1_LV_tuning", "n_components": lv, "n_desc": len(desc_pls_vip), "descriptors": "; ".join(desc_pls_vip), "Q2_LOO": q2, "RMSECV": rv})
            if np.isfinite(q2) and q2 > best_vip_q2:
                best_vip_lv = lv
                best_vip_q2 = q2
                best_vip_rv = rv
        if best_vip_lv is not None:
            selected.append(SelectedModel(
                name=f"PLS_VIPgt1_LV{best_vip_lv}",
                family="PLS",
                descriptors=desc_pls_vip,
                estimator=PLSRegression(n_components=best_vip_lv),
                selection_Q2_LOO=best_vip_q2,
                selection_RMSECV=best_vip_rv,
                notes="Additional optional PLS model using only VIP>1 descriptors from TOP-15 model",
                params={"n_components": best_vip_lv, "n_raw_descriptors": len(desc_pls_vip)},
                pls_components=best_vip_lv,
            ))

    # SVR
    print(f"\n--- SVR RFE ranking + RBF grid-search with LOO ---")
    top_svr = top_svr[:min(len(top_svr), Xts.shape[1])]
    if len(top_svr) < args.min_desc:
        raise RuntimeError("Too few descriptors for SVR RFE ranking.")

    rfe = RFE(SVR(kernel="linear"), n_features_to_select=args.min_desc)
    rfe.fit(Xts[top_svr].to_numpy(dtype=float), yts)
    rank_svr = pd.Series(rfe.ranking_, index=top_svr).sort_values(kind="mergesort")
    rank_svr_df = pd.DataFrame({"descriptor": rank_svr.index, "RFE_rank": rank_svr.values})
    rank_svr_df.to_csv(output_dir / "SVR_RFE_ranking.csv", index=False, encoding="utf-8-sig")
    print(f"  RFE ranking top-8: {rank_svr.head(8).index.tolist()}")

    grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1],
        "epsilon": [0.05, 0.1, 0.2],
    }
    best_svr_cols = None
    best_svr_params = None
    best_svr_q2 = -np.inf
    best_svr_rv = np.inf
    svr_rows = []
    for n_d in range(args.min_desc, min(args.max_desc, len(rank_svr)) + 1):
        cols = rank_svr.head(n_d).index.tolist()
        gs = GridSearchCV(
            SVR(kernel="rbf"),
            grid,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
            n_jobs=args.n_jobs,
        )
        gs.fit(Xts[cols].to_numpy(dtype=float), yts)
        best_est = SVR(kernel="rbf", **gs.best_params_)
        q2, rv, _ = q2_loo(best_est, Xts[cols].to_numpy(dtype=float), yts)
        print(f"    n_desc={n_d}: Q2_LOO={q2:.3f} RMSECV={rv:.3f} params={gs.best_params_} desc={cols}")
        row = {"family": "SVR_RFE_grid", "n_desc": n_d, "descriptors": "; ".join(cols), "Q2_LOO": q2, "RMSECV": rv, **{f"param_{k}": v for k, v in gs.best_params_.items()}}
        svr_rows.append(row)
        if np.isfinite(q2) and q2 > best_svr_q2:
            best_svr_cols = cols
            best_svr_params = gs.best_params_
            best_svr_q2 = q2
            best_svr_rv = rv
    selection_rows.extend(svr_rows)
    if best_svr_cols is None or best_svr_params is None:
        raise RuntimeError("SVR selection failed for every descriptor count.")

    print(f"  >> SVR: {len(best_svr_cols)} desc | Q2_LOO={best_svr_q2:.3f} RMSECV={best_svr_rv:.3f}\n     params={best_svr_params}\n     {best_svr_cols}")
    selected.append(SelectedModel(
        name="SVR_RFE_RBF_grid",
        family="SVR",
        descriptors=best_svr_cols,
        estimator=SVR(kernel="rbf", **best_svr_params),
        selection_Q2_LOO=best_svr_q2,
        selection_RMSECV=best_svr_rv,
        notes="RFE ranking on linear SVR; descriptor count 2-4; RBF SVR grid-search by LOO",
        params=best_svr_params,
    ))

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(output_dir / "descriptor_selection_details.csv", index=False, encoding="utf-8-sig")
    return selected, selection_df, vip_df


# ============================================================
# Diagnostics
# ============================================================
def leverage_from_matrix(X_train: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    X_all = np.asarray(X_all, dtype=float)
    Xtr_aug = np.column_stack([np.ones(X_train.shape[0]), X_train])
    Xall_aug = np.column_stack([np.ones(X_all.shape[0]), X_all])
    xtx_inv = np.linalg.pinv(Xtr_aug.T @ Xtr_aug)
    return np.einsum("ij,jk,ik->i", Xall_aug, xtx_inv, Xall_aug)


def residual_scale(y_train: np.ndarray, pred_train: np.ndarray, p_eff: int) -> float:
    resid = np.asarray(y_train, dtype=float) - np.asarray(pred_train, dtype=float)
    df = max(1, len(resid) - p_eff - 1)
    scale = math.sqrt(float(np.sum(resid ** 2)) / df)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return scale


def calculate_williams_table(
    model_info: SelectedModel,
    fitted_model,
    Xts: pd.DataFrame,
    Xall: pd.DataFrame,
    y_all: np.ndarray,
    pred_all: np.ndarray,
    split: pd.Series,
    names: np.ndarray,
    target_col: str,
    name_col: str,
) -> pd.DataFrame:
    train_mask = split.values == "TS"
    y_train = y_all[train_mask]
    pred_train = pred_all[train_mask]

    # For PLS use latent score space for leverage; for the others use descriptor space.
    if model_info.family.upper() == "PLS":
        try:
            T_train = fitted_model.transform(Xts.to_numpy(dtype=float))
            T_all = fitted_model.transform(Xall.to_numpy(dtype=float))
            p_eff = int(model_info.pls_components or T_train.shape[1])
            Xlev_train = T_train[:, :p_eff]
            Xlev_all = T_all[:, :p_eff]
            leverage_space = "PLS latent-score space"
        except Exception:
            p_eff = len(model_info.descriptors)
            Xlev_train = Xts.to_numpy(dtype=float)
            Xlev_all = Xall.to_numpy(dtype=float)
            leverage_space = "descriptor space fallback"
    else:
        p_eff = len(model_info.descriptors)
        Xlev_train = Xts.to_numpy(dtype=float)
        Xlev_all = Xall.to_numpy(dtype=float)
        leverage_space = "descriptor space"

    h = leverage_from_matrix(Xlev_train, Xlev_all)
    h_star = 3.0 * (p_eff + 1) / len(Xts)
    residuals = y_all - pred_all
    scale = residual_scale(y_train, pred_train, p_eff=p_eff)
    std_resid = residuals / scale

    out = pd.DataFrame({
        "Model": model_info.name,
        "Family": model_info.family,
        "Set": split.values,
        name_col: names,
        target_col: y_all,
        "Predicted LogKda": pred_all,
        "Residual": residuals,
        "Standardized residual": std_resid,
        "Leverage h": h,
        "Warning leverage h*": h_star,
        "Leverage space": leverage_space,
        "p_eff_for_h_star": p_eff,
        "n_descriptors": len(model_info.descriptors),
        "descriptors": "; ".join(model_info.descriptors),
        "Outside leverage domain": h > h_star,
        "Outlier residual": np.abs(std_resid) > STD_RESIDUAL_LIMIT,
    })
    out["Outside applicability domain"] = out["Outside leverage domain"] | out["Outlier residual"]
    return out


def save_predicted_vs_experimental(pred_table: pd.DataFrame, model_name: str, target_col: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for set_name, group in pred_table.groupby("Set"):
        ax.scatter(group[target_col], group["Predicted LogKda"], label=set_name, alpha=0.85)
    min_val = float(min(pred_table[target_col].min(), pred_table["Predicted LogKda"].min()))
    max_val = float(max(pred_table[target_col].max(), pred_table["Predicted LogKda"].max()))
    pad = 0.05 * (max_val - min_val if max_val > min_val else 1.0)
    ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], linestyle="--", linewidth=1)
    ax.set_xlabel("Experimental LogKda")
    ax.set_ylabel("Predicted LogKda")
    ax.set_title(f"Predicted vs experimental: {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(model_name)}_predicted_vs_experimental.png", dpi=300)
    plt.close(fig)


def save_residual_plot(pred_table: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for set_name, group in pred_table.groupby("Set"):
        ax.scatter(group["Predicted LogKda"], group["Residual"], label=set_name, alpha=0.85)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted LogKda")
    ax.set_ylabel("Residual = experimental - predicted")
    ax.set_title(f"Residual plot: {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(model_name)}_residuals_vs_predicted.png", dpi=300)
    plt.close(fig)


def save_williams_plot(williams: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for set_name, group in williams.groupby("Set"):
        ax.scatter(group["Leverage h"], group["Standardized residual"], label=set_name, alpha=0.85)
    h_star = float(williams["Warning leverage h*"].iloc[0])
    ax.axhline(STD_RESIDUAL_LIMIT, linestyle="--", linewidth=1)
    ax.axhline(-STD_RESIDUAL_LIMIT, linestyle="--", linewidth=1)
    ax.axvline(h_star, linestyle="--", linewidth=1)
    ax.set_xlabel("Leverage h")
    ax.set_ylabel("Standardized residual")
    ax.set_title(f"Williams plot: {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(model_name)}_Williams_plot.png", dpi=300)
    plt.close(fig)


def save_correlation_heatmap(corr: pd.DataFrame, model_name: str, output_dir: Path) -> None:
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(corr.columns)), max(5, 1.0 * len(corr.columns))))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    if len(corr.columns) <= 20:
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Descriptor correlation on TS: {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(model_name)}_descriptor_correlation.png", dpi=300)
    plt.close(fig)


def write_excel(output_path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_sheet = re.sub(r"[\\/*?:\[\]]", "_", str(sheet_name))[:31] or "Sheet"
            df.to_excel(writer, sheet_name=safe_sheet, index=False)


def evaluate_models(
    selected: List[SelectedModel],
    data: Dict[str, object],
    target_col: str,
    name_col: str,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    print("=" * 72)
    print("PHASE C: FINAL EVALUATION + DIAGNOSTICS")
    print("=" * 72)

    Xs: pd.DataFrame = data["X_scaled"]
    y: np.ndarray = data["y"]
    names: np.ndarray = data["names"]
    ts_idx: List[int] = data["ts_idx"]
    vs_idx: List[int] = data["vs_idx"]
    split: pd.Series = data["split"]

    train_mask = split.values == "TS"
    valid_mask = split.values == "VS"

    summary_rows: List[Dict[str, object]] = []
    pred_tables: List[pd.DataFrame] = []
    williams_tables: List[pd.DataFrame] = []
    corr_sheets: Dict[str, pd.DataFrame] = {}
    coefficient_rows: List[pd.DataFrame] = []

    for model_info in selected:
        print(f"\n--- {model_info.name} [{model_info.family}] ---")
        cols = model_info.descriptors
        X_all = Xs[cols]
        Xts = X_all.iloc[ts_idx]
        Xvs = X_all.iloc[vs_idx]
        yts = y[ts_idx]
        yvs = y[vs_idx]

        est = clone(model_info.estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            est.fit(Xts.to_numpy(dtype=float), yts)
            pred_ts = flatten_pred(est.predict(Xts.to_numpy(dtype=float)))
            pred_vs = flatten_pred(est.predict(Xvs.to_numpy(dtype=float)))
            pred_all = flatten_pred(est.predict(X_all.to_numpy(dtype=float)))

        q2, rmsecv, loo_pred = q2_loo(clone(model_info.estimator), Xts.to_numpy(dtype=float), yts)
        m_ts = metrics(yts, pred_ts)
        m_vs = metrics(yvs, pred_vs)
        corr_train = Xts.corr(method="pearson")
        max_corr = max_abs_intercorrelation(Xts)
        rank = effective_rank(Xts.to_numpy(dtype=float))

        print(f"  descriptors: {'; '.join(cols)}")
        print(f"  Q2_LOO/RMSECV: {q2:.3f}/{rmsecv:.3f}")
        print(f"  TS RMSE/R2:    {m_ts['RMSE']:.3f}/{m_ts['R2']:.3f}")
        print(f"  VS RMSEP/R2:   {m_vs['RMSE']:.3f}/{m_vs['R2']:.3f}")
        print(f"  max |r_XX| TS: {max_corr:.3f}; effective rank={rank}/{len(cols)}")

        summary_rows.append({
            "Model": model_info.name,
            "Family": model_info.family,
            "n_descriptors": len(cols),
            "descriptors": "; ".join(cols),
            "selection_Q2_LOO": model_info.selection_Q2_LOO,
            "selection_RMSECV": model_info.selection_RMSECV,
            "Q2_LOO_recalculated": q2,
            "RMSECV_recalculated": rmsecv,
            "TS_RMSE": m_ts["RMSE"],
            "TS_MAE": m_ts["MAE"],
            "TS_R2": m_ts["R2"],
            "VS_RMSEP": m_vs["RMSE"],
            "VS_MAE": m_vs["MAE"],
            "VS_R2": m_vs["R2"],
            "max_abs_intercorrelation_TS": max_corr,
            "effective_rank_TS": rank,
            "n_TS": int(len(ts_idx)),
            "n_VS": int(len(vs_idx)),
            "params": json.dumps(model_info.params, ensure_ascii=False),
            "notes": model_info.notes,
        })

        pred_table = pd.DataFrame({
            "Model": model_info.name,
            "Family": model_info.family,
            "Set": split.values,
            name_col: names,
            target_col: y,
            "Predicted LogKda": pred_all,
            "Residual": y - pred_all,
            "n_descriptors": len(cols),
            "descriptors": "; ".join(cols),
        })
        pred_tables.append(pred_table)

        # LOO predictions table for TS only.
        loo_df = pd.DataFrame({
            "Model": model_info.name,
            "Family": model_info.family,
            "index_original": np.asarray(ts_idx, dtype=int),
            name_col: names[ts_idx],
            target_col: yts,
            "LOO_predicted": loo_pred,
            "LOO_residual": yts - loo_pred,
        })
        loo_df.to_csv(output_dir / f"{safe_filename(model_info.name)}_LOO_predictions_TS.csv", index=False, encoding="utf-8-sig")

        williams = calculate_williams_table(model_info, est, Xts, X_all, y, pred_all, split, names, target_col, name_col)
        williams_tables.append(williams)

        corr_sheets[safe_filename(model_info.name)[:31]] = corr_train.reset_index().rename(columns={"index": "descriptor"})

        # Coefficients for MLR; PLS coefficients are on scaled X-space.
        if model_info.family.upper() == "MLR":
            coef = np.asarray(est.coef_).ravel()
            coefficient_rows.append(pd.DataFrame({
                "Model": model_info.name,
                "term": ["Intercept"] + cols,
                "coefficient_on_scaled_X": [float(est.intercept_)] + [float(c) for c in coef],
            }))
        elif model_info.family.upper() == "PLS":
            coef = np.asarray(est.coef_).ravel()
            coefficient_rows.append(pd.DataFrame({
                "Model": model_info.name,
                "term": cols[:len(coef)],
                "coefficient_on_scaled_X": [float(c) for c in coef],
            }))

        save_predicted_vs_experimental(pred_table, model_info.name, target_col, output_dir)
        save_residual_plot(pred_table, model_info.name, output_dir)
        save_williams_plot(williams, model_info.name, output_dir)
        save_correlation_heatmap(corr_train, model_info.name, output_dir)

    summary_df = pd.DataFrame(summary_rows).sort_values(["VS_RMSEP", "RMSECV_recalculated"], ascending=[True, True])
    predictions_df = pd.concat(pred_tables, ignore_index=True)
    williams_df = pd.concat(williams_tables, ignore_index=True)
    coefficients_df = pd.concat(coefficient_rows, ignore_index=True) if coefficient_rows else pd.DataFrame()

    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(output_dir / "all_predictions.csv", index=False, encoding="utf-8-sig")
    williams_df.to_csv(output_dir / "all_Williams_AD.csv", index=False, encoding="utf-8-sig")
    if not coefficients_df.empty:
        coefficients_df.to_csv(output_dir / "model_coefficients.csv", index=False, encoding="utf-8-sig")

    # Correlation matrices workbook.
    write_excel(output_dir / "descriptor_correlation_matrices.xlsx", corr_sheets)

    return {
        "model_summary": summary_df,
        "all_predictions": predictions_df,
        "Williams_AD": williams_df,
        "coefficients": coefficients_df,
    }


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Final QSPR pipeline based on lecturer's Kennard-Stone + LOO script, extended with diagnostics.")
    parser.add_argument("--excel", required=True, help="Path to Excel file with QSAR/QSPR data.")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Excel sheet name. Default: {DEFAULT_SHEET}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--name-col", default=DEFAULT_NAME_COL, help=f"Compound-name column. Default: {DEFAULT_NAME_COL}")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help=f"Target column. Default: {DEFAULT_TARGET_COL}")
    parser.add_argument("--mw-col", default=DEFAULT_MW_COL, help=f"Molecular-weight column for reporting. Default: {DEFAULT_MW_COL}")
    parser.add_argument("--set-col", default=DEFAULT_SET_COL, help=f"Set/meta column to exclude. Default: {DEFAULT_SET_COL}")
    parser.add_argument("--exclude-cols", default="", help="Comma-separated extra columns to exclude from descriptor pool.")

    parser.add_argument("--n-vs", type=int, default=N_VS, help=f"Number of validation compounds for Kennard-Stone. Default: {N_VS}")
    parser.add_argument("--var-thresh", type=float, default=VAR_THRESH, help=f"Variance threshold. Default: {VAR_THRESH}")
    parser.add_argument("--corr-y-min", type=float, default=CORR_Y_MIN, help=f"Minimum |r_y| on TS. Default: {CORR_Y_MIN}")
    parser.add_argument("--corr-xx-max", type=float, default=CORR_XX_MAX, help=f"Maximum allowed |r_XX| on TS. Default: {CORR_XX_MAX}")
    parser.add_argument("--min-desc", type=int, default=MIN_DESC, help=f"Minimum descriptors for all-subsets models. Default: {MIN_DESC}")
    parser.add_argument("--max-desc", type=int, default=MAX_DESC, help=f"Maximum descriptors for MLR/kNN/SVR. Default: {MAX_DESC}")
    parser.add_argument("--top-allsub", type=int, default=TOP_ALLSUB, help=f"TOP-k descriptors for MLR/kNN all-subsets. Default: {TOP_ALLSUB}")
    parser.add_argument("--top-pls", type=int, default=TOP_FOR_PLS, help=f"TOP-k descriptors for PLS. Default: {TOP_FOR_PLS}")
    parser.add_argument("--top-svr-rfe", type=int, default=TOP_FOR_SVR_RFE, help=f"TOP-k descriptors for SVR RFE. Default: {TOP_FOR_SVR_RFE}")
    parser.add_argument("--pls-max-lv", type=int, default=8, help="Maximum tested PLS latent variables. Default: 8")
    parser.add_argument("--evaluate-pls-vip", action="store_true", help="Also evaluate an optional PLS model using descriptors with VIP > 1.")
    parser.add_argument("--knn-k-min", type=int, default=2, help="Minimum k for kNN tuning. Default: 2")
    parser.add_argument("--knn-k-max", type=int, default=7, help="Maximum k for kNN tuning. Default: 7")
    parser.add_argument("--n-jobs", type=int, default=-1, help="n_jobs for SVR GridSearchCV. Default: -1")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_cols = [x.strip() for x in args.exclude_cols.split(",") if x.strip()]

    print(f"Loading: {args.excel}")
    df = load_excel(Path(args.excel), args.sheet, args.target_col, args.name_col)

    data = prepare_data(
        df=df,
        target_col=args.target_col,
        name_col=args.name_col,
        mw_col=args.mw_col,
        set_col=args.set_col,
        n_vs=args.n_vs,
        var_thresh=args.var_thresh,
        corr_y_min=args.corr_y_min,
        corr_xx_max=args.corr_xx_max,
        exclude_cols=exclude_cols,
        output_dir=output_dir,
    )

    selected, selection_df, vip_df = select_models(
        Xs=data["X_scaled"],
        y=data["y"],
        ts_idx=data["ts_idx"],
        corr_rank=data["corr_rank"],
        args=args,
        output_dir=output_dir,
    )

    diagnostics = evaluate_models(
        selected=selected,
        data=data,
        target_col=args.target_col,
        name_col=args.name_col,
        output_dir=output_dir,
    )

    # Main workbook.
    split_table = pd.read_csv(output_dir / "TS_VS_split_Kennard_Stone.csv")
    candidate_rank = pd.read_csv(output_dir / "candidate_ranking_after_filters.csv")
    preprocessing_log = pd.read_csv(output_dir / "preprocessing_log.csv")
    sheets = {
        "model_summary": diagnostics["model_summary"],
        "TS_VS_split": split_table,
        "candidate_ranking": candidate_rank,
        "selection_details": selection_df,
        "PLS_VIP": vip_df,
        "all_predictions": diagnostics["all_predictions"],
        "Williams_AD": diagnostics["Williams_AD"],
        "preprocessing_log": preprocessing_log,
    }
    if not diagnostics["coefficients"].empty:
        sheets["coefficients"] = diagnostics["coefficients"]
    write_excel(output_dir / "QSPR_final_Kennard_Stone_diagnostics.xlsx", sheets)

    # Human-readable summary text.
    with open(output_dir / "wyniki_qspr_final_ks.txt", "w", encoding="utf-8") as f:
        f.write("QSPR final pipeline based on lecturer's Kennard-Stone methodology\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Excel: {args.excel}\nSheet: {args.sheet}\nOutput: {output_dir}\n")
        f.write(f"Split: Kennard-Stone, TS={len(data['ts_idx'])}, VS={len(data['vs_idx'])}\n")
        f.write(f"Filters: var>{args.var_thresh}, |r_y|>={args.corr_y_min}, |r_XX|<={args.corr_xx_max}\n\n")
        f.write("Selected models and metrics:\n")
        f.write(diagnostics["model_summary"].to_string(index=False))
        f.write("\n\nPLS VIP ranking:\n")
        f.write(vip_df.to_string(index=False))

    manifest = {
        "script": "qspr_final_ks_based_on_teacher.py",
        "excel": args.excel,
        "sheet": args.sheet,
        "methodology": "Kennard-Stone split + TS-only descriptor filters + LOO model selection, based on qspr_v2.py",
        "args": vars(args),
        "selected_models": [],
    }
    for m in selected:
        manifest["selected_models"].append({
            "name": m.name,
            "family": m.family,
            "descriptors": m.descriptors,
            "selection_Q2_LOO": m.selection_Q2_LOO,
            "selection_RMSECV": m.selection_RMSECV,
            "notes": m.notes,
            "params": m.params,
            "pls_components": m.pls_components,
            "estimator": str(m.estimator),
        })
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Main Excel:   {output_dir / 'QSPR_final_Kennard_Stone_diagnostics.xlsx'}")
    print(f"Summary CSV:  {output_dir / 'model_summary.csv'}")
    print(f"Summary TXT:  {output_dir / 'wyniki_qspr_final_ks.txt'}")
    print(f"Plots folder: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
