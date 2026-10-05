#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSAR final model diagnostics
============================

What this script does:
- fits selected final/candidate QSAR models on TS,
- evaluates them on TS and VS,
- calculates CV metrics on TS,
- creates predicted-vs-experimental plots,
- creates residual plots,
- creates Williams plots / Applicability Domain tables,
- calculates descriptor intercorrelation matrices,
- saves summary tables to CSV/XLSX.

Default split:
    sorted_every_third by EXP LogKda
    -> sort compounds by target, every third compound goes to VS.

Default models are editable in the MODEL_SPECS block below.

Required packages:
    pip install numpy pandas scikit-learn matplotlib openpyxl

Example:
    python qsar_final_model_diagnostics.py --excel "data_qsar_pls.xlsx"

Optional:
    python qsar_final_model_diagnostics.py --excel "data.xlsx" --sheet PROJEKT --output qsar_diagnostics
    python qsar_final_model_diagnostics.py --excel "data.xlsx" --split-mode set --set-col Set
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================
# 1) EDIT MODELS HERE
# ============================================================
# kind can be: "MLR", "kNN", "SVR", "PLS"
# descriptors must match column names in the Excel file after stripping spaces/non-breaking spaces.
#
# How to change descriptors:
#   descriptors=["log D", "TPSA", "naAromAtom"]
#
# How to add another model:
#   copy one ModelSpec(...) block, change name/kind/descriptors/parameters.
#
# PLS:
#   pls_components=None  -> choose n_components by CV on TS
#   pls_components=1/2/3 -> force exact number of components
#
# kNN:
#   n_neighbors=3 and weights="distance" match our exhaustive scripts.
#
# SVR:
#   C=1.0, epsilon=0.1, gamma="scale" match our fixed SVR protocol.
#
@dataclass
class ModelSpec:
    name: str
    kind: str
    descriptors: List[str]
    # Optional model-specific settings
    n_neighbors: int = 3
    weights: str = "distance"
    svr_C: float = 1.0
    svr_epsilon: float = 0.1
    svr_gamma: str | float = "scale"
    pls_components: Optional[int] = None
    notes: str = ""


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="MLR_uncorrelated",
        kind="MLR",
        descriptors=["AATS4i", "AMR", "ATSC1i", "minwHBa"],
        notes="Best MLR by CV RMSE in 50-descriptor uncorrelated exhaustive run",
    ),
    ModelSpec(
        name="PLS_correlated_SAFE_bestVS",
        kind="PLS",
        descriptors=["log D", "TPSA", "naAromAtom", "AATSC1i"],
        pls_components=4,
        notes="PLS-only correlated run, best VS RMSE among saved top 100; change if you prefer CV-rank-1 PLS",
    ),
    ModelSpec(
        name="EXH50_kNN_uncorrelated_CVrank1",
        kind="kNN",
        descriptors=["GATS1i", "AMR", "nHeteroRing", "SwHBa"],
        n_neighbors=3,
        weights="distance",
        notes="Best kNN by CV RMSE in 50-descriptor uncorrelated exhaustive run",
    ),
    ModelSpec(
        name="GA_SVR_2desc",
        kind="SVR",
        descriptors=["ATS0p", "GATS1i"],
        svr_C=1.0,
        svr_epsilon=0.1,
        svr_gamma="scale",
        notes="Best GA-SVR comparison model",
    ),
]


# ============================================================
# 2) GLOBAL SETTINGS
# ============================================================
TARGET_COL = "EXP LogKda"
NAME_COL = "Organic compounds"
DEFAULT_SHEET = "PROJEKT"
DEFAULT_OUTPUT_DIR = "qsar_final_model_diagnostics_outputs"
RANDOM_SEED = 42
CV_FOLDS = 5

# Williams plot settings
STD_RESIDUAL_LIMIT = 3.0

# If you do not want diagnostic Williams plots for non-linear models, set to False.
# The table is still useful as a descriptor-space AD approximation, but interpret
# leverage more cautiously for kNN/SVR.
DO_WILLIAMS_FOR_ALL_MODELS = True


# ============================================================
# Utilities
# ============================================================
def clean_column_name(col) -> str:
    """Normalize column names from Excel: remove non-breaking spaces and trim."""
    return str(col).replace("\xa0", " ").strip()


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "model"


def suggest_close_columns(name: str, columns: Iterable[str], max_suggestions: int = 8) -> List[str]:
    """Very simple fallback suggestions without extra dependencies."""
    n = name.lower().replace(" ", "")
    cols = list(columns)
    candidates = []
    for c in cols:
        cl = c.lower().replace(" ", "")
        score = 0
        if n == cl:
            score += 100
        if n in cl or cl in n:
            score += 50
        # crude character overlap
        score += len(set(n).intersection(set(cl))) / max(1, len(set(n).union(set(cl))))
        if score > 0.25:
            candidates.append((score, c))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in candidates[:max_suggestions]]


def load_data(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [clean_column_name(c) for c in df.columns]
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Available columns include: {df.columns[:20].tolist()}")
    if NAME_COL not in df.columns:
        print(f"WARNING: name column '{NAME_COL}' not found. A generic compound index will be used.")
        df[NAME_COL] = [f"compound_{i+1}" for i in range(len(df))]
    return df


def validate_model_specs(df: pd.DataFrame, specs: List[ModelSpec]) -> None:
    columns = set(df.columns)
    errors = []
    for spec in specs:
        bad = [d for d in spec.descriptors if d not in columns]
        if bad:
            msg = [f"Model '{spec.name}' has missing descriptors: {bad}"]
            for d in bad:
                suggestions = suggest_close_columns(d, df.columns)
                if suggestions:
                    msg.append(f"  Suggestions for '{d}': {suggestions}")
            errors.append("\n".join(msg))
    if errors:
        raise ValueError("\n\n".join(errors))


def make_split(df: pd.DataFrame, split_mode: str, set_col: str = "Set") -> pd.Series:
    """Return a Series with values 'TS' or 'VS'."""
    split_mode = split_mode.lower().strip()

    if split_mode == "sorted_every_third":
        y = pd.to_numeric(df[TARGET_COL], errors="coerce")
        if y.isna().any():
            bad = df.loc[y.isna(), [NAME_COL, TARGET_COL]]
            raise ValueError(f"Target has non-numeric/missing values:\n{bad}")
        sorted_idx = np.argsort(y.to_numpy())
        valid_positions = np.arange(2, len(sorted_idx), 3)
        idx_valid = set(sorted_idx[valid_positions].tolist())
        split = pd.Series(["VS" if i in idx_valid else "TS" for i in range(len(df))], index=df.index, name="Set")
        return split

    if split_mode == "set":
        if set_col not in df.columns:
            raise ValueError(f"split-mode='set' requested, but column '{set_col}' was not found.")
        vals = df[set_col].astype(str).str.strip().str.upper()
        mapping = {
            "TRAIN": "TS", "TR": "TS", "TS": "TS", "TRAINING": "TS",
            "VALID": "VS", "VALIDATION": "VS", "TEST": "VS", "VS": "VS",
        }
        split = vals.map(mapping)
        if split.isna().any():
            bad_vals = sorted(vals[split.isna()].unique().tolist())
            raise ValueError(f"Unrecognized values in '{set_col}': {bad_vals}. Use TS/VS or Train/Test.")
        split.name = "Set"
        return split

    raise ValueError("split_mode must be 'sorted_every_third' or 'set'.")


def metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def get_numeric_frame(df: pd.DataFrame, descriptors: List[str]) -> pd.DataFrame:
    X = df[descriptors].copy()
    for c in descriptors:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing/non-numeric descriptor values in columns: {bad_cols}")
    return X


def effective_rank(X: np.ndarray, tol: float = 1e-10) -> int:
    if X.size == 0:
        return 0
    return int(np.linalg.matrix_rank(np.asarray(X, dtype=float), tol=tol))


def max_abs_intercorrelation(X: pd.DataFrame) -> float:
    if X.shape[1] <= 1:
        return 0.0
    corr = X.corr(method="pearson").to_numpy(dtype=float)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else 0.0


def build_model(spec: ModelSpec, n_train_samples: int, n_features: int, pls_components: Optional[int] = None) -> Pipeline:
    kind = spec.kind.upper()
    if kind == "MLR":
        estimator = LinearRegression()
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    if kind == "KNN":
        k = min(int(spec.n_neighbors), max(1, n_train_samples))
        estimator = KNeighborsRegressor(n_neighbors=k, weights=spec.weights, metric="minkowski")
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    if kind == "SVR":
        estimator = SVR(kernel="rbf", C=float(spec.svr_C), epsilon=float(spec.svr_epsilon), gamma=spec.svr_gamma, cache_size=500)
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    if kind == "PLS":
        n_comp = pls_components if pls_components is not None else spec.pls_components
        if n_comp is None:
            n_comp = min(2, n_features, max(1, n_train_samples - 1))
        n_comp = int(max(1, min(n_comp, n_features, n_train_samples - 1)))
        estimator = PLSRegression(n_components=n_comp, scale=False)
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    raise ValueError(f"Unknown model kind '{spec.kind}' in spec '{spec.name}'. Use MLR, kNN, SVR, or PLS.")


def predict_flat(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X)).ravel()


def cv_score_model(spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int, seed: int) -> Dict[str, float | int | None]:
    """CV RMSE on TS. For PLS with pls_components=None, choose best components by CV."""
    n = len(X_train)
    n_splits = min(cv_folds, n)
    if n_splits < 2:
        return {"CV_RMSE_mean": np.nan, "CV_RMSE_sd": np.nan, "CV_MAE_mean": np.nan, "CV_R2_mean": np.nan, "selected_pls_components": spec.pls_components}

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def eval_for_components(pls_components: Optional[int]) -> Tuple[List[float], List[float], List[float]]:
        rmses, maes, r2s = [], [], []
        for tr, te in cv.split(X_train):
            X_tr = X_train.iloc[tr]
            X_te = X_train.iloc[te]
            y_tr = y_train.iloc[tr]
            y_te = y_train.iloc[te]
            if spec.kind.upper() == "PLS":
                # Fit scaler first only to estimate fold rank safely.
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                rank = effective_rank(X_tr_scaled)
                max_allowed = max(1, min(X_tr.shape[1], len(tr) - 1, rank))
                pc = pls_components if pls_components is not None else spec.pls_components
                if pc is None:
                    pc = min(2, max_allowed)
                pc = int(max(1, min(pc, max_allowed)))
                model = build_model(spec, n_train_samples=len(tr), n_features=X_tr.shape[1], pls_components=pc)
            else:
                model = build_model(spec, n_train_samples=len(tr), n_features=X_tr.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                model.fit(X_tr, y_tr)
                pred = predict_flat(model, X_te)
            m = metrics(y_te, pred)
            rmses.append(m["RMSE"])
            maes.append(m["MAE"])
            r2s.append(m["R2"])
        return rmses, maes, r2s

    if spec.kind.upper() == "PLS" and spec.pls_components is None:
        # Choose PLS components by lowest mean CV RMSE.
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        rank = effective_rank(Xs)
        max_comp = max(1, min(X_train.shape[1], n - 1, rank))
        best = None
        for pc in range(1, max_comp + 1):
            try:
                rmses, maes, r2s = eval_for_components(pc)
                row = (float(np.mean(rmses)), pc, rmses, maes, r2s)
                if best is None or row[0] < best[0]:
                    best = row
            except Exception as exc:
                print(f"WARNING: PLS CV failed for {spec.name}, n_components={pc}: {exc}")
        if best is None:
            raise RuntimeError(f"All PLS CV component options failed for model {spec.name}")
        _, selected_pc, rmses, maes, r2s = best
        return {
            "CV_RMSE_mean": float(np.mean(rmses)),
            "CV_RMSE_sd": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            "CV_MAE_mean": float(np.mean(maes)),
            "CV_R2_mean": float(np.mean(r2s)),
            "selected_pls_components": int(selected_pc),
        }

    # Non-PLS or fixed PLS components
    rmses, maes, r2s = eval_for_components(spec.pls_components)
    return {
        "CV_RMSE_mean": float(np.mean(rmses)),
        "CV_RMSE_sd": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "CV_MAE_mean": float(np.mean(maes)),
        "CV_R2_mean": float(np.mean(r2s)),
        "selected_pls_components": spec.pls_components,
    }


def leverage_values(X_train: pd.DataFrame, X_all: pd.DataFrame) -> np.ndarray:
    """Calculate leverage h for TS and new samples using standardized descriptor matrix and intercept."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xall = scaler.transform(X_all)

    # Add intercept column for classical leverage with intercept.
    Xtr_aug = np.column_stack([np.ones(Xtr.shape[0]), Xtr])
    Xall_aug = np.column_stack([np.ones(Xall.shape[0]), Xall])

    xtx_inv = np.linalg.pinv(Xtr_aug.T @ Xtr_aug)
    h = np.einsum("ij,jk,ik->i", Xall_aug, xtx_inv, Xall_aug)
    return h


def classical_residual_scale(y_train: np.ndarray, pred_train: np.ndarray, p: int) -> float:
    """Residual standard error used for standardized residuals."""
    resid = np.asarray(y_train, dtype=float) - np.asarray(pred_train, dtype=float)
    n = len(resid)
    df = max(1, n - p - 1)
    sse = float(np.sum(resid ** 2))
    scale = math.sqrt(sse / df)
    if not np.isfinite(scale) or scale <= 1e-12:
        # fallback: avoid division by zero for nearly interpolating models such as kNN
        scale = float(np.std(resid, ddof=1)) if n > 1 else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return scale


def make_williams_table(
    spec: ModelSpec,
    df: pd.DataFrame,
    split: pd.Series,
    X_train: pd.DataFrame,
    X_all: pd.DataFrame,
    y_train: pd.Series,
    pred_all: np.ndarray,
) -> pd.DataFrame:
    p = X_train.shape[1]
    n = X_train.shape[0]
    h_star = 3.0 * (p + 1) / n
    h = leverage_values(X_train, X_all)

    y_all = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    residuals = y_all - pred_all
    pred_train = pred_all[split.values == "TS"]
    scale = classical_residual_scale(y_train.to_numpy(dtype=float), pred_train, p=p)
    standardized_residuals = residuals / scale

    out = pd.DataFrame({
        "Model": spec.name,
        "Model_type": spec.kind,
        "Set": split.values,
        NAME_COL: df[NAME_COL].values,
        TARGET_COL: y_all,
        "Predicted LogKda": pred_all,
        "Residual": residuals,
        "Standardized residual": standardized_residuals,
        "Leverage h": h,
        "Warning leverage h*": h_star,
        "n_descriptors": p,
        "descriptors": "; ".join(spec.descriptors),
        "Outside leverage domain": h > h_star,
        "Outlier residual": np.abs(standardized_residuals) > STD_RESIDUAL_LIMIT,
    })
    out["Outside applicability domain"] = out["Outside leverage domain"] | out["Outlier residual"]
    return out


def save_predicted_vs_experimental_plot(pred_table: pd.DataFrame, spec: ModelSpec, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for set_name, group in pred_table.groupby("Set"):
        ax.scatter(group[TARGET_COL], group["Predicted LogKda"], label=set_name, alpha=0.85)

    min_val = float(min(pred_table[TARGET_COL].min(), pred_table["Predicted LogKda"].min()))
    max_val = float(max(pred_table[TARGET_COL].max(), pred_table["Predicted LogKda"].max()))
    pad = 0.05 * (max_val - min_val if max_val > min_val else 1.0)
    ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], linestyle="--", linewidth=1)
    ax.set_xlabel("Experimental LogKda")
    ax.set_ylabel("Predicted LogKda")
    ax.set_title(f"Predicted vs experimental: {spec.name}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(spec.name)}_predicted_vs_experimental.png", dpi=300)
    plt.close(fig)


def save_residual_plot(pred_table: pd.DataFrame, spec: ModelSpec, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for set_name, group in pred_table.groupby("Set"):
        ax.scatter(group["Predicted LogKda"], group["Residual"], label=set_name, alpha=0.85)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted LogKda")
    ax.set_ylabel("Residual = experimental - predicted")
    ax.set_title(f"Residual plot: {spec.name}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(spec.name)}_residuals_vs_predicted.png", dpi=300)
    plt.close(fig)


def save_williams_plot(williams: pd.DataFrame, spec: ModelSpec, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for set_name, group in williams.groupby("Set"):
        ax.scatter(group["Leverage h"], group["Standardized residual"], label=set_name, alpha=0.85)
    h_star = float(williams["Warning leverage h*"].iloc[0])
    ax.axhline(STD_RESIDUAL_LIMIT, linestyle="--", linewidth=1)
    ax.axhline(-STD_RESIDUAL_LIMIT, linestyle="--", linewidth=1)
    ax.axvline(h_star, linestyle="--", linewidth=1)
    ax.set_xlabel("Leverage h")
    ax.set_ylabel("Standardized residual")
    ax.set_title(f"Williams plot: {spec.name}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(spec.name)}_Williams_plot.png", dpi=300)
    plt.close(fig)


def save_correlation_heatmap(corr: pd.DataFrame, spec: ModelSpec, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(corr.columns)), max(5, 1.1 * len(corr.columns))))
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Descriptor correlation: {spec.name}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_filename(spec.name)}_descriptor_correlation.png", dpi=300)
    plt.close(fig)


def extract_coefficients(spec: ModelSpec, fitted_model: Pipeline, descriptors: List[str]) -> Optional[pd.DataFrame]:
    kind = spec.kind.upper()
    model = fitted_model.named_steps["model"]
    scaler = fitted_model.named_steps["scaler"]

    if kind == "MLR":
        coef_scaled = np.asarray(model.coef_).ravel()
        intercept_scaled = float(model.intercept_)
        # Convert coefficients back to original descriptor scale:
        # y = intercept_scaled + sum coef_scaled_j * ((x_j - mean_j) / scale_j)
        original_coef = coef_scaled / scaler.scale_
        original_intercept = intercept_scaled - np.sum(coef_scaled * scaler.mean_ / scaler.scale_)
        rows = [{"Model": spec.name, "term": "Intercept", "coefficient_scaled_X": intercept_scaled, "coefficient_original_X": original_intercept}]
        for d, cs, co in zip(descriptors, coef_scaled, original_coef):
            rows.append({"Model": spec.name, "term": d, "coefficient_scaled_X": cs, "coefficient_original_X": co})
        return pd.DataFrame(rows)

    if kind == "PLS":
        coef = np.asarray(model.coef_).ravel()
        rows = []
        for d, c in zip(descriptors, coef):
            rows.append({"Model": spec.name, "term": d, "coefficient_on_scaled_X": c})
        return pd.DataFrame(rows)

    return None


def write_excel_safely(output_path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets.items():
                # Excel sheet names max 31 chars
                safe_sheet = re.sub(r"[\\/*?:\[\]]", "_", sheet_name)[:31] or "Sheet"
                df.to_excel(writer, sheet_name=safe_sheet, index=False)
    except Exception as exc:
        print(f"WARNING: Could not write Excel file {output_path}: {exc}")
        print("CSV files were still written.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final QSAR model diagnostics: metrics, predictions, residuals, Williams plot, descriptor correlations.")
    parser.add_argument("--excel", required=True, help="Path to Excel file with QSAR data.")
    parser.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Excel sheet name. Default: {DEFAULT_SHEET}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--split-mode", default="sorted_every_third", choices=["sorted_every_third", "set"], help="How to create TS/VS split.")
    parser.add_argument("--set-col", default="Set", help="Column with TS/VS labels if --split-mode set.")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS, help=f"CV folds inside TS. Default: {CV_FOLDS}")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help=f"Random seed for KFold. Default: {RANDOM_SEED}")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {excel_path}")
    df = load_data(excel_path, args.sheet)
    validate_model_specs(df, MODEL_SPECS)

    split = make_split(df, split_mode=args.split_mode, set_col=args.set_col)
    df = df.copy()
    df["Set_used"] = split.values

    train_mask = split.values == "TS"
    valid_mask = split.values == "VS"
    if train_mask.sum() < 3 or valid_mask.sum() < 1:
        raise ValueError(f"Bad split: TS={train_mask.sum()}, VS={valid_mask.sum()}.")

    print(f"Split: TS={train_mask.sum()}, VS={valid_mask.sum()} using mode={args.split_mode}")
    split_table = df[[NAME_COL, TARGET_COL, "Set_used"]].copy()
    split_table = split_table.sort_values(["Set_used", TARGET_COL], ascending=[True, True])
    split_table.to_csv(output_dir / "TS_VS_split_used.csv", index=False, encoding="utf-8-sig")

    y_all = pd.to_numeric(df[TARGET_COL], errors="coerce")
    y_train = y_all.loc[train_mask].reset_index(drop=True)
    y_valid = y_all.loc[valid_mask].reset_index(drop=True)

    summary_rows = []
    all_predictions = []
    all_williams = []
    coefficient_tables = []
    corr_sheets: Dict[str, pd.DataFrame] = {}

    for spec in MODEL_SPECS:
        print(f"\n=== {spec.name} [{spec.kind}] ===")
        descriptors = spec.descriptors
        X_all_raw = get_numeric_frame(df, descriptors)
        X_train = X_all_raw.loc[train_mask].reset_index(drop=True)
        X_valid = X_all_raw.loc[valid_mask].reset_index(drop=True)
        X_all = X_all_raw.reset_index(drop=True)

        corr_train = X_train.corr(method="pearson")
        corr_sheets[safe_filename(spec.name)[:31]] = corr_train.reset_index().rename(columns={"index": "descriptor"})
        max_corr = max_abs_intercorrelation(X_train)
        rank_train = effective_rank(StandardScaler().fit_transform(X_train))
        print(f"descriptors: {'; '.join(descriptors)}")
        print(f"max |r| on TS: {max_corr:.3f}; effective rank: {rank_train}/{len(descriptors)}")

        # CV on TS
        cv_info = cv_score_model(spec, X_train, y_train, cv_folds=args.cv_folds, seed=args.random_seed)
        selected_pls_components = cv_info.get("selected_pls_components")
        if spec.kind.upper() == "PLS" and selected_pls_components is None:
            # Defensive fallback; should not happen, but keep it safe.
            selected_pls_components = min(2, len(descriptors), len(X_train) - 1)

        # Fit final model on full TS
        final_model = build_model(spec, n_train_samples=len(X_train), n_features=len(descriptors), pls_components=selected_pls_components)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            final_model.fit(X_train, y_train)
        pred_train = predict_flat(final_model, X_train)
        pred_valid = predict_flat(final_model, X_valid)
        pred_all = np.empty(len(df), dtype=float)
        pred_all[train_mask] = pred_train
        pred_all[valid_mask] = pred_valid

        ts_metrics = metrics(y_train, pred_train)
        vs_metrics = metrics(y_valid, pred_valid)

        summary = {
            "Model": spec.name,
            "Model_type": spec.kind,
            "n_descriptors": len(descriptors),
            "descriptors": "; ".join(descriptors),
            "notes": spec.notes,
            "CV_RMSE_mean": cv_info["CV_RMSE_mean"],
            "CV_RMSE_sd": cv_info["CV_RMSE_sd"],
            "CV_MAE_mean": cv_info["CV_MAE_mean"],
            "CV_R2_mean": cv_info["CV_R2_mean"],
            "TS_RMSE": ts_metrics["RMSE"],
            "TS_MAE": ts_metrics["MAE"],
            "TS_R2": ts_metrics["R2"],
            "VS_RMSE": vs_metrics["RMSE"],
            "VS_MAE": vs_metrics["MAE"],
            "VS_R2": vs_metrics["R2"],
            "max_abs_intercorrelation_TS": max_corr,
            "effective_rank_TS": rank_train,
            "selected_pls_components": selected_pls_components if spec.kind.upper() == "PLS" else np.nan,
            "n_TS": int(train_mask.sum()),
            "n_VS": int(valid_mask.sum()),
            "split_mode": args.split_mode,
            "cv_folds": args.cv_folds,
            "random_seed": args.random_seed,
        }
        summary_rows.append(summary)

        print(f"CV RMSE: {summary['CV_RMSE_mean']:.3f} ± {summary['CV_RMSE_sd']:.3f}")
        print(f"TS RMSE/R2: {summary['TS_RMSE']:.3f} / {summary['TS_R2']:.3f}")
        print(f"VS RMSE/R2: {summary['VS_RMSE']:.3f} / {summary['VS_R2']:.3f}")

        pred_table = pd.DataFrame({
            "Model": spec.name,
            "Model_type": spec.kind,
            "Set": split.values,
            NAME_COL: df[NAME_COL].values,
            TARGET_COL: y_all.to_numpy(dtype=float),
            "Predicted LogKda": pred_all,
            "Residual": y_all.to_numpy(dtype=float) - pred_all,
            "n_descriptors": len(descriptors),
            "descriptors": "; ".join(descriptors),
        })
        all_predictions.append(pred_table)

        williams = make_williams_table(spec, df, split, X_train, X_all, y_train, pred_all)
        all_williams.append(williams)

        coef_df = extract_coefficients(spec, final_model, descriptors)
        if coef_df is not None:
            coefficient_tables.append(coef_df)

        save_predicted_vs_experimental_plot(pred_table, spec, output_dir)
        save_residual_plot(pred_table, spec, output_dir)
        if DO_WILLIAMS_FOR_ALL_MODELS or spec.kind.upper() in {"MLR", "PLS"}:
            save_williams_plot(williams, spec, output_dir)
        save_correlation_heatmap(corr_train, spec, output_dir)

    summary_df = pd.DataFrame(summary_rows)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    williams_df = pd.concat(all_williams, ignore_index=True)
    coef_all_df = pd.concat(coefficient_tables, ignore_index=True) if coefficient_tables else pd.DataFrame()

    summary_df = summary_df.sort_values(["VS_RMSE", "CV_RMSE_mean"], ascending=[True, True])

    # CSV outputs
    summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(output_dir / "all_predictions.csv", index=False, encoding="utf-8-sig")
    williams_df.to_csv(output_dir / "all_Williams_AD.csv", index=False, encoding="utf-8-sig")
    if not coef_all_df.empty:
        coef_all_df.to_csv(output_dir / "model_coefficients.csv", index=False, encoding="utf-8-sig")

    # Excel output
    excel_sheets = {
        "model_summary": summary_df,
        "TS_VS_split": split_table,
        "all_predictions": pred_df,
        "Williams_AD": williams_df,
    }
    if not coef_all_df.empty:
        excel_sheets["coefficients"] = coef_all_df
    write_excel_safely(output_dir / "QSAR_final_model_diagnostics.xlsx", excel_sheets)

    # Separate Excel for correlation matrices
    write_excel_safely(output_dir / "descriptor_correlation_matrices.xlsx", corr_sheets)

    # Manifest
    manifest = {
        "excel": str(excel_path),
        "sheet": args.sheet,
        "output": str(output_dir),
        "target_col": TARGET_COL,
        "name_col": NAME_COL,
        "split_mode": args.split_mode,
        "n_TS": int(train_mask.sum()),
        "n_VS": int(valid_mask.sum()),
        "cv_folds": args.cv_folds,
        "random_seed": args.random_seed,
        "models": [spec.__dict__ for spec in MODEL_SPECS],
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Main summary: {output_dir / 'model_summary.csv'}")
    print(f"Main Excel:   {output_dir / 'QSAR_final_model_diagnostics.xlsx'}")
    print(f"Plots saved in: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        sys.exit(1)
