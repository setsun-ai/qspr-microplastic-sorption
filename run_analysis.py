"""Reproduce the final QSPR model for log Kd,a on polypropylene from the input data.

Usage:
    python run_analysis.py            # writes results/final_reproduction/
    python run_analysis.py --quick    # 20 instead of 200 stability splits (used by CI)

The original stage-7 analysis script was not preserved, so this is a reconstruction based on
the final report (final/QSPR_LogKda_report.pdf). It covers:

1. Final MLR {log D, eps_beta} fitted on all 35 compounds (equation, R2, Q2_LOO, VIF).
2. Main 70:30 split (seed 42): MLR, MLR + pi, k-NN, decision tree and RBF-SVR. Scaling,
   hyper-parameter tuning and the descriptor-subset search use the training set only.
3. Stability test: 200 random 70:30 splits (seeds 0-199).
4. External prediction for naphthalene and nitrobenzene with Williams-plot leverage.
5. A comparison of reproduced vs reported numbers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "data_qsar_pls.xlsx"
EXTERNAL = ROOT / "data" / "external_lser.csv"
OUT = ROOT / "results" / "final_reproduction"

SEED = 42
TARGET = "EXP LogKda"
NAME = "Organic compounds"
LSER = ["log D", "Mw'", "εα", "εβ", "V'", "π"]
FINAL = ["log D", "εβ"]

# Numbers printed in final/QSPR_LogKda_report.pdf (section 6-9), used only for the comparison table.
REPORTED = {
    "full_R2": 0.939, "full_Q2_LOO": 0.913,
    "split_MLR_R2_TS": 0.956, "split_MLR_Q2_CV": 0.917, "split_MLR_Q2_ext": 0.737, "split_MLR_RMSEP": 0.398,
    "split_MLR_pi_Q2_ext": 0.748, "split_kNN_k1_Q2_ext": 0.787, "split_DT_Q2_ext": 0.801,
    "stab_MLR_mean": 0.909, "stab_MLR_sd": 0.035, "stab_kNN_mean": 0.730, "stab_kNN_sd": 0.147,
    "stab_DT_mean": 0.797, "stab_DT_sd": 0.213,
    "ext_naphthalene": 3.50, "ext_nitrobenzene": 1.81,
}


# ----------------------------------------------------------------------------- data
def load_data(path: Path = DATA) -> pd.DataFrame:
    """Modelling set: 35 compounds with experimental log Kd,a and the 6 LSER descriptors."""
    df = pd.read_excel(path, sheet_name="PROJEKT")
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df[[NAME, TARGET, *LSER]]


def split(n: int, seed: int = SEED, test_size: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Random train/validation split of row indices."""
    return train_test_split(np.arange(n), test_size=test_size, random_state=seed)


# ----------------------------------------------------------------------------- metrics
def q2(y_true: np.ndarray, y_pred: np.ndarray, y_ref_mean: float) -> float:
    """1 - PRESS / SS around the training mean (Q2_F1 for external sets)."""
    return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_ref_mean) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ----------------------------------------------------------------------------- models
def make_model(kind: str, **params) -> Pipeline:
    """Every model is a Pipeline, so the scaler is always fitted on the data passed to .fit()."""
    est = {
        "MLR": lambda: LinearRegression(),
        "kNN": lambda: KNeighborsRegressor(n_neighbors=params.get("k", 3)),
        "DT": lambda: DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=0),
        "SVR": lambda: SVR(kernel="rbf", **{k: v for k, v in params.items() if k in ("C", "gamma", "epsilon")}),
    }[kind]()
    return Pipeline([("scale", StandardScaler()), ("model", est)])


def tune_knn(X_tr: np.ndarray, y_tr: np.ndarray, ks=range(1, 11)) -> int:
    """k with the highest LOO Q2 on the training set."""
    scores = {k: q2(y_tr, cross_val_predict(make_model("kNN", k=k), X_tr, y_tr, cv=LeaveOneOut()), y_tr.mean())
              for k in ks}
    return max(scores, key=scores.get)


SVR_GRID = {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.1, 1.0], "model__epsilon": [0.05, 0.1, 0.2]}


def tune_svr(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    """RBF-SVR hyper-parameters by 5-fold CV on the training set only."""
    gs = GridSearchCV(make_model("SVR"), SVR_GRID, cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="neg_mean_squared_error")
    gs.fit(X_tr, y_tr)
    return {k.split("__")[1]: v for k, v in gs.best_params_.items()}


def select_subset(X_tr: np.ndarray, y_tr: np.ndarray, names: list[str], max_k: int = 4) -> tuple[list[str], float]:
    """Exhaustive MLR subset search (k = 1..max_k) by LOO Q2, using training rows only.

    Used to check that the a-priori pair {log D, eps_beta} is also what a data-driven
    search on the training set would pick among the top subsets.
    """
    best, best_q2 = None, -np.inf
    for k in range(1, max_k + 1):
        for cols in itertools.combinations(range(len(names)), k):
            Xs = X_tr[:, cols]
            p = cross_val_predict(make_model("MLR"), Xs, y_tr, cv=LeaveOneOut())
            s = q2(y_tr, p, y_tr.mean())
            if s > best_q2 + 1e-9:
                best, best_q2 = [names[c] for c in cols], s
    return best, best_q2


def vif(X: np.ndarray) -> list[float]:
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j]) if others.shape[1] else 0.0
        out.append(float(1 / (1 - r2)))
    return out


def leverage(X_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Hat values h = x (X'X)^-1 x' with an intercept column (Williams plot)."""
    A = np.column_stack([np.ones(len(X_train)), X_train])
    Q = np.column_stack([np.ones(len(X_query)), X_query])
    inv = np.linalg.pinv(A.T @ A)
    return np.einsum("ij,jk,ik->i", Q, inv, Q)


# ----------------------------------------------------------------------------- analysis steps
def final_model(df: pd.DataFrame) -> dict:
    X, y = df[FINAL].to_numpy(float), df[TARGET].to_numpy(float)
    lr = LinearRegression().fit(X, y)
    p_loo = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    n, p = X.shape
    r2 = lr.score(X, y)
    std = make_model("MLR").fit(X, y).named_steps["model"]
    return {
        "n": n,
        "equation_raw": {"intercept": lr.intercept_, **dict(zip(FINAL, lr.coef_, strict=True))},
        "equation_standardised": {"intercept": std.intercept_, **dict(zip(FINAL, std.coef_, strict=True))},
        "R2": r2, "R2_adj": 1 - (1 - r2) * (n - 1) / (n - p - 1),
        "RMSE_fit": rmse(y, lr.predict(X)), "Q2_LOO": q2(y, p_loo, y.mean()),
        "VIF": dict(zip(FINAL, vif(X), strict=True)),
        "h_star": 3 * (p + 1) / n,
    }


def main_split(df: pd.DataFrame, seed: int = SEED) -> dict:
    y = df[TARGET].to_numpy(float)
    tr, te = split(len(df), seed)
    res = {"n_train": len(tr), "n_test": len(te),
           "test_range": [float(y[te].min()), float(y[te].max())]}
    sel, sel_q2 = select_subset(df.loc[tr, LSER].to_numpy(float), y[tr], LSER)
    res["subset_search_on_TS"] = {"best": sel, "Q2_LOO": sel_q2}
    specs = {"MLR {log D, εβ}": ("MLR", FINAL, {}), "MLR {log D, εβ, π}": ("MLR", FINAL + ["π"], {})}
    Xtr = df.loc[tr, FINAL].to_numpy(float)
    k = tune_knn(Xtr, y[tr])
    specs[f"kNN (k = {k})"] = ("kNN", FINAL, {"k": k})
    specs["Decision tree (depth 3, leaf 2)"] = ("DT", FINAL, {})
    svr = tune_svr(Xtr, y[tr])
    specs["SVR (RBF, " + ", ".join(f"{a}={b}" for a, b in svr.items()) + ")"] = ("SVR", FINAL, svr)
    rows = []
    for label, (kind, cols, params) in specs.items():
        Xa = df[cols].to_numpy(float)
        m = make_model(kind, **params).fit(Xa[tr], y[tr])
        p_cv = cross_val_predict(make_model(kind, **params), Xa[tr], y[tr], cv=LeaveOneOut())
        p_tr, p_te = m.predict(Xa[tr]), m.predict(Xa[te])
        rows.append({"model": label, "R2_TS": q2(y[tr], p_tr, y[tr].mean()), "RMSE_TS": rmse(y[tr], p_tr),
                     "Q2_CV": q2(y[tr], p_cv, y[tr].mean()), "Q2_ext": q2(y[te], p_te, y[tr].mean()),
                     "RMSEP": rmse(y[te], p_te), "MAE_ext": float(np.mean(np.abs(y[te] - p_te)))})
    res["table"] = rows
    return res


def stability(df: pd.DataFrame, n_splits: int = 200) -> dict:
    X, y = df[FINAL].to_numpy(float), df[TARGET].to_numpy(float)
    scores = {"MLR": [], "kNN (k = 3)": [], "Decision tree": [], "SVR (tuned per split)": []}
    for s in range(n_splits):
        tr, te = split(len(y), seed=s)
        models = {"MLR": make_model("MLR"), "kNN (k = 3)": make_model("kNN", k=3),
                  "Decision tree": make_model("DT"),
                  "SVR (tuned per split)": make_model("SVR", **tune_svr(X[tr], y[tr]))}
        for name, m in models.items():
            m.fit(X[tr], y[tr])
            scores[name].append(q2(y[te], m.predict(X[te]), y[tr].mean()))
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "median": float(np.median(v)),
                "min": float(np.min(v)), "pct_negative": float(np.mean(np.array(v) < 0) * 100), "values": v}
            for k, v in scores.items()}


def external(df: pd.DataFrame, final: dict) -> list[dict]:
    ext = pd.read_csv(EXTERNAL)
    X = df[FINAL].to_numpy(float)
    lr = LinearRegression().fit(X, df[TARGET].to_numpy(float))
    Xe = ext[FINAL].to_numpy(float)
    h = leverage(X, Xe)
    return [{"compound": c, "log D": float(a), "εβ": float(b), "predicted_logKda": float(p), "leverage": float(hh),
             "inside_AD": bool(hh < final["h_star"])}
            for c, a, b, p, hh in zip(ext["compound"], Xe[:, 0], Xe[:, 1], lr.predict(Xe), h, strict=True)]


def comparison(final: dict, sp: dict, stab: dict, ext: list[dict]) -> pd.DataFrame:
    t = {r["model"]: r for r in sp["table"]}
    knn1 = next((r for k, r in t.items() if k.startswith("kNN (k = 1)")), None)
    mine = {
        "full_R2": final["R2"], "full_Q2_LOO": final["Q2_LOO"],
        "split_MLR_R2_TS": t["MLR {log D, εβ}"]["R2_TS"], "split_MLR_Q2_CV": t["MLR {log D, εβ}"]["Q2_CV"],
        "split_MLR_Q2_ext": t["MLR {log D, εβ}"]["Q2_ext"], "split_MLR_RMSEP": t["MLR {log D, εβ}"]["RMSEP"],
        "split_MLR_pi_Q2_ext": t["MLR {log D, εβ, π}"]["Q2_ext"],
        "split_kNN_k1_Q2_ext": knn1["Q2_ext"] if knn1 else np.nan,
        "split_DT_Q2_ext": t["Decision tree (depth 3, leaf 2)"]["Q2_ext"],
        "stab_MLR_mean": stab["MLR"]["mean"], "stab_MLR_sd": stab["MLR"]["sd"],
        "stab_kNN_mean": stab["kNN (k = 3)"]["mean"], "stab_kNN_sd": stab["kNN (k = 3)"]["sd"],
        "stab_DT_mean": stab["Decision tree"]["mean"], "stab_DT_sd": stab["Decision tree"]["sd"],
        "ext_naphthalene": ext[0]["predicted_logKda"], "ext_nitrobenzene": ext[1]["predicted_logKda"],
    }
    out = pd.DataFrame({"reported": REPORTED, "reproduced": mine}).round(3)
    out["abs_diff"] = (out["reproduced"] - out["reported"]).abs().round(3)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", type=int, default=200, help="number of random splits in the stability test")
    ap.add_argument("--quick", action="store_true", help="20 stability splits (smoke run)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    n_splits = 20 if a.quick else a.splits
    np.random.seed(SEED)

    df = load_data()
    final = final_model(df)
    sp = main_split(df)
    stab = stability(df, n_splits)
    ext = external(df, final)
    cmp_ = comparison(final, sp, stab, ext)

    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sp["table"]).round(3).to_csv(a.out / "main_split_metrics.csv", index=False)
    stab_values = pd.DataFrame({k: v["values"] for k, v in stab.items()}).round(4)
    stab_values.to_csv(a.out / "stability_q2ext.csv", index_label="seed")
    pd.DataFrame(ext).round(3).to_csv(a.out / "external_predictions.csv", index=False)
    cmp_.to_csv(a.out / "reported_vs_reproduced.csv", index_label="quantity")
    summary = {"final_model": final, "main_split": {k: v for k, v in sp.items() if k != "table"},
               "stability": {k: {m: v for m, v in d.items() if m != "values"} for k, d in stab.items()},
               "n_stability_splits": n_splits, "external": ext}
    text = json.dumps(summary, indent=2, ensure_ascii=False, default=float)
    (a.out / "summary.json").write_text(text, encoding="utf-8")

    eq = final["equation_raw"]
    print(f"Final MLR (n = {final['n']}): logKda = {eq['intercept']:.3f} + {eq['log D']:.3f}·logD "
          f"{eq['εβ']:+.2f}·εβ | R2 = {final['R2']:.3f}, Q2_LOO = {final['Q2_LOO']:.3f}")
    print("\nMain split (seed 42, TS = %d, VS = %d):" % (sp["n_train"], sp["n_test"]))
    print(pd.DataFrame(sp["table"]).round(3).to_string(index=False))
    print(f"\nStability, {n_splits} splits (Q2_ext):")
    for k, v in stab.items():
        print(f"  {k:<22} {v['mean']:.3f} ± {v['sd']:.3f}  median {v['median']:.3f}  min {v['min']:.3f}  "
              f"negative {v['pct_negative']:.1f}%")
    print("\nExternal compounds:")
    for e in ext:
        print(f"  {e['compound']:<13} {e['predicted_logKda']:.2f}  h = {e['leverage']:.3f}  "
              f"{'inside AD' if e['inside_AD'] else 'outside AD'} (h* = {final['h_star']:.3f})")
    print("\nReported vs reproduced:\n" + cmp_.to_string())
    return summary


if __name__ == "__main__":
    main()
