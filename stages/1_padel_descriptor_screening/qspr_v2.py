"""
===========================================================================
 QSPR v2 — selekcja deskryptorów + podział TS/VS + 4 modele
 MLR | kNN | PLS | SVR
===========================================================================
 Dane: data_qsar_pls.xlsx  (35 związków, ~2000 deskryptorów, y = EXP LogKd)

 Kluczowe decyzje metodologiczne (cytowalne):
   - Podział TS/VS: algorytm Kennard-Stone na znorm. przestrzeni deskryptorów
     → TS maksymalnie "obejmuje" VS, minimalne ryzyko ekstrapolacji
     (Kennard & Stone 1969; Tropsha 2010)
   - Pre-processing: filtr wariancji + filtr |r_y| ≥ 0.30 + filtr
     współliniowości |r_XX| ≤ 0.85 → z ~2000 do ~100 kandydatów
     (OECD 2007 Guidance Document on QSAR)
   - MLR:  all-subsets 2–4 desc z TOP-10 |r_y|, kryterium max Q²_LOO
     (Tropsha 2010 — all-subsets + LOO to złoty standard dla MLR QSAR)
   - PLS:  TOP-15 desc wg |r_y|  (PLS "lubi" korelujące predyktory —
     wyciąga wspólny sygnał przez ukryte składowe LV), LV wg min RMSECV,
     ważność zmiennych przez VIP (Wold et al. 2001)
   - kNN:  all-subsets 2–4 desc z TOP-10 |r_y| (k=3 startowe), potem
     dobór k=2..7 na wybranych deskryptorach — 2-etapowa procedura
     redukuje liczbę iteracji o 2 rzędy wielkości (Tropsha 2010)
   - SVR:  RFE na liniowym SVR → ranking → zbiór 2–4 desc,
     grid-search C/γ/ε (jądro RBF) z LOO (Guyon & Elisseeff 2003)
===========================================================================
 Wymagania:  pip install pandas numpy scikit-learn openpyxl
===========================================================================
"""

import warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import pairwise_distances

# ─── USTAWIENIA ────────────────────────────────────────────────────────────
PLIK          = "data_qsar_pls.xlsx"
ARKUSZ        = "PROJEKT"
KOL_NAZWA     = "Organic compounds"
KOL_Y         = "EXP LogKda"
KOL_MW        = "MW"

VAR_THRESH    = 1e-8   # próg wariancji — usuń deskryptory stałe
CORR_Y_MIN    = 0.30   # min |r z y| kandydata
CORR_XX_MAX   = 0.70   # max |r między X| — filtr współliniowości

N_VS          = 10     # liczba związków w VS
MIN_DESC      = 2      # min deskryptory w modelu
MAX_DESC      = 4      # max deskryptory w modelu
TOP_ALLSUB    = 20     # pula kandydatów do all-subsets (MLR, kNN)
TOP_FOR_PLS   = 4     # pula kandydatów dla PLS
# ───────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════
#  METRYKI
# ══════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def q2_loo(estimator, X, y):
    """Q² i RMSECV z Leave-One-Out — na zbiorze uczącym."""
    y_pred = cross_val_predict(estimator, X, y, cv=LeaveOneOut())
    return r2_score(y, y_pred), rmse(y, y_pred)


# ══════════════════════════════════════════════════════════════════════════
#  KENNARD-STONE
# ══════════════════════════════════════════════════════════════════════════

def kennard_stone(X_scaled, n_val):
    """
    Kennard-Stone (1969): wybiera n_val obiektów do VS tak,
    by TS równomiernie "pokrywał" przestrzeń chemiczną.
    Zaczyna od dwóch najdalszych punktów → TS, potem sekwencyjnie
    dodaje punkt najdalszy od dotychczasowego TS.
    """
    n = X_scaled.shape[0]
    D = pairwise_distances(X_scaled)
    i, j = np.unravel_index(D.argmax(), D.shape)
    ts = [int(i), int(j)]
    remaining = [k for k in range(n) if k not in ts]
    while len(ts) < (n - n_val):
        min_dists = D[remaining][:, ts].min(axis=1)
        best = remaining[int(min_dists.argmax())]
        ts.append(best)
        remaining.remove(best)
    return sorted(ts), sorted(remaining)  # remaining = VS


# ══════════════════════════════════════════════════════════════════════════
#  FAZA A: WCZYTANIE + PRE-PROCESSING + PODZIAŁ KS
# ══════════════════════════════════════════════════════════════════════════

def wczytaj_i_przygotuj():
    print("="*68)
    print(" FAZA A: WCZYTANIE + PRE-PROCESSING")
    print("="*68)

    df    = pd.read_excel(PLIK, sheet_name=ARKUSZ)
    meta  = [KOL_NAZWA, KOL_Y, "Set"]
    dcols = [c for c in df.columns if c not in meta]

    X_raw = df[dcols].apply(pd.to_numeric, errors="coerce")
    y_all = pd.to_numeric(df[KOL_Y], errors="coerce").values
    names = df[KOL_NAZWA].values
    mw    = pd.to_numeric(df[KOL_MW], errors="coerce").values

    # A1: braki
    brak  = X_raw.columns[X_raw.isna().any()]
    X_raw = X_raw.drop(columns=brak)
    print(f"[A1] Usunięto {len(brak):4d} deskryptorów z brakami.")

    # A2: zerowa wariancja
    stale = X_raw.columns[X_raw.var() <= VAR_THRESH]
    X_raw = X_raw.drop(columns=stale)
    print(f"[A2] Usunięto {len(stale):4d} deskryptorów stałych.")

    # A3: Kennard-Stone na pełnej znormalizowanej przestrzeni
    sc_full  = StandardScaler().fit(X_raw)
    Xs_full  = sc_full.transform(X_raw)
    ts_idx, vs_idx = kennard_stone(Xs_full, N_VS)
    print(f"\n[A3] Kennard-Stone: TS={len(ts_idx)}, VS={len(vs_idx)}")
    print(f"     TS  MW={mw[ts_idx].min():.1f}–{mw[ts_idx].max():.1f}  "
          f"LogKd={y_all[ts_idx].min():.2f}–{y_all[ts_idx].max():.2f}")
    print(f"     VS  MW={mw[vs_idx].min():.1f}–{mw[vs_idx].max():.1f}  "
          f"LogKd={y_all[vs_idx].min():.2f}–{y_all[vs_idx].max():.2f}")
    print("     Związki w VS:")
    for i in vs_idx:
        print(f"       {names[i]:45s} MW={mw[i]:.1f}  LogKd={y_all[i]:.3f}")

    # ── od tego momentu WSZYSTKIE filtry TYLKO na TS ──────────────────────
    Xdf = pd.DataFrame(X_raw.values, columns=X_raw.columns)
    y_s = pd.Series(y_all)

    # A4: filtr |r_y|
    corr_y = Xdf.iloc[ts_idx].corrwith(y_s.iloc[ts_idx]).abs()
    slabe  = corr_y[corr_y < CORR_Y_MIN].index
    Xdf    = Xdf.drop(columns=slabe)
    print(f"\n[A4] Filtr |r_y|>={CORR_Y_MIN}: usunięto {len(slabe)}, "
          f"pozostało {Xdf.shape[1]}")

    # A5: filtr współliniowości — z pary zachowaj wyższy |r_y|
    corr_y2 = Xdf.iloc[ts_idx].corrwith(y_s.iloc[ts_idx]).abs()
    cm      = Xdf.iloc[ts_idx].corr().abs()
    order   = corr_y2.sort_values(ascending=False).index.tolist()
    drop    = set()
    for ii, a in enumerate(order):
        if a in drop: continue
        for b in order[ii+1:]:
            if b in drop: continue
            if cm.loc[a, b] > CORR_XX_MAX:
                drop.add(b)
    Xdf = Xdf.drop(columns=list(drop))
    print(f"[A5] Filtr |r_XX|<={CORR_XX_MAX}: usunięto {len(drop)}, "
          f"pozostało {Xdf.shape[1]}")

    kand = list(Xdf.columns)

    # A6: autoskalowanie — parametry z TS → zastosowane do VS
    scaler = StandardScaler().fit(Xdf.iloc[ts_idx])
    Xs     = pd.DataFrame(scaler.transform(Xdf), columns=kand)

    # ranking wg |r_y| na TS
    corr_rank = (Xs.iloc[ts_idx]
                   .corrwith(y_s.iloc[ts_idx])
                   .abs()
                   .sort_values(ascending=False))

    print(f"\n     TOP 10 kandydatów (|r z LogKd| na TS):")
    for nm, rv in corr_rank.head(10).items():
        print(f"       {nm:30s}  |r|={rv:.3f}")
    print()

    return dict(Xs=Xs, y=y_all, kand=kand, ts_idx=ts_idx, vs_idx=vs_idx,
                corr_rank=corr_rank, names=names, mw=mw)


# ══════════════════════════════════════════════════════════════════════════
#  VIP (wersja wektorowa — bez pętli, bez błędu kształtu)
# ══════════════════════════════════════════════════════════════════════════

def oblicz_vip(pls):
    """
    VIP_j = sqrt( p * Σ_h[ s_h * (w*_jh/||w*_h||)² ] / Σ_h[s_h] )
    Średni VIP² = 1, więc VIP > 1 → ponadprzeciętna ważność (Wold 1994).
    """
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h    = w.shape
    s       = np.diag(t.T @ t @ q.T @ q).ravel()   # (h,) — wariancja y/LV
    w_norm  = w / np.linalg.norm(w, axis=0)          # normalizacja kolumnowa
    return np.sqrt(p * ((w_norm ** 2) @ s) / s.sum()) # (p,)


# ══════════════════════════════════════════════════════════════════════════
#  ALL-SUBSETS (MLR i pierwszy krok kNN)
# ══════════════════════════════════════════════════════════════════════════

def all_subsets(Xts, yts, estimator, pula, min_d=MIN_DESC, max_d=MAX_DESC):
    """
    Przeszukuje WSZYSTKIE kombinacje 2–4 deskryptorów z listy `pula`
    i zwraca zestaw o najwyższym Q²_LOO.
    Na top-10 i rozmiarach 2–4 to 375 kombinacji × 25 LOO = ok. 9 000
    wywołań modelu — kilka sekund.
    """
    best = {"cols": None, "q2": -np.inf, "rmsecv": np.inf}
    for n in range(min_d, max_d + 1):
        for cols in itertools.combinations(pula, n):
            cols = list(cols)
            q2, rv = q2_loo(estimator, Xts[cols].values, yts)
            if q2 > best["q2"]:
                best = {"cols": cols, "q2": q2, "rmsecv": rv}
    return best["cols"], best["q2"], best["rmsecv"]


# ══════════════════════════════════════════════════════════════════════════
#  OCENA NA VS
# ══════════════════════════════════════════════════════════════════════════

def ocena(model, Xts, yts, Xvs, yvs, cols, nazwa):
    model.fit(Xts[cols].values, yts)
    q2, rmsecv = q2_loo(model, Xts[cols].values, yts)
    r2_ts  = r2_score(yts, model.predict(Xts[cols].values).ravel())
    r2_vs  = r2_score(yvs, model.predict(Xvs[cols].values).ravel())
    rmse_vs = rmse(yvs, model.predict(Xvs[cols].values).ravel())
    print(f"  [{nazwa:22s}]  "
          f"R²_TS={r2_ts:.3f}  Q²_LOO={q2:.3f}  RMSECV={rmsecv:.3f}  ||  "
          f"R²_VS={r2_vs:.3f}  RMSEP={rmse_vs:.3f}  "
          f"n={len(cols)}")
    return dict(Model=nazwa, R2_TS=r2_ts, Q2_LOO=q2, RMSECV=rmsecv,
                R2_VS=r2_vs, RMSEP=rmse_vs, n_desc=len(cols),
                deskryptory=cols)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    d          = wczytaj_i_przygotuj()
    Xs, y      = d["Xs"], d["y"]
    ts_idx     = d["ts_idx"]
    vs_idx     = d["vs_idx"]
    corr_rank  = d["corr_rank"]

    Xts, yts = Xs.iloc[ts_idx], y[ts_idx]
    Xvs, yvs = Xs.iloc[vs_idx], y[vs_idx]
    top10    = corr_rank.head(TOP_ALLSUB).index.tolist()
    top15    = corr_rank.head(TOP_FOR_PLS).index.tolist()

    print("="*68)
    print(" FAZA B: SELEKCJA DESKRYPTORÓW")
    print("="*68)

    # ── MLR ───────────────────────────────────────────────────────────────
    # All-subsets na TOP-10, kryterium max Q²_LOO.
    # Cytowanie: Tropsha (2010) J. Chem. Inf. Model. 50, 1189-1204
    print(f"\n--- MLR  all-subsets {MIN_DESC}–{MAX_DESC} desc z TOP-{TOP_ALLSUB} ---")
    desc_mlr, q2_mlr, rv_mlr = all_subsets(
        Xts, yts, LinearRegression(), top10)
    print(f"  >> MLR: {len(desc_mlr)} desc | Q²_LOO={q2_mlr:.3f}  "
          f"RMSECV={rv_mlr:.3f}\n     {desc_mlr}")

    # ── kNN ───────────────────────────────────────────────────────────────
    # Krok 1: all-subsets z k=3 (startowe) → najlepszy zestaw deskryptorów.
    # Krok 2: na wybranych deskryptorach szukamy optymalnego k=2..7.
    # Rozdzielenie tych dwóch kroków redukuje obliczenia ~6×.
    # Cytowanie: Tropsha (2010) j.w.
    print(f"\n--- kNN  all-subsets {MIN_DESC}–{MAX_DESC} desc z TOP-{TOP_ALLSUB}, "
          f"potem dobór k ---")
    desc_knn, _, _ = all_subsets(
        Xts, yts, KNeighborsRegressor(n_neighbors=3), top10)
    best_k, best_q2_knn, best_rv_knn = 3, -np.inf, np.inf
    for k in range(2, 8):
        q2, rv = q2_loo(KNeighborsRegressor(n_neighbors=k),
                        Xts[desc_knn].values, yts)
        if q2 > best_q2_knn:
            best_k, best_q2_knn, best_rv_knn = k, q2, rv
    print(f"  >> kNN: k={best_k}, {len(desc_knn)} desc | "
          f"Q²_LOO={best_q2_knn:.3f}  RMSECV={best_rv_knn:.3f}\n     {desc_knn}")

    # ── PLS ───────────────────────────────────────────────────────────────
    # Pula: TOP-15 wg |r_y|. PLS lubi korelujące predyktory — wyciąga z nich
    # wspólny sygnał przez LV. Liczba LV dobierana przez min RMSECV (LOO).
    # VIP>1 identyfikuje najważniejsze deskryptory post-hoc.
    # Cytowanie: Wold et al. (2001) Chemom. Intell. Lab. Syst. 58, 109-130
    print(f"\n--- PLS  TOP-{TOP_FOR_PLS} desc wg |r_y|, LV wg min RMSECV ---")
    print(f"  Pula: {top15}")
    best_lv = {"n": 1, "q2": -np.inf, "rv": np.inf}
    for n in range(1, min(9, len(top15), Xts.shape[0])):
        q2, rv = q2_loo(PLSRegression(n_components=n),
                        Xts[top15].values, yts)
        print(f"    LV={n}: Q²_LOO={q2:.3f}  RMSECV={rv:.3f}")
        if q2 > best_lv["q2"]:
            best_lv = {"n": n, "q2": q2, "rv": rv}
    n_lv = best_lv["n"]
    pls_fit = PLSRegression(n_components=n_lv).fit(Xts[top15].values, yts)
    vip = oblicz_vip(pls_fit)
    vip_ser = pd.Series(vip, index=top15).sort_values(ascending=False)
    desc_pls_vip = vip_ser[vip_ser > 1.0].index.tolist()
    print(f"  >> PLS: LV={n_lv} | Q²_LOO={best_lv['q2']:.3f}  "
          f"RMSECV={best_lv['rv']:.3f}")
    print(f"     VIP ranking: {dict(vip_ser.round(3))}")
    print(f"     VIP>1 ({len(desc_pls_vip)}): {desc_pls_vip}")

    # ── SVR ───────────────────────────────────────────────────────────────
    # RFE na liniowym SVR daje ranking zmiennych (Guyon & Elisseeff 2003).
    # Testujemy rozmiary 2–4, dla każdego grid-search LOO na RBF.
    # Wybieramy rozmiar z najwyższym Q²_LOO na wygrywającym zestawie.
    print(f"\n--- SVR  RFE ranking + grid-search C/γ/ε (RBF) ---")
    top20 = corr_rank.head(20).index.tolist()
    rfe = RFE(SVR(kernel="linear"), n_features_to_select=MIN_DESC)
    rfe.fit(Xts[top20].values, yts)
    rank_svr = pd.Series(rfe.ranking_, index=top20).sort_values()
    print(f"  RFE ranking (top-8): {rank_svr.head(8).index.tolist()}")

    siatka = {"C": [0.1, 1, 10, 100],
              "gamma": ["scale", 0.01, 0.1, 1],
              "epsilon": [0.05, 0.1, 0.2]}
    best_svr = {"cols": None, "params": {}, "q2": -np.inf, "rv": np.inf}
    for n_d in range(MIN_DESC, MAX_DESC + 1):
        cols = rank_svr.head(n_d).index.tolist()
        gs   = GridSearchCV(SVR(kernel="rbf"), siatka,
                            cv=LeaveOneOut(),
                            scoring="neg_mean_squared_error",
                            n_jobs=-1)
        gs.fit(Xts[cols].values, yts)
        q2, rv = q2_loo(gs.best_estimator_, Xts[cols].values, yts)
        print(f"    n_desc={n_d}: Q²_LOO={q2:.3f}  RMSECV={rv:.3f}  "
              f"params={gs.best_params_}  desc={cols}")
        if q2 > best_svr["q2"]:
            best_svr = {"cols": cols, "params": gs.best_params_,
                        "q2": q2, "rv": rv}
    desc_svr = best_svr["cols"]
    print(f"  >> SVR: {len(desc_svr)} desc | Q²_LOO={best_svr['q2']:.3f}  "
          f"RMSECV={best_svr['rv']:.3f}\n     params={best_svr['params']}\n"
          f"     {desc_svr}")

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*68)
    print(" OCENA KOŃCOWA NA VS")
    print("="*68)
    wyniki = []
    wyniki.append(ocena(LinearRegression(),
                        Xts, yts, Xvs, yvs, desc_mlr, "MLR"))
    wyniki.append(ocena(KNeighborsRegressor(n_neighbors=best_k),
                        Xts, yts, Xvs, yvs, desc_knn, f"kNN(k={best_k})"))
    wyniki.append(ocena(PLSRegression(n_components=n_lv),
                        Xts, yts, Xvs, yvs, top15, f"PLS(LV={n_lv},n=15)"))
    wyniki.append(ocena(SVR(kernel="rbf", **best_svr["params"]),
                        Xts, yts, Xvs, yvs, desc_svr, "SVR"))

    print("\n" + "="*68)
    print(" PODSUMOWANIE")
    print("="*68)
    tab = pd.DataFrame(wyniki).set_index("Model")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(tab.drop(columns=["deskryptory"]).to_string())

    # zapis
    with open("wyniki_qspr_v2.txt", "w", encoding="utf-8") as f:
        f.write("PODZIAŁ TS/VS (Kennard-Stone)\n")
        f.write(f"  TS ({len(ts_idx)} zw.): idx {ts_idx}\n")
        f.write(f"  VS ({len(vs_idx)} zw.): idx {vs_idx}\n\n")
        for w in wyniki:
            f.write(f"{w['Model']}:\n  Deskryptory: {w['deskryptory']}\n"
                    f"  R²_TS={w['R2_TS']:.3f}  Q²_LOO={w['Q2_LOO']:.3f}  "
                    f"RMSECV={w['RMSECV']:.3f}  "
                    f"R²_VS={w['R2_VS']:.3f}  RMSEP={w['RMSEP']:.3f}\n\n")
        f.write(f"PLS VIP:\n{vip_ser.round(3).to_string()}\n")
    print("\nWyniki → wyniki_qspr_v2.txt")


if __name__ == "__main__":
    main()
