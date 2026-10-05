"""
===========================================================================
 QSPR v3 — DOBÓR DESKRYPTORÓW Z OCHRONĄ DOMENY APLIKABILNOŚCI
 + predykcja LogKd dla nitrobenzenu i naftalenu (true external validation)
===========================================================================
 Cel: wybrać chemicznie sensowne deskryptory tak, by predykcja dla
      docelowych związków (nitrobenzen, naftalen) NIE była ekstrapolacją.

 Różnica względem v2: DODATKOWY filtr — przed selekcją usuwamy deskryptory,
 dla których którykolwiek z docelowych związków wypada poza zakres TS.
 Dopiero z tej "domenowo-bezpiecznej" puli robimy all-subsets.

 Deskryptory rozważane (mechanistycznie uzasadnione dla adsorpcji na PP):
   - hydrofobowość:  CrippenLogP, XLogP, MLogP, ALogP   (siła napędowa)
   - polarność:      TopoPSA, MLFER_E                    (osłabia adsorpcję)
   - rozmiar/kształt: AMR, nAromBond, MW                  (van der Waals)
 (świadomie POMIJAMY MLFER_L — w tym zbiorze koreluje z rozmiarem i
  powoduje ekstrapolację dla małych związków, co wykazała analiza v2)

 Cytowania metodologiczne:
   - Domena aplikabilności (range-based): Netzeva et al. (2005),
     Jaworska et al. (2005); OECD (2007) Guidance Document, zasada 3
   - All-subsets + Q²_LOO: Tropsha (2010)
===========================================================================
 Wymagania:  pip install pandas numpy scikit-learn openpyxl
 Pliki:  data_qsar_pls.xlsx + 4 pliki CSV z PaDEL dla nowych związków
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
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# ─── USTAWIENIA ────────────────────────────────────────────────────────────
PLIK     = "data_qsar_pls.xlsx"
ARKUSZ   = "PROJEKT"
KOL_Y    = "EXP LogKda"
KOL_NAME = "Organic compounds"

# Podział Kennard-Stone z poprzedniej analizy (zafiksowany dla powtarzalności)
TS_IDX = [0,1,4,6,7,8,9,10,11,12,13,14,15,18,19,20,22,23,24,26,28,30,32,33,34]
VS_IDX = [2,3,5,16,17,21,25,27,29,31]

MIN_DESC = 2
MAX_DESC = 4
CORR_XX_MAX = 0.85   # filtr współliniowości
DOMAIN_MARGIN = 0.0  # 0 = ścisły zakres TS; >0 = dopuszcza lekkie wyjście

# Deskryptory-kandydaci (mechanistycznie sensowne dla adsorpcji na PP)
KANDYDACI_CHEM = ['CrippenLogP', 'XLogP', 'MLogP', 'ALogP',
                  'TopoPSA', 'MLFER_E', 'AMR', 'nAromBond', 'MW']

# Wartości deskryptorów docelowych związków (z PaDEL — patrz pliki CSV)
# UWAGA: jeśli zmienisz pulę kandydatów, uzupełnij tu odpowiednie wartości
DOCELOWE = {
    'Naphthalene':  {'CrippenLogP': 2.840, 'XLogP': 3.224, 'MLogP': 2.560,
                     'ALogP': 1.948, 'TopoPSA': 0.000, 'MLFER_E': 1.392,
                     'AMR': 49.153, 'nAromBond': 11.0, 'MW': 128.063},
    'Nitrobenzene': {'CrippenLogP': 1.454, 'XLogP': 1.244, 'MLogP': 1.790,
                     'ALogP': 1.617, 'TopoPSA': 43.140, 'MLFER_E': 0.937,
                     'AMR': 34.173, 'nAromBond': 6.0, 'MW': 123.032},
}

# Eksperymentalne wartości LogKd z literatury — DO UZUPEŁNIENIA przez studenta
# (skrypt NIE ma dostępu do internetu; wpisz tu wartości znalezione w pracach,
#  np. hasła: "naphthalene polypropylene microplastic sorption logKd")
# Jeśli zostawisz None, skrypt poda tylko oszacowanie z korelacji logKd–logP.
LIT_EXP = {
    # Wang et al. (2019) "Sorption behaviors of phenanthrene, nitrobenzene,
    # and naphthalene on mesoplastics and microplastics", Log Kd (Henry), PP
    'Naphthalene':  4.09,   # PP microplastics (mesoplastics: 2.80)
    'Nitrobenzene': 3.23,   # PP microplastics (mesoplastics: 2.25)
}
# Benchmark uzupełniający (PP mesoplastics) — do porównania morfologii
LIT_EXP_MESO = {'Naphthalene': 2.80, 'Nitrobenzene': 2.25}
# ───────────────────────────────────────────────────────────────────────────


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

def q2_loo(est, X, y):
    pred = cross_val_predict(est, X, y, cv=LeaveOneOut())
    return r2_score(y, pred), rmse(y, pred)


# ══════════════════════════════════════════════════════════════════════════
#  WCZYTANIE I PRZYGOTOWANIE
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print(" QSPR v3 — SELEKCJA DESKRYPTORÓW Z OCHRONĄ DOMENY")
    print("="*70)

    df = pd.read_excel(PLIK, sheet_name=ARKUSZ)
    y  = pd.to_numeric(df[KOL_Y], errors="coerce").values
    names = df[KOL_NAME].values

    # zbierz dostępne deskryptory-kandydaci
    avail = {}
    for c in KANDYDACI_CHEM:
        cols = [col for col in df.columns if str(col).strip() == c]
        if cols:
            avail[c] = pd.to_numeric(df[cols[0]], errors="coerce")
    Xall = pd.DataFrame(avail)
    print(f"\n[1] Dostępne deskryptory-kandydaci: {list(Xall.columns)}")

    yts, yvs = y[TS_IDX], y[VS_IDX]
    Xts_raw = Xall.iloc[TS_IDX]

    # ── FILTR DOMENY: usuń deskryptory ekstrapolujące dla docelowych ──────
    print(f"\n[2] FILTR DOMENY APLIKABILNOŚCI (zakres TS, margines={DOMAIN_MARGIN})")
    print(f"    {'Deskryptor':12s} {'zakres TS':>20s}  Naftalen  Nitrobenzen  status")
    safe = []
    for c in Xall.columns:
        lo, hi = Xts_raw[c].min(), Xts_raw[c].max()
        span = hi - lo
        lo_m, hi_m = lo - DOMAIN_MARGIN*span, hi + DOMAIN_MARGIN*span
        vn = DOCELOWE['Naphthalene'][c]
        vt = DOCELOWE['Nitrobenzene'][c]
        ok = (lo_m <= vn <= hi_m) and (lo_m <= vt <= hi_m)
        if ok: safe.append(c)
        s = "BEZPIECZNY" if ok else "ekstrapolacja"
        print(f"    {c:12s} [{lo:7.2f},{hi:7.2f}]  {vn:7.2f}  {vt:9.2f}   {s}")

    print(f"\n    Pula domenowo-bezpieczna: {safe}")
    if len(safe) < MIN_DESC:
        print("    UWAGA: za mało bezpiecznych deskryptorów — zwiększ DOMAIN_MARGIN")
        return

    # ── filtr współliniowości na bezpiecznej puli ────────────────────────
    cm = Xts_raw[safe].corr().abs()
    corr_y = Xts_raw[safe].corrwith(pd.Series(yts, index=Xts_raw.index)).abs()
    order = corr_y.sort_values(ascending=False).index.tolist()
    drop = set()
    for i, a in enumerate(order):
        if a in drop: continue
        for b in order[i+1:]:
            if b in drop: continue
            if cm.loc[a, b] > CORR_XX_MAX:
                drop.add(b)
    safe = [c for c in safe if c not in drop]
    print(f"[3] Po filtrze współliniowości |r_XX|<={CORR_XX_MAX}: {safe}")

    # ── standaryzacja (parametry z TS) ───────────────────────────────────
    sc = StandardScaler().fit(Xts_raw[safe])
    Xts = pd.DataFrame(sc.transform(Xts_raw[safe]), columns=safe)
    Xvs = pd.DataFrame(sc.transform(Xall.iloc[VS_IDX][safe]), columns=safe)
    # docelowe związki
    Xnew_raw = pd.DataFrame([{c: DOCELOWE[n][c] for c in safe}
                             for n in DOCELOWE], index=list(DOCELOWE))
    Xnew = pd.DataFrame(sc.transform(Xnew_raw), columns=safe, index=list(DOCELOWE))

    # ── selekcja all-subsets dla MLR ─────────────────────────────────────
    print(f"\n[4] SELEKCJA all-subsets {MIN_DESC}–{MAX_DESC} desc (kryterium Q²_LOO)")
    def allsub(est):
        best = {"cols": None, "q2": -np.inf, "rv": np.inf}
        for n in range(MIN_DESC, min(MAX_DESC, len(safe)) + 1):
            for cc in itertools.combinations(safe, n):
                cc = list(cc)
                q2, rv = q2_loo(est, Xts[cc].values, yts)
                if q2 > best["q2"]:
                    best = {"cols": cc, "q2": q2, "rv": rv}
        return best

    mlr_b = allsub(LinearRegression())
    desc_mlr = mlr_b["cols"]
    print(f"    MLR: {desc_mlr}  Q²_LOO={mlr_b['q2']:.3f}  RMSECV={mlr_b['rv']:.3f}")

    # kNN: all-subsets z k=3, potem dobór k
    knn_b = allsub(KNeighborsRegressor(n_neighbors=3))
    desc_knn = knn_b["cols"]
    best_k, bq = 3, -np.inf
    for k in range(2, 8):
        q2, _ = q2_loo(KNeighborsRegressor(n_neighbors=k), Xts[desc_knn].values, yts)
        if q2 > bq: bq, best_k = q2, k
    print(f"    kNN(k={best_k}): {desc_knn}  Q²_LOO={bq:.3f}")

    # PLS: cała bezpieczna pula, dobór LV
    best_lv, blq = 1, -np.inf
    for n in range(1, min(len(safe), len(TS_IDX)-1) + 1):
        q2, _ = q2_loo(PLSRegression(n_components=n), Xts[safe].values, yts)
        if q2 > blq: blq, best_lv = q2, n
    print(f"    PLS(LV={best_lv}): cała pula {safe}  Q²_LOO={blq:.3f}")

    # SVR: all-subsets z RBF (mała pula → all-subsets jest wykonalne)
    siatka = {"C": [0.1,1,10,100], "gamma": ["scale",0.1,1], "epsilon": [0.05,0.1,0.2]}
    svr_b = {"cols": None, "params": {}, "q2": -np.inf}
    for n in range(MIN_DESC, min(MAX_DESC, len(safe)) + 1):
        for cc in itertools.combinations(safe, n):
            cc = list(cc)
            gs = GridSearchCV(SVR(kernel="rbf"), siatka, cv=LeaveOneOut(),
                              scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(Xts[cc].values, yts)
            q2, _ = q2_loo(gs.best_estimator_, Xts[cc].values, yts)
            if q2 > svr_b["q2"]:
                svr_b = {"cols": cc, "params": gs.best_params_, "q2": q2}
    desc_svr = svr_b["cols"]
    print(f"    SVR: {desc_svr}  Q²_LOO={svr_b['q2']:.3f}  {svr_b['params']}")

    # ── ocena na VS + predykcje na docelowych ────────────────────────────
    print(f"\n[5] OCENA NA VS + PREDYKCJE")
    print("-"*70)

    def fit_eval(model, cols, label):
        model.fit(Xts[cols].values, yts)
        r2t = r2_score(yts, model.predict(Xts[cols].values).ravel())
        r2v = r2_score(yvs, model.predict(Xvs[cols].values).ravel())
        rmsev = rmse(yvs, model.predict(Xvs[cols].values).ravel())
        q2, rmsecv = q2_loo(model, Xts[cols].values, yts)
        pred_new = model.predict(Xnew[cols].values).ravel()
        print(f"  {label:14s} R²_TS={r2t:.3f} Q²_LOO={q2:.3f} R²_VS={r2v:.3f} "
              f"RMSEP={rmsev:.3f} | Naft={pred_new[0]:.2f} Nitro={pred_new[1]:.2f}")
        return dict(model=label, R2_TS=r2t, Q2_LOO=q2, R2_VS=r2v, RMSEP=rmsev,
                    naft=pred_new[0], nitro=pred_new[1], desc=cols)

    res = []
    res.append(fit_eval(LinearRegression(), desc_mlr, "MLR"))
    res.append(fit_eval(KNeighborsRegressor(n_neighbors=best_k), desc_knn, f"kNN(k={best_k})"))
    res.append(fit_eval(PLSRegression(n_components=best_lv), safe, f"PLS(LV={best_lv})"))
    res.append(fit_eval(SVR(kernel="rbf", **svr_b["params"]), desc_svr, "SVR"))

    # ── oszacowanie referencyjne z korelacji logKd–logP (na 35 zw.) ──────
    print(f"\n[6] OSZACOWANIE REFERENCYJNE z korelacji LogKd–logP (Twoje 35 zw.)")
    # użyj najlepszego dostępnego logP (CrippenLogP — najwyższe |r|)
    logp_col = [c for c in df.columns if str(c).strip()=="CrippenLogP"][0]
    logp = pd.to_numeric(df[logp_col], errors="coerce").values
    reg_ref = LinearRegression().fit(logp.reshape(-1,1), y)
    print(f"    Model:  LogKd = {reg_ref.coef_[0]:.3f}·CrippenLogP + {reg_ref.intercept_:.3f}")
    print(f"    R² tej korelacji: {r2_score(y, reg_ref.predict(logp.reshape(-1,1))):.3f}")
    for n in DOCELOWE:
        lp = DOCELOWE[n]['CrippenLogP']
        est = reg_ref.predict([[lp]])[0]
        litval = LIT_EXP[n]
        lit_s = f"{litval:.2f}" if litval is not None else "DO UZUPEŁNIENIA"
        print(f"    {n:14s} CrippenLogP={lp:.3f} -> LogKd_oczek ≈ {est:.2f}  "
              f"(lit.: {lit_s})")

    # ── podsumowanie ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(" PODSUMOWANIE PREDYKCJI")
    print("="*70)
    tab = pd.DataFrame([{k: r[k] for k in ['model','R2_VS','naft','nitro']} for r in res])
    tab.columns = ['Model', 'R²_VS', 'Naftalen', 'Nitrobenzen']
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(tab.to_string(index=False))

    with open("wyniki_qspr_v3.txt", "w", encoding="utf-8") as f:
        f.write("QSPR v3 — selekcja z ochroną domeny\n\n")
        f.write(f"Pula domenowo-bezpieczna: {safe}\n\n")
        for r in res:
            f.write(f"{r['model']}: desc={r['desc']}\n"
                    f"  R²_TS={r['R2_TS']:.3f} Q²_LOO={r['Q2_LOO']:.3f} "
                    f"R²_VS={r['R2_VS']:.3f} RMSEP={r['RMSEP']:.3f}\n"
                    f"  Naftalen={r['naft']:.3f}  Nitrobenzen={r['nitro']:.3f}\n\n")
    print("\nWyniki → wyniki_qspr_v3.txt")


if __name__ == "__main__":
    main()
