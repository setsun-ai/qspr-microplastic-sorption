"""
============================================================================
 SELEKCJA DESKRYPTOROW QSPR DLA 4 MODELI: MLR, kNN, PLS, SVR
============================================================================
 Dane: data_qsar_pls.xlsx, arkusz "PROJEKT"
   - kolumna A: nazwa zwiazku   (Organic compounds)
   - kolumna B: odpowiedz        (EXP LogKd)   <- y
   - kolumna C: podzial          (Set: TS / VS)
   - kolumny D..      : deskryptory (zmienne X)
 35 zwiazkow: TS = 24 (uczacy), VS = 11 (walidacyjny)

 ZASADA NADRZEDNA:
   Cala selekcja i wszystkie parametry (skalowanie, progi, dobor zmiennych)
   liczone SA WYLACZNIE NA TS. Zbior VS sluzy tylko do koncowej oceny.
   To zapobiega wyciekowi informacji (data leakage) i zawyzaniu walidacji.

 Wymagane biblioteki:
   pip install pandas numpy scikit-learn openpyxl
   (deap jest opcjonalne - dla algorytmu genetycznego; jesli brak,
    skrypt automatycznie uzyje selekcji krokowej / RFE)
============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------------------------------------------------------
# USTAWIENIA - mozesz je tu dostroic
# ----------------------------------------------------------------------------
PLIK          = "data_qsar_pls.xlsx"   # sciezka do pliku
ARKUSZ        = "PROJEKT"
KOL_NAZWA     = "Organic compounds"
KOL_Y         = "EXP LogKda"           # uwaga: w pliku naglowek to "EXP LogKda"
KOL_SET       = "Set"
TS_LABEL      = "TS"
VS_LABEL      = "VS"

VAR_THRESH    = 1e-8    # prog wariancji - usun deskryptory ~stale
CORR_Y_MIN    = 0.15    # min |korelacja z y|, by deskryptor przeszedl filtr
CORR_XX_MAX   = 0.70    # max |korelacja miedzy X| - usun wspolliniowe pary
MAX_VARS_MLR  = 5       # max liczba deskryptorow w MLR / kNN (regula ~5 obiektow/zmienna)
RANDOM_STATE  = 42


# ----------------------------------------------------------------------------
# METRYKI POMOCNICZE
# ----------------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def q2_loo(model, X, y):
    """Q^2 z walidacji krzyzowej Leave-One-Out na zbiorze uczacym."""
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)
    return r2_score(y, y_pred), rmse(y, y_pred)


# ----------------------------------------------------------------------------
# FAZA A: WCZYTANIE + WSPOLNY PRE-PROCESSING (liczony na TS)
# ----------------------------------------------------------------------------
def wczytaj_i_przygotuj():
    df = pd.read_excel(PLIK, sheet_name=ARKUSZ)

    # rozdziel kolumny opisowe od deskryptorow
    meta_cols = [KOL_NAZWA, KOL_Y, KOL_SET]
    desc_cols = [c for c in df.columns if c not in meta_cols]

    # maska TS / VS
    is_ts = df[KOL_SET] == TS_LABEL
    is_vs = df[KOL_SET] == VS_LABEL

    X_all = df[desc_cols].apply(pd.to_numeric, errors="coerce")
    y_all = pd.to_numeric(df[KOL_Y], errors="coerce")

    # --- (1) usun kolumny z brakami danych ---
    braki = X_all.columns[X_all.isna().any()]
    X_all = X_all.drop(columns=braki)
    print(f"[A1] Usunieto {len(braki)} deskryptorow z brakami danych.")

    # --- (2) usun deskryptory o ~zerowej wariancji (liczone na TS) ---
    war_ts = X_all[is_ts].var()
    stale = war_ts[war_ts <= VAR_THRESH].index
    X_all = X_all.drop(columns=stale)
    print(f"[A2] Usunieto {len(stale)} deskryptorow ~stalych (zerowa wariancja).")

    # --- (3) filtr korelacji z odpowiedzia (na TS) ---
    corr_y = X_all[is_ts].corrwith(y_all[is_ts]).abs()
    slabe = corr_y[corr_y < CORR_Y_MIN].index
    X_all = X_all.drop(columns=slabe)
    print(f"[A3] Usunieto {len(slabe)} deskryptorow slabo skorelowanych z y "
          f"(|r| < {CORR_Y_MIN}).")

    # --- (4) filtr wspolliniowosci (na TS): z pary |r|>prog zostaw lepszy ---
    corr_y = X_all[is_ts].corrwith(y_all[is_ts]).abs()
    cm = X_all[is_ts].corr().abs()
    kolejnosc = corr_y.sort_values(ascending=False).index.tolist()
    do_usuniecia = set()
    for i, a in enumerate(kolejnosc):
        if a in do_usuniecia:
            continue
        for b in kolejnosc[i+1:]:
            if b in do_usuniecia:
                continue
            if cm.loc[a, b] > CORR_XX_MAX:
                do_usuniecia.add(b)   # zostaje 'a' (lepiej skorelowany z y)
    X_all = X_all.drop(columns=list(do_usuniecia))
    print(f"[A4] Usunieto {len(do_usuniecia)} wspolliniowych deskryptorow "
          f"(|r_XX| > {CORR_XX_MAX}).")

    kandydaci = list(X_all.columns)
    print(f"[A ] PULA KANDYDATOW po filtrach: {len(kandydaci)} deskryptorow.\n")

    # --- (5) standaryzacja (autoskalowanie) - parametry liczone na TS ---
    scaler = StandardScaler().fit(X_all[is_ts])
    Xs = pd.DataFrame(scaler.transform(X_all), columns=kandydaci, index=X_all.index)

    return {
        "Xs_ts": Xs[is_ts], "Xs_vs": Xs[is_vs],
        "y_ts": y_all[is_ts].values, "y_vs": y_all[is_vs].values,
        "kandydaci": kandydaci,
    }


# ----------------------------------------------------------------------------
# FAZA B: SELEKCJA WLASCIWA - osobno dla kazdej metody
# ----------------------------------------------------------------------------

def selekcja_mlr_knn(Xts, yts, kandydaci, estimator, nazwa, max_vars=MAX_VARS_MLR):
    """
    Selekcja krokowa w przod (forward stepwise) z kryterium Q^2 (LOO).
    Wspolna dla MLR i kNN - oba potrzebuja malego, dobranego zestawu.
    Dodajemy deskryptory jeden po drugim, dopoki rosnie Q^2.
    """
    wybrane, najlepszy_q2 = [], -np.inf
    pozostale = list(kandydaci)

    for _ in range(max_vars):
        kandydat_q2 = []
        for c in pozostale:
            cols = wybrane + [c]
            q2, _ = q2_loo(estimator, Xts[cols].values, yts)
            kandydat_q2.append((q2, c))
        kandydat_q2.sort(reverse=True)
        best_q2, best_c = kandydat_q2[0]
        if best_q2 > najlepszy_q2 + 1e-4:   # dodaj tylko jesli realnie poprawia
            najlepszy_q2 = best_q2
            wybrane.append(best_c)
            pozostale.remove(best_c)
        else:
            break

    q2, rmsecv = q2_loo(estimator, Xts[wybrane].values, yts)
    print(f"  >> {nazwa}: wybrano {len(wybrane)} deskryptorow | "
          f"Q2_LOO={q2:.3f}, RMSECV={rmsecv:.3f}")
    print(f"     {wybrane}")
    return wybrane


def selekcja_pls(Xts, yts, kandydaci, max_komponenty=8):
    """
    PLS toleruje wspolliniowosc i p>>n -> podajemy CALA pule kandydatow,
    a liczbe komponentow dobieramy minimalizujac RMSECV (LOO).
    Nastepnie wskazujemy najwazniejsze deskryptory wg VIP > 1.
    """
    najlepszy = (None, -np.inf, None)  # (n_comp, q2, rmsecv)
    maks = min(max_komponenty, len(kandydaci), Xts.shape[0]-1)
    for n in range(1, maks+1):
        pls = PLSRegression(n_components=n)
        q2, rmsecv = q2_loo(pls, Xts[kandydaci].values, yts)
        if q2 > najlepszy[1]:
            najlepszy = (n, q2, rmsecv)
    n_opt, q2, rmsecv = najlepszy

    # policz VIP dla modelu o optymalnej liczbie komponentow
    pls = PLSRegression(n_components=n_opt).fit(Xts[kandydaci].values, yts)
    vip = oblicz_vip(pls)
    vip_ser = pd.Series(vip, index=kandydaci).sort_values(ascending=False)
    istotne = vip_ser[vip_ser > 1.0].index.tolist()

    print(f"  >> PLS: optymalna liczba komponentow = {n_opt} | "
          f"Q2_LOO={q2:.3f}, RMSECV={rmsecv:.3f}")
    print(f"     Deskryptory istotne (VIP>1): {len(istotne)} szt.")
    print(f"     TOP 10 wg VIP: {vip_ser.head(10).index.tolist()}")
    return {"n_components": n_opt, "vip_istotne": istotne,
            "vip_ranking": vip_ser}


def oblicz_vip(pls):
    """
    Variable Importance in Projection dla modelu PLS (wersja wektorowa).

    Dla kazdego deskryptora:
       VIP = sqrt( p * SUMA_h[ (w_norm[h])^2 * s[h] ] / SUMA_h[ s[h] ] )
    gdzie:
       p       = liczba deskryptorow,
       s[h]    = wariancja y wyjasniona przez h-ty komponent,
       w_norm  = wagi X znormalizowane w obrebie kazdego komponentu (kolumny).
    Sredni kwadrat VIP wynosi 1, wiec prog VIP>1 wskazuje deskryptory
    o ponadprzecietnym wkladzie.
    """
    t = pls.x_scores_      # (n, h)
    w = pls.x_weights_     # (p, h)
    q = pls.y_loadings_    # (1, h)
    p, h = w.shape

    # wariancja wyjasniona przez kazdy komponent (wektor dlugosci h)
    s = np.diag(t.T @ t @ q.T @ q).ravel()
    total_s = s.sum()

    # normalizacja kolumn macierzy wag (kazdy komponent osobno)
    w_norm = w / np.linalg.norm(w, axis=0)

    # jedna operacja macierzowa -> jeden VIP na deskryptor (wektor dlugosci p)
    vips = np.sqrt(p * ((w_norm ** 2) @ s) / total_s)
    return vips


def selekcja_svr(Xts, yts, kandydaci, max_vars=MAX_VARS_MLR):
    """
    SVR (jadro RBF) jest nieliniowy. Selekcja wrapper przez RFE wymaga
    estymatora liniowego, wiec do samego RANKINGU zmiennych uzywamy SVR
    liniowego (daje coef_), a finalny model budujemy z jadrem RBF.
    Hiperparametry (C, gamma, epsilon) dostrajamy GridSearchem z LOO.
    """
    # ranking zmiennych: RFE na SVR liniowym
    svr_lin = SVR(kernel="linear")
    rfe = RFE(svr_lin, n_features_to_select=min(max_vars, len(kandydaci)))
    rfe.fit(Xts[kandydaci].values, yts)
    wybrane = [c for c, keep in zip(kandydaci, rfe.support_) if keep]

    # dostrojenie SVR-RBF na wybranych deskryptorach
    siatka = {
        "C":       [0.1, 1, 10, 100],
        "gamma":   ["scale", 0.01, 0.1, 1],
        "epsilon": [0.01, 0.1, 0.2],
    }
    gs = GridSearchCV(SVR(kernel="rbf"), siatka,
                      cv=LeaveOneOut(), scoring="neg_mean_squared_error")
    gs.fit(Xts[wybrane].values, yts)
    best_svr = gs.best_estimator_
    q2, rmsecv = q2_loo(best_svr, Xts[wybrane].values, yts)

    print(f"  >> SVR: wybrano {len(wybrane)} deskryptorow | "
          f"Q2_LOO={q2:.3f}, RMSECV={rmsecv:.3f}")
    print(f"     Najlepsze hiperparametry: {gs.best_params_}")
    print(f"     {wybrane}")
    return {"deskryptory": wybrane, "params": gs.best_params_}


# ----------------------------------------------------------------------------
# OCENA KONCOWA: walidacja na VS (true-ish external validation)
# ----------------------------------------------------------------------------
def ocena_na_vs(model, Xts, yts, Xvs, yvs, cols, nazwa):
    model.fit(Xts[cols].values, yts)
    yhat_ts = model.predict(Xts[cols].values).ravel()
    yhat_vs = model.predict(Xvs[cols].values).ravel()
    r2_ts  = r2_score(yts, yhat_ts)
    q2, rmsecv = q2_loo(model, Xts[cols].values, yts)
    r2_vs  = r2_score(yvs, yhat_vs)
    rmse_vs = rmse(yvs, yhat_vs)
    print(f"\n  [{nazwa}]  R2_TS={r2_ts:.3f} | Q2_LOO={q2:.3f} | "
          f"RMSECV={rmsecv:.3f} || R2_VS={r2_vs:.3f} | RMSEP={rmse_vs:.3f}")
    return {"model": nazwa, "R2_TS": r2_ts, "Q2_LOO": q2,
            "RMSECV": rmsecv, "R2_VS": r2_vs, "RMSEP": rmse_vs, "n_desc": len(cols)}


# ----------------------------------------------------------------------------
# GLOWNY PRZEBIEG
# ----------------------------------------------------------------------------
def main():
    print("="*70)
    print(" FAZA A: PRE-PROCESSING I REDUKCJA PULI DESKRYPTOROW (na TS)")
    print("="*70)
    d = wczytaj_i_przygotuj()
    Xts, Xvs = d["Xs_ts"], d["Xs_vs"]
    yts, yvs = d["y_ts"], d["y_vs"]
    kand = d["kandydaci"]

    print("="*70)
    print(" FAZA B: SELEKCJA DESKRYPTOROW - OSOBNO DLA KAZDEJ METODY")
    print("="*70)

    print("\n--- MLR (regresja krokowa, kryterium Q2_LOO) ---")
    desc_mlr = selekcja_mlr_knn(Xts, yts, kand,
                                LinearRegression(), "MLR")

    print("\n--- kNN (regresja krokowa + dobor k) ---")
    # prosty dobor k: testujemy k=3..7 na pelnej forward-selekcji z k=3,
    # nastepnie dostrajamy k na wybranym zestawie
    desc_knn = selekcja_mlr_knn(Xts, yts, kand,
                                KNeighborsRegressor(n_neighbors=3), "kNN")
    best_k, best_q2 = 3, -np.inf
    for k in range(2, 8):
        q2, _ = q2_loo(KNeighborsRegressor(n_neighbors=k),
                       Xts[desc_knn].values, yts)
        if q2 > best_q2:
            best_q2, best_k = q2, k
    print(f"     -> optymalne k = {best_k}")

    print("\n--- PLS (cala pula kandydatow + VIP) ---")
    pls_info = selekcja_pls(Xts, yts, kand)

    print("\n--- SVR (RFE + dostrojenie jadra RBF) ---")
    svr_info = selekcja_svr(Xts, yts, kand)

    print("\n" + "="*70)
    print(" OCENA KONCOWA NA ZBIORZE WALIDACYJNYM (VS)")
    print("="*70)
    wyniki = []
    wyniki.append(ocena_na_vs(LinearRegression(), Xts, yts, Xvs, yvs,
                              desc_mlr, "MLR"))
    wyniki.append(ocena_na_vs(KNeighborsRegressor(n_neighbors=best_k),
                              Xts, yts, Xvs, yvs, desc_knn, f"kNN (k={best_k})"))
    wyniki.append(ocena_na_vs(PLSRegression(n_components=pls_info["n_components"]),
                              Xts, yts, Xvs, yvs, kand, "PLS (cala pula)"))
    wyniki.append(ocena_na_vs(SVR(kernel="rbf", **svr_info["params"]),
                              Xts, yts, Xvs, yvs, svr_info["deskryptory"], "SVR"))

    print("\n" + "="*70)
    print(" PODSUMOWANIE")
    print("="*70)
    tab = pd.DataFrame(wyniki).set_index("model")
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(tab.to_string())

    # zapis wybranych deskryptorow do pliku
    with open("wybrane_deskryptory.txt", "w", encoding="utf-8") as f:
        f.write("MLR:\n  " + ", ".join(desc_mlr) + "\n\n")
        f.write(f"kNN (k={best_k}):\n  " + ", ".join(desc_knn) + "\n\n")
        f.write(f"PLS (n_comp={pls_info['n_components']}, cala pula "
                f"{len(kand)} deskr.):\n  TOP wg VIP: " +
                ", ".join(pls_info["vip_ranking"].head(15).index) + "\n\n")
        f.write("SVR:\n  " + ", ".join(svr_info["deskryptory"]) + "\n")
    print("\nWybrane deskryptory zapisano do: wybrane_deskryptory.txt")


if __name__ == "__main__":
    main()
