# QSPR model of organic pollutant sorption on polypropylene microplastics

Predicting the microplastic–seawater partition coefficient **log K<sub>d,a</sub>** for polypropylene (PP) from molecular descriptors.
MSc student project, Gdańsk University of Technology, 2026. Course: *QSAR/QSPR Modelling*.

This repository keeps **all modelling iterations**, not only the final model. It covers:
- 7 stages
- 2 descriptor families
- 5 algorithms (MLR, PLS, SVR, k-NN, decision tree)
- the reasons each approach was kept or dropped

## Final model

**MLR on two mechanistic descriptors:**

log K<sub>d,a</sub> = 6.735 + 0.751·log D − 19.32·ε<sub>β</sub>

| n | R² | Q²<sub>LOO</sub> | Q²<sub>ext</sub> over 200 random splits | VIF |
|---|---|---|---|---|
| 35 | 0.939 | 0.913 | **0.909 ± 0.035** (min 0.78) | ≈ 1.0 |

- **log D (lipophilicity)** increases sorption on non-polar PP.
- **ε<sub>β</sub> (H-bond basicity)** keeps the compound in the water phase.
- Both signs are mechanistically correct, and log D is about 2.3× the stronger factor.

Reports and presentations (PL) are in [`final/`](final/).

## How the model evolved

![Model development journey](docs/model_development_journey.png)

| # | Stage | Descriptors | Algorithms | Best result | What it taught me |
|---|---|---|---|---|---|
| 1 | [PaDEL screening](stages/1_padel_descriptor_screening) | ~2000 PaDEL → exhaustive search over 2–4 descriptor subsets | MLR, PLS, SVR, k-NN | MLR {Mi, THSA, MLFER_L}: Q²<sub>LOO</sub> 0.929, R²<sub>VS</sub> 0.975 | Very high scores, but abstract descriptors picked from ~2000 candidates for 35 compounds carry a real risk of chance correlation |
| 2 | [Extended PaDEL pipeline](stages/2_padel_extended_pipeline) | variance and correlation filters, all-subsets, PLS-VIP, SVR-RFE, GA-SVR | MLR, PLS, SVR, k-NN | MLR {CrippenLogP, SHBa, ALogP, RDF30i}: Q²<sub>LOO</sub> 0.931, R²<sub>VS</sub> 0.973 | Lipophilicity descriptors (CrippenLogP, ALogP, XLogP) dominate every model; PLS VIP confirms it |
| 3 | [Domain-safe descriptors](stages/3_domain_safe_descriptors) | only descriptors also valid for the external compounds (XLogP, TopoPSA, nAromBond) | MLR, PLS, SVR, k-NN | SVR: Q²<sub>LOO</sub> 0.872, R²<sub>VS</sub> 0.937; MLR fell to R²<sub>VS</sub> 0.62 | Once descriptors must also cover new compounds, much of the stage 1–2 performance disappears |
| 4 | [Automated workflow + external check](stages/4_auto_workflow_external_validation) | automated pruning of collinear descriptors, ranking by a combined score, applicability domain | MLR, PLS, SVR, k-NN | SVR {RDF135e, CrippenLogP, SHBa}: best overall score, 0 VS compounds outside the AD | Naphthalene predicted reasonably, but **nitrobenzene failed for every model** (errors 1.0–2.9 log units, outside the AD) |
| 5 | [LSER run 1](stages/5_lser_run1) | switched to 6 mechanistic LSER descriptors (log D, M′<sub>w</sub>, ε<sub>α</sub>, ε<sub>β</sub>, V′, π), scaled on the training set only | MLR, k-NN, decision tree | MLR {log D, ε<sub>β</sub>, π}: Q²<sub>CV</sub> 0.871, R²<sub>VS</sub> 0.969 | Comparable accuracy with 3 interpretable descriptors instead of 4 abstract ones. k-NN with k = 1 reached R²<sub>TS</sub> = 1.0, i.e. memorisation |
| 6 | [LSER run 2 + reference check](stages/6_lser_run2_reference_check) | MLR-2 {log D, ε<sub>β</sub>} vs MLR-4 (+ ε<sub>α</sub>, V′) | MLR, k-NN, decision tree | MLR-2: Q²<sub>CV</sub> 0.897, R²<sub>VS</sub> 0.922 | MLR-4 matched the literature values best, but its nitrobenzene prediction was outside the AD. A good match outside the domain is not validation |
| 7 | [Final model](final/) | {log D, ε<sub>β</sub>}, chosen a priori from mechanism and collinearity | MLR, k-NN, decision tree (+ SVR, PLS cross-check in [Orange](orange/)) | MLR: Q²<sub>ext</sub> 0.909 ± 0.035 over 200 splits | A single train/test split was misleading. The 200-split stability test decided the final model |

## Algorithm comparison

| Algorithm | Strengths seen in this project | Weaknesses seen in this project |
|---|---|---|
| **MLR** | Explicit equation with interpretable signs. Most stable across 200 random splits (never below 0.78). Smallest gap between fit and prediction (R² 0.939 vs Q² 0.913). Can extrapolate linearly. | Sensitive to collinearity (needs VIF control). With thousands of candidate descriptors, subset search easily finds chance correlations (stage 1). |
| **PLS** | Handles many correlated descriptors without explicit selection, and VIP ranks descriptor importance. Highest single-split R²<sub>VS</sub> (0.983, stage 4). | Many-descriptor models (15–19 variables) are hard to interpret. Least robust once restricted to domain-safe descriptors (R²<sub>VS</sub> 0.54). Worst external prediction (nitrobenzene error 2.9). |
| **SVR (RBF)** | Most robust in the PaDEL stages, including the domain-safe set (R²<sub>VS</sub> 0.94). No VS compounds outside the AD in stage 4. | Black box. Needs a grid search over C, ε and γ. The GA-selected 2-descriptor SVR was unstable in cross-validation (CV R² 0.42 despite R²<sub>VS</sub> 0.96). |
| **k-NN** | Simple and non-parametric. Confirms log D as the key neighbour metric. | LOO Q² falls steadily as k grows (0.93 → 0.33 for k = 1 → 10), a sign of memorisation. Cannot extrapolate beyond the training range. Wide spread across splits (0.73 ± 0.15). |
| **Decision tree** | Shows the log D dominance directly (feature importance 0.97). | Step-wise predictions. Least stable model (Q²<sub>ext</sub> 0.80 ± 0.21, minimum −0.50). Threshold artefact for naphthalene (predicted 0.72 vs reference 3.7). |

**Closest predictions overall:**
- On a single split, the PaDEL-based MLR, SVR and PLS (stages 1, 2 and 4) reached the highest R²<sub>VS</sub> (0.96–0.98).
- When stability, interpretability and behaviour on new compounds are also counted, **MLR {log D, ε<sub>β</sub>}** is the most reliable.
- SVR was the best of the non-linear methods.

## External compounds: naphthalene and nitrobenzene

| Stage / model | Naphthalene | Nitrobenzene | Comment |
|---|---|---|---|
| 3: SVR (domain-safe) | 5.19 | 2.00 | Overestimates naphthalene |
| 4: MLR (PaDEL) | 3.72 | 1.29 | Nitrobenzene is a residual outlier |
| 4: PLS (PaDEL) | 3.90 | 0.28 | Nitrobenzene outside the AD |
| 6: MLR-4 (LSER) | 3.76 | 2.31 | Closest values, but nitrobenzene is outside the AD |
| 6: decision tree | 0.72 | 2.44 | Threshold artefact |
| **7: final MLR** | **3.50** (inside AD) | **1.81** (borderline) | Nitrobenzene falls in an empty data gap (log D ≈ 1–4.2) |

Reference values: naphthalene ≈ 3.7–4.1 and nitrobenzene ≈ 2.5–3.2, depending on the source. They were used **only after** the models were built.

## Problems found and fixed along the way

- **Corrupted external rows.** The descriptor rows for naphthalene and nitrobenzene in the course Excel file had ε<sub>α</sub> = 0 and wrong molar masses. Corrected descriptors were used instead.
- **Spurious ε<sub>α</sub>.** Its correlation with log K<sub>d,a</sub> came entirely from two pharmaceuticals with high leverage. Within the 33 hydrophobic compounds it dropped to r = −0.15, so it was excluded.
- **Data leakage.** Scaling, descriptor selection and hyper-parameter tuning were done on the training set only. External reference values were never used for model choice.
- **Misleading single splits.** The final 70:30 split contained only interpolation cases (log K<sub>d,a</sub> 4.5–6.1), so all models looked similar on it. The 200-split test separated them clearly.

## Data sources

- **Experimental log K<sub>d,a</sub> (PP–seawater) for the 35 compounds** and the 6 LSER descriptors (log D, M′<sub>w</sub>, ε<sub>α</sub>, ε<sub>β</sub>, V′, π) come from Li Y. et al. (2020), *QSPR models for predicting the adsorption capacity for microplastics of polyethylene, polypropylene and polystyrene*, **Scientific Reports 10:14597**, [doi:10.1038/s41598-020-71390-3](https://doi.org/10.1038/s41598-020-71390-3). The course compiled them into `data/data_qsar_pls.xlsx`.
- **PaDEL descriptors** (~2000 per compound) were computed with PaDEL-Descriptor. MOPAC PM6 geometries are in `quantum_chemistry/`.
- **External compounds (naphthalene, nitrobenzene):** descriptors are in `data/external_naphthalene_nitrobenzene.xlsx`. Reference sorption values were taken from the literature listed in the final report (e.g. Lee et al. 2014, Hüffer & Hofmann 2016).

None of the data are personal or sensitive. They are literature values and computed descriptors.

## Repository structure

```
final/                 final report, presentations (PDF + PPTX), figures
stages/1…6/            every earlier iteration: scripts, outputs, reports
orange/                Orange Data Mining workflow (Linear Regression, kNN, PLS, SVM, Test & Score)
data/                  modelling datasets + external-set descriptors
descriptors/           PaDEL 2D/3D descriptors per compound
quantum_chemistry/     MOPAC PM6 optimisation (gas phase and COSMO, ε = 78.4) for 5 compounds
docs/                  development-journey chart
```

The scripts used for stages 6–7 are not included. Their outputs, figures and reports are.

## How to run

```bash
pip install -r requirements.txt
cd stages/2_padel_extended_pipeline
python qspr_final_ks_based_on_teacher.py --excel ../../data/data_qsar_pls.xlsx --output results_kennard_stone
```

On Windows, set `PYTHONIOENCODING=utf-8` first. Each script takes `--help`. The stage 1 and 3 scripts read `data_qsar_pls.xlsx` from the working directory, so copy it from `data/` before running them.

**Tools:** Python (pandas, scikit-learn, matplotlib), PaDEL-Descriptor, MOPAC (PM6), Orange Data Mining.

## AI assistance

The code in this repository was written with the help of AI tools (large language models). Defining the tasks, running the analyses, and checking and interpreting the results were my part of the work.

---

## 🇵🇱 Opis po polsku

Projekt z *Modelowania QSAR/QSPR*: przewidywanie współczynnika podziału mikroplastik (PP)–woda morska log K<sub>d,a</sub> dla 35 związków organicznych. Repozytorium pokazuje **całą drogę dojścia do modelu**, a nie tylko wynik końcowy:

1. **Deskryptory PaDEL (~2000):** przeszukiwanie podzbiorów, MLR, PLS, SVR i k-NN. Wyniki bardzo wysokie (Q² ≈ 0,93), ale przy 35 związkach istnieje ryzyko przypadkowych korelacji.
2. **Rozszerzony pipeline:** filtry wariancji i korelacji, PLS-VIP, SVR-RFE, GA-SVR, diagnostyka i wykres Williamsa. We wszystkich modelach dominuje lipofilowość.
3. **Tylko deskryptory dostępne dla związków zewnętrznych:** wyniki wyraźnie spadają (MLR R²<sub>VS</sub> = 0,62). Najodporniejszy okazał się SVR.
4. **Automatyczny workflow z walidacją zewnętrzną:** naftalen jest przewidywany poprawnie, nitrobenzen przez wszystkie modele źle (poza domeną).
5. **Przejście na 6 deskryptorów LSER:** podobna trafność przy pełnej interpretowalności. k-NN z k = 1 tylko zapamiętuje dane.
6. **Porównanie MLR-2 z MLR-4 i wartościami referencyjnymi:** dobra zgodność poza domeną stosowalności nie jest walidacją.
7. **Model końcowy MLR {log D, ε<sub>β</sub>}:** R² = 0,939, Q²<sub>LOO</sub> = 0,913. Test stabilności na 200 podziałach dał Q²<sub>ext</sub> = 0,909 ± 0,035, wyraźnie lepiej niż k-NN (0,73 ± 0,15) i drzewo decyzyjne (0,80 ± 0,21).

Dane eksperymentalne pochodzą z pracy Li i wsp. (2020), *Scientific Reports* 10:14597. Deskryptory PaDEL i obliczenia MOPAC zostały wykonane w ramach projektu.

Zalety i wady każdego algorytmu opisuje tabela *Algorithm comparison* powyżej. Raporty i prezentacje są w folderach `final/` oraz `stages/`.

Projekt studencki (studia II stopnia), Politechnika Gdańska, 2026.

**Wsparcie AI:** kod w tym repozytorium powstał z pomocą narzędzi AI (dużych modeli językowych). Określenie zadań, uruchamianie analiz oraz sprawdzenie i interpretacja wyników były moją częścią pracy.
