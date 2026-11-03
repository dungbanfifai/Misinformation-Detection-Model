# Misinformation Detection: Pipeline Design (Draft)

**Status:** all seven stages implemented and tested on synthetic data; not yet run on the real data
**Tasks:** A. Classification (per dataset, then a cross-dataset ensemble) · B. Regression (LIAR 0–5 score) · C. Clustering (per dataset)
**Runs on:** Kaggle **and** locally (conda), with the same notebooks
**Rule for every stage:** a notebook only reads files written by an earlier stage and writes files
that a later stage reads. Nothing is passed through memory between notebooks.

**Change history** (newest first)
- Stages 3–7 implemented. Each checkpoint now writes ONE prediction file
  (`pred__<model>__<dataset>.csv`) covering all datasets' val and test parts; clustering's
  comparison method is Gaussian Mixture (it can assign new texts); LightGBM is used without an
  XGBoost comparison. `mu.FinalModel` loads the final ensemble for Stage 7 and the website.
- Stage 2 implemented (`02_split`). Its only outputs are the 15 split files; the split summary
  is shown in the notebook. Cross-dataset duplicates are kept in the first dataset of
  LIAR → COVID → FakeNewsNet → WELFake.
- Stage 1 saves only the four cleaned files (`<dataset>_cleaned.csv`); cleaning logs
  and plots stay inside the notebooks. 
- New Stage 6: pick the **best model from each dataset** (on validation), then ensemble
  those four "dataset experts" into one final classifier, which the website also uses. Stage 5
  now predicts every dataset's validation **and** test set (needed to train and check the
  ensemble). Per-dataset blenders and `FINAL_MEMBERS` removed.
- Models are now trained **per dataset**, with no merged training set: 20 classification
  checkpoints (5 models × 4 datasets), 3 regression checkpoints (LIAR), 8 clustering checkpoints
  (2 methods × 4 datasets). Every classifier is tested on all four test sets, so the
  cross-dataset matrix is built into the main pipeline (the optional `08` notebook is no longer
  needed). Features and ensembles are per dataset. MLP tier dropped.
- Stage 1 implemented: `misinfo_utils.py`, notebooks 01a–01d, `environment.yml`,
  `run_pipeline.py`, README. Tested in Kaggle and local mode.
- Added clustering, model tiers, Kaggle + local running.
- First pipeline draft, split from the single notebook (`..._ver9.ipynb`).

---

## 1. Checkpoints at a glance

| Task | Models | Datasets | Checkpoints |
|---|---|---|---|
| A. Classification | LogReg, ComplementNB, LinearSVM, LightGBM, DistilBERT | LIAR, WELFake, COVID, FakeNewsNet | **5 × 4 = 20** |
| B. Regression | Dummy (baseline), Ridge, LightGBM regressor | LIAR only (the only dataset with 0–5 ratings) | **3** |
| C. Clustering | K-Means, Gaussian Mixture | LIAR, WELFake, COVID, FakeNewsNet | **2 × 4 = 8** |
| Final ensemble (Stage 6) | Best checkpoint from each dataset, combined | all 4 | 1 (built from 4 of the 20) |

Checkpoint names: `<model>__<dataset>`, e.g. `linearsvm__covid.joblib`, `distilbert__liar/`.

Why regression is not 20: WELFake, COVID and FakeNewsNet only say real or fake, so there is no
number to predict. Regression stays on LIAR and is compared with the LIAR classifiers.

---

## 2. Pipeline overview (stages = notebooks)

```
 RAW FILES  (Kaggle inputs  /  data/raw/ locally)
   │
   ▼
 STAGE 1  PREPROCESSING — 01a liar · 01b welfake · 01c covid · 01d fakenewsnet      [DONE]
   │  → liar_cleaned.csv · welfake_cleaned.csv · covid_cleaned.csv · fakenewsnet_cleaned.csv
   ▼
 STAGE 2  SPLIT — 02_split                                                          [DONE]
   │  → <dataset>_train/val/test.csv           (Task A + C, 4 datasets, LIAR without half-true)
   │  → liar_reg_train/val/test.csv             (Task B, all 6 ratings, same LIAR split)
   ▼
 STAGE 3  EDA — 03_eda                                                             [DONE]
   │
 STAGE 4  FEATURES — 04_features                                                   [DONE]
   │  → features__<dataset>.joblib  (one TF-IDF + SVD + scaler per dataset, fitted on its train)
   │  → features__liar_reg.joblib
   ▼
 STAGE 5  TRAINING — one notebook per model; each loops over the 4 datasets        [DONE]
   05a logreg · 05b complementnb · 05c linearsvm · 05d lightgbm · 05e distilbert   → 20 checkpoints
   05f regression (Dummy, Ridge, LightGBM regressor on LIAR)                        →  3 checkpoints
   05g clustering (K-Means + Gaussian Mixture, per dataset)                         →  8 checkpoints
   ▼
 STAGE 6  SELECT + ENSEMBLE — 06_ensemble  (validation predictions only)            [DONE]
   │  1. best of 5 models for each dataset → 4 "dataset experts"
   │  2. combine the 4 experts → final classifier (used by the website)
   ▼
 STAGE 7  EVALUATION — 07_evaluation  (the ONLY place test sets are scored)          [DONE]
```

### One notebook per model, not per checkpoint

Each Stage 5 classification notebook trains the same model four times:

```
for dataset in ["liar", "welfake", "covid", "fakenewsnet"]:
    load <dataset>_train / _val  and  features__<dataset>
    train the model             → save <model>__<dataset>
    predict ALL 4 datasets' val and test sets (with this dataset's feature builder)
        own val   → choosing the best model for this dataset (Stage 6)
        all val   → training the final ensemble (Stage 6)
        all test  → in-dataset and cross-dataset results (Stage 7)
```

Five notebooks instead of twenty keeps the code in one place per model, and every checkpoint is
trained with exactly the same settings.

---

## 3. Stage-by-stage plan

### Stage 1: preprocessing — unchanged (done)

### Stage 2: split (was "merge and split")

- No merged training set. Each dataset is split **on its own**: 70 / 15 / 15, stratified by label.
- LIAR is split **once**, stratified by its 0–5 score. Classification uses those rows minus
  half-true; regression uses all of them. The same LIAR statement is therefore in the same part
  (train / val / test) for both tasks.
- Cross-dataset duplicates are still removed (LIAR and FakeNewsNet's PolitiFact part overlap).
  Otherwise a model could be "tested" on another dataset with a text it trained on. A shared
  text is kept in the first dataset of LIAR → COVID → FakeNewsNet → WELFake.
- `WELFAKE_MAX_ROWS` stays as a setting, now to limit **training time** (63k long articles is
  slow for DistilBERT), not to balance datasets against each other.
- `BALANCE_GOSSIPCOP` stays as an option: it balances FakeNewsNet's own classes.

Outputs: `liar_train/val/test.csv`, `welfake_…`, `covid_…`, `fakenewsnet_…`,
`liar_reg_train/val/test.csv` (15 files).

### Stage 4: features — one builder per dataset

Each dataset gets its own TF-IDF vocabulary, fitted on **its** training rows only. When a model
trained on dataset X is tested on dataset Y, Y's text goes through **X's** feature builder (the
model only knows X's vocabulary; this is part of what the cross-dataset test measures).

### Stage 5: training

| Notebook | Checkpoints | Input | Notes |
|---|---|---|---|
| 05a LogReg | `logreg__<ds>` × 4 | TF-IDF | |
| 05b ComplementNB | `complementnb__<ds>` × 4 | TF-IDF | |
| 05c LinearSVM | `linearsvm__<ds>` × 4 | TF-IDF | calibrated for probabilities |
| 05d LightGBM | `lightgbm__<ds>` × 4 | SVD + style | fixed settings |
| 05e DistilBERT | `distilbert__<ds>/` × 4 | raw text | GPU; 4 fine-tunes, ~30–45 min total on Kaggle T4; ~270 MB each |
| 05f Regression | `dummy_reg__liar`, `ridge_reg__liar`, `lightgbm_reg__liar` | LIAR features | |
| 05g Clustering | `kmeans__<ds>`, `<comparison>__<ds>` × 4 | SVD (no labels) | k chosen per dataset (elbow + silhouette) |

**Prediction files** (all matched by `row_id`):

| File | Columns |
|---|---|
| `pred__<model>__<trained_on>.csv` (one per checkpoint: all 4 datasets' val and test parts) | `row_id`, `tested_on`, `split`, `prob_misleading` |
| `pred__<reg_model>__liar.csv` | `row_id`, `split`, `pred_score` |
| `clusters__<ds>.csv` | `row_id`, `split`, `kmeans`, `gmm` |

### Stage 6: select the best model per dataset, then ensemble them

```
            LIAR models (5)     WELFake models (5)     COVID models (5)     FakeNewsNet models (5)
                  │                    │                     │                      │
  Step 1    best on LIAR val    best on WELFake val   best on COVID val   best on FakeNewsNet val
                  │                    │                     │                      │
                  ▼                    ▼                     ▼                      ▼
             LIAR expert         WELFake expert         COVID expert       FakeNewsNet expert
                  └──────────────┬─────┴──────────────┬──────┘                      │
  Step 2                         └──── combine 4 × P(misleading) ◄───────────────────┘
                                               │
                                       FINAL CLASSIFIER
```

**Step 1: choose one expert per dataset**
- Criterion: highest **macro F1 on that dataset's own validation set**; tie-break: faster
  prediction. Test sets are not used, so the choice stays fair.
- Saved as a selection table (dataset × 5 models, winner highlighted) for the report.
- The same model type can win several datasets (e.g. DistilBERT four times). That is fine: the
  four copies were trained on different data, so they still disagree in useful ways.

**Step 2: combine the four experts**

Each expert gives a probability of "misleading" for **any** text, so for every validation row of
every dataset there are 4 probabilities. Three combiners are compared:

| Combiner | How | Notes |
|---|---|---|
| Soft vote | average of the 4 probabilities | baseline, nothing to train |
| Weighted average | weights chosen on validation (e.g. by each expert's validation F1) | simple, explainable |
| **Blender** | Logistic Regression on the 4 probabilities, trained on all four validation sets together | learns when to trust which expert; its weights go in the report |

- The blender is trained with **each dataset weighted equally** (sample weights), so WELFake's
  larger validation set does not dominate.
- Its validation score uses 5-fold cross-validation (it is trained on the same rows).
- The best combiner (by validation macro F1 across all four datasets) becomes the final classifier.

Outputs: `final_model.json` (the chosen experts, combiner and weights), `blender.joblib`,
`pred__final.csv`. The selection table is shown in the notebook.

**Optional check:** choosing the expert by its **average** validation F1 over all four datasets
instead of its own dataset only. Report it only if it picks different experts.

### Stage 7: evaluation

| Section | Contents |
|---|---|
| **A. In-dataset results** | For each dataset: accuracy, precision, recall, macro F1, ROC-AUC of its 5 models. 4 tables, plus one summary table (macro F1, model × dataset) with the chosen expert marked |
| **A. Final ensemble** | Final classifier vs each expert alone, on each test set and on all four together; confusion matrix; ms per text. This is the website's accuracy |
| **A. Cross-dataset matrix** | For each model: 4 × 4 heatmap of macro F1 (rows = trained on, columns = tested on). Diagonal = in-dataset, off-diagonal = generalisation |
| **A. Best model per cell** | Which model generalises best from each dataset to each other |
| **B. Regression (LIAR)** | MAE, RMSE, R², within-±1 % for Dummy, Ridge, LightGBM; predicted-vs-true plot |
| **LIAR across all models** | LIAR row and column in every matrix; LIAR regression vs the 5 LIAR classifiers on the **same** test statements (half-true excluded) |
| **C. Clustering** | Per dataset: silhouette and Davies–Bouldin for both methods; cluster profiles (size, top words, % misleading); 2-D scatter |
| Discussion | Error analysis sample, top words per dataset, LIME example |

---

## 4. Consequences to plan for

1. **The website uses the final ensemble** (4 experts + combiner). It must load each expert with
   its own feature builder (or DistilBERT tokenizer), so 4 experts' files + `combiner.joblib`.
   If an expert is DistilBERT, the website needs ~270 MB per DistilBERT expert and is slower on
   CPU; the Stage 7 speed column shows the cost.
2. **Per-dataset training is smaller.** Each model sees 7–20k training rows instead of ~35k
   merged. In-dataset scores may drop slightly; the cross-dataset matrix will show how much
   each dataset's model generalises.
3. **DistilBERT cost:** 4 fine-tunes and 4 saved models (~1.1 GB). Fine on Kaggle; locally,
   train on Kaggle and copy the folders (05e skips training when a checkpoint folder exists).
4. **Report size:** 4 datasets × 5 models produces many numbers. Lead with the summary table and
   the cross-dataset heatmaps; put the per-dataset detail tables in the appendix.

---

## 5. Running on Kaggle and locally — unchanged

`misinfo_utils.setup("<stage>")` decides the read/write folders (Kaggle: `/kaggle/input` →
`/kaggle/working`; local: `data/raw` + `outputs/` → `outputs/<stage>/`). Every output file name
is unique, because notebooks find inputs by name. Kaggle: chain notebooks with **Add Input → Your
Work → Notebooks**. Local: `python run_pipeline.py`.

---

## 6. Still open

- Nothing in the design. Next: run Stages 1–7 on the real data and check the outputs listed in the
  README.
