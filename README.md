# Misinformation Detection Pipeline

Seven stages turn four public datasets into a misinformation classifier (an ensemble of one
expert model per dataset), a LIAR truthfulness-score regressor (0–5) and topic clusters. Every
stage is a notebook that reads files from earlier stages and writes files for later ones. The
same notebooks run on Kaggle and locally. Design and reasons: see `PIPELINE_DRAFT.md`.

## Project layout

```
misinfo-project/
├── environment.yml          conda environment
├── run_pipeline.py          runs the notebooks in stage order (local only)
├── src/misinfo_utils.py     shared code (Kaggle: Utility Script named misinfo_utils)
├── data/raw/                the 4 raw datasets (local only)
├── notebooks/
│   ├── 01a_preprocess_liar.ipynb        01b_preprocess_welfake.ipynb
│   ├── 01c_preprocess_covid.ipynb       01d_preprocess_fakenewsnet.ipynb
│   ├── 02_split.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_features.ipynb
│   ├── 05a_train_logreg.ipynb           05b_train_complementnb.ipynb
│   ├── 05c_train_linearsvm.ipynb        05d_train_lightgbm.ipynb
│   ├── 05e_train_distilbert.ipynb       05f_train_regression.ipynb
│   ├── 05g_train_clustering.ipynb
│   ├── 06_ensemble.ipynb
│   └── 07_evaluation.ipynb
└── outputs/                 one folder per stage, created when it runs
```

## Raw files needed

| Dataset | Files (found by name, any subfolder) |
|---|---|
| LIAR | `train.tsv`, `valid.tsv`, `test.tsv` |
| WELFake | `WELFake_Dataset.csv` |
| COVID-19 Constraint | `Constraint_Train.csv`, `Constraint_Val.csv`, `english_test_with_labels.csv` |
| FakeNewsNet (KaiDMML) | `politifact_fake.csv`, `politifact_real.csv`, `gossipcop_fake.csv`, `gossipcop_real.csv` |

## Run locally

```bash
conda env create -f environment.yml
conda activate misinfo
# put the raw files anywhere inside data/raw/ (subfolders are fine)
python run_pipeline.py              # every stage in order
python run_pipeline.py 05 06 07     # or only some stages
```

Or open the notebooks in Jupyter (`jupyter lab`) and run them in order. Outputs go to
`outputs/<stage>/`.

**DistilBERT on a laptop** (CPU) is very slow. Train `05e_train_distilbert` on Kaggle with a
GPU, download its four `distilbert__<dataset>` folders into `outputs/05_models/`, and the local
notebook will reuse them and only make predictions.

## Run on Kaggle

1. **Utility script (once):** create a notebook, paste `src/misinfo_utils.py`, name it exactly
   `misinfo_utils`, **File → Set as Utility Script**, **Save Version**. After any change to the
   file, save a new version.
2. **Each notebook:** File → Import Notebook; **File → Add utility script → misinfo_utils**;
   **Add Input** as below; then **Save Version → Save & Run All**.

| Notebook | Add as inputs | Accelerator |
|---|---|---|
| 01a–01d | the raw dataset it needs | none |
| 02_split | notebooks 01a, 01b, 01c, 01d | none |
| 03_eda | 02_split | none |
| 04_features | 02_split | none |
| 05a–05d, 05f, 05g | 02_split, 04_features | none |
| 05e_train_distilbert | 02_split (Internet **on**) | **GPU T4** |
| 06_ensemble | 02_split, all of 05a–05g | none |
| 07_evaluation | 02_split, 04_features, all of 05a–05g, 06_ensemble | GPU if an expert is DistilBERT |

## What each stage produces

| Stage | Output files |
|---|---|
| 1 Preprocess | `liar_cleaned.csv`, `welfake_cleaned.csv`, `covid_cleaned.csv`, `fakenewsnet_cleaned.csv` |
| 2 Split | `<dataset>_train/val/test.csv` × 4, `liar_reg_train/val/test.csv` |
| 3 EDA | none (tables and plots in the notebook) |
| 4 Features | per key (`liar`, `welfake`, `covid`, `fakenewsnet`, `liar_reg`): `features__<key>.joblib`, `X_tfidf__<key>__<split>.npz`, `X_dense__<key>__<split>.npy` |
| 5 Training | checkpoints `<model>__<dataset>.joblib` (DistilBERT: `distilbert__<dataset>/` folders), predictions `pred__<model>__<dataset>.csv`, clusters `clusters__<dataset>.csv`, `clustering_k.json` |
| 6 Ensemble | `final_model.json`, `blender.joblib`, `pred__final.csv`, `reg_final.json`, `pred__average_reg__liar.csv` |
| 7 Evaluation | `results_in_dataset.csv`, `results_cross_dataset.csv`, `results_final.csv`, `results_regression.csv`, `results_liar_comparison.csv`, `results_clustering.csv`, `error_analysis_sample.csv` |

Checkpoints: 20 classification (LogReg, ComplementNB, LinearSVM, LightGBM, DistilBERT × 4
datasets), 3 regression (Dummy, Ridge, LightGBM regressor on LIAR), 8 clustering (K-Means and
Gaussian Mixture × 4 datasets). Each `pred__<model>__<dataset>.csv` holds that checkpoint's
P(misleading) for the validation and test parts of **all four** datasets
(`row_id`, `tested_on`, `split`, `prob_misleading`).

## Predict new text

```python
import misinfo_utils as mu
mu.setup("07")                      # finds the saved files (locally or on Kaggle)
model = mu.FinalModel()             # loads the 4 experts + combiner chosen in Stage 6
model.predict_proba(["Scientists confirm the moon landing was filmed in a studio."])
# -> array([P(misleading)])
```

Labels everywhere: **0 = reliable, 1 = misleading**. LIAR score: 0 pants-fire … 5 true.
