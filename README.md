# Misinformation Detection Model

COS30049 Computing Technology Innovation Project, Assignment 2 (AI4Cyber: misinformation detection on social media).

This project trains the machine-learning models behind a misinformation-checking website. A user pastes a
claim, post, headline or article and gets:

- a **label**: reliable or misleading, from an ensemble of four DistilBERT "experts" (one per kind of text)
  combined by a blender;
- a **truthfulness score** from 0 (pants on fire) to 5 (true), from a Ridge regression model trained on LIAR;
- **topic clusters** (K-Means) that show which topics in the data lean towards misinformation.

Labels everywhere: **0 = reliable, 1 = misleading**. LIAR score: 0 pants-fire … 5 true.

The whole pipeline is seven stages of Jupyter notebooks. Each stage reads files written by earlier stages
and writes files for later ones. The same notebooks run locally (conda) and on Kaggle.

## Quick guide

| Task | Section |
|---|---|
| 1. Configure the project environment with conda | [Set up the environment](#1-set-up-the-environment-conda) |
| 2. Further data processing from our prepared training data | [Further data processing](#2-further-data-processing-from-the-prepared-data) |
| 3. Train the models | [Train the models](#3-train-the-models) |
| 4. Use the model for prediction | [Predict new text](#4-predict-new-text) |

---

## Project layout

```
Misinformation-Detection-Model/
├── .gitignore               files Git leaves out (raw data, feature matrices, caches)
├── .gitattributes           files stored with Git LFS (model weights, welfake_cleaned.csv)
├── environment.yml          conda environment
├── run_pipeline.py          runs the notebooks in stage order (local only)
├── src/misinfo_utils.py     shared code: paths, cleaning, loading, FinalModel
│                            (on Kaggle: a Utility Script named misinfo_utils)
├── data/raw/                the 4 raw datasets (not in the repo; see "Datasets")
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
└── outputs/                 one folder per stage (see "What each stage produces")
```

---

## Get the project

Install Git LFS once, then clone:

```bash
git lfs install
git clone https://github.com/dungbanfifai/Misinformation-Detection-Model.git
cd Misinformation-Detection-Model
```

The repository includes the **processed data** (`outputs/01_preprocess`, `outputs/02_split`) and the
**trained models** (`outputs/05_models`, `outputs/06_ensemble`). The four `model.safetensors` files and
`welfake_cleaned.csv` are stored with Git LFS.

If you cloned before installing Git LFS (the model files are only a few hundred bytes), run
`git lfs install` and then `git lfs pull` inside the folder.

---

## Datasets

The raw datasets are not in the repository. They are only needed to rerun Stage 1. Download them and put
the files anywhere inside `data/raw/`; subfolders are fine, because files are found by name.

| Dataset | Kind of text | Files | Downloaded from | Original paper |
|---|---|---|---|---|
| LIAR | short US political claims (PolitiFact) | `train.tsv`, `valid.tsv`, `test.tsv` | [Kaggle: doanquanvietnamca/liar-dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset) | Wang (2017) |
| WELFake | full news articles | `WELFake_Dataset.csv` | [Kaggle: saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) | Verma et al. (2021) |
| COVID-19 Constraint | social-media posts | `Constraint_Train.csv`, `Constraint_Val.csv`, `english_test_with_labels.csv` | [Kaggle: elvinagammed/covid19-fake-news-dataset-nlp](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp) | Patwa et al. (2021) |
| FakeNewsNet | news headlines (PolitiFact, GossipCop) | `politifact_fake.csv`, `politifact_real.csv`, `gossipcop_fake.csv`, `gossipcop_real.csv` | [GitHub: KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) (our Kaggle copy: [phmdng1827/fakenewsnet](https://www.kaggle.com/datasets/phmdng1827/fakenewsnet)) | Shu et al. (2020) |

Notes on the labels:

- **WELFake:** its documentation says 1 = real, but the data shows the opposite (Reuters is named in 61.8%
  of label-0 articles and 1.8% of label-1 articles), so the pipeline uses **0 = real, 1 = fake**.
- **LIAR:** pants-fire, false and barely-true become misleading; mostly-true and true become reliable;
  half-true is left out of classification but kept for regression (scores 0–5).
- **COVID-19:** use `english_test_with_labels.csv`, not the unlabelled test file.

---

## 1. Set up the environment (conda)

Run these commands in Anaconda Prompt (Windows) or a terminal (macOS/Linux), from the project folder:

```bash
conda env create -f environment.yml                 # create the "misinfo" environment (once)
conda activate misinfo                              # activate it (every session)
python -m ipykernel install --user --name misinfo   # make it selectable in Jupyter (once)
jupyter lab                                         # open the notebooks
```

In Jupyter, choose the **misinfo** kernel for every notebook. To update the environment after
`environment.yml` changes: `conda env update -f environment.yml --prune`.

On Windows, if Jupyter opens in `C:\`, start it from the project folder instead
(`cd /d D:\path\to\Misinformation-Detection-Model`, then `jupyter lab`).

---

## 2. Further data processing (from the prepared data)

The repository (and the submitted zip) contains our **prepared datasets**, so you do not need the raw data:

| Folder | Files | Contents |
|---|---|---|
| `outputs/01_preprocess/` | `liar_cleaned.csv`, `welfake_cleaned.csv`, `covid_cleaned.csv`, `fakenewsnet_cleaned.csv` | cleaned texts with labels (0 = reliable, 1 = misleading), one file per dataset |
| `outputs/02_split/` | `<dataset>_train.csv`, `<dataset>_val.csv`, `<dataset>_test.csv` × 4 | the stratified 70/15/15 splits used for classification (50,791 texts) |
| `outputs/02_split/` | `liar_reg_train.csv`, `liar_reg_val.csv`, `liar_reg_test.csv` | LIAR with its 0–5 score, used for regression (12,791 claims) |

Each split file has one row per text, with the cleaned `text`, the `label` and helper columns such as the
word count (`n_words`) and the source dataset. The LIAR regression files also have the `score` (0–5).

Cleaning already done in Stage 1: HTML entities decoded; links, HTML tags, hashtags, @mentions and source
names that give the label away (for example "Reuters") removed; empty, too-short, conflicting and duplicate
texts removed. Stage 2 removed texts shared between datasets, capped WELFake at 20,000 articles and
balanced GossipCop.

Further processing starts from these files:

1. **Explore the data:** run `03_eda.ipynb` (class balance, text length, LIAR ratings, words typical of each
   class, shortcut checks).
2. **Build the features:** run `04_features.ipynb`. For each dataset it fits, on the training split only:
   - TF-IDF on words and word pairs (for Logistic Regression, Complement NB, Linear SVM and Ridge);
   - TF-IDF reduced to 300 dimensions with truncated SVD, plus 8 style features (for LightGBM and
     clustering).

   It saves them to `outputs/04_features/<key>_preprocessed/`. DistilBERT does not use these; it reads
   the cleaned text directly.

```bash
python run_pipeline.py 03 04        # explore the data and build the features
```

To change the cleaning itself (for example the minimum text length), edit the cleaning functions in
`src/misinfo_utils.py` or the settings cell of the 01a–01d notebooks, put the raw data in `data/raw/`
(see "Datasets") and rerun from Stage 1: `python run_pipeline.py`.

---

## 3. Train the models

Train after Stage 4 has built the features. Each training notebook trains one model type on each of the
four datasets and saves the checkpoints and predictions to `outputs/05_models/`.

| Notebook | Trains | Notes |
|---|---|---|
| `05a_train_logreg` | Logistic Regression × 4 | C tuned by grid search (1, 4, 16) |
| `05b_train_complementnb` | Complement Naive Bayes × 4 | alpha tuned (0.01–1.0) |
| `05c_train_linearsvm` | Linear SVM × 4 (calibrated) | C tuned (0.1, 0.5, 2.0) |
| `05d_train_lightgbm` | LightGBM × 4 | fixed settings (400 trees, 31 leaves, learning rate 0.05) |
| `05e_train_distilbert` | DistilBERT × 4 | needs a GPU: train on Kaggle (see below) |
| `05f_train_regression` | Dummy, Ridge, LightGBM regressor on LIAR (0–5 score) | Ridge is the chosen model |
| `05g_train_clustering` | K-Means and Gaussian Mixture × 4 | k chosen by silhouette score (2–10) |

Then build and test the final model:

| Notebook | Does |
|---|---|
| `06_ensemble` | picks the best model per dataset on validation data (DistilBERT for all four) and trains the blender that combines them; saves `final_model.json` and `blender.joblib` |
| `07_evaluation` | scores everything once on the test splits and saves the results tables to `outputs/07_evaluation/` |

```bash
python run_pipeline.py 05 06 07     # train, build the ensemble, evaluate
python run_pipeline.py              # or the whole pipeline from Stage 1
```

`run_pipeline.py` runs each notebook in place (so its tables and plots are saved inside it) and stops at
the first failure. You can also run the notebooks in Jupyter in this order:
01a–01d → 02_split → 03_eda and 04_features → 05a–05g → 06_ensemble → 07_evaluation.
Always run the stages in order: each stage needs the files written by the stages before it.

**DistilBERT:** on a laptop CPU it takes many hours per dataset, so train it on Kaggle (see "Run on
Kaggle"). The local `05e_train_distilbert` notebook reuses any saved model folder it finds and then only
makes predictions; `RETRAIN = False` in its settings cell controls this. Settings used:
`distilbert-base-uncased`, up to 256 tokens per text, learning rate 2e-5, batch size 16, 10% warm-up, up
to 3 epochs with early stopping on validation loss (the epoch with the lowest validation loss is kept).

**Windows note:** if a notebook fails with `WinError 127 ... torch\lib\shm.dll`, put `import torch` as the
**first line** of the first code cell, restart the kernel and run again.

---

## Run on Kaggle

1. **Utility script (once):** create a notebook, paste all of `src/misinfo_utils.py`, name it exactly
   `misinfo_utils`, choose **File → Set as Utility Script**, then **Save Version**. After any change to the
   file, paste it again and save a new version, or Kaggle keeps the old code.
2. **Each notebook:** File → Import Notebook; **File → Add utility script → misinfo_utils**; **Add Input**
   as below; then **Save Version → Save & Run All**.

| Notebook | Add as inputs | Accelerator |
|---|---|---|
| 01a–01d | the raw dataset it needs | none |
| 02_split | notebooks 01a, 01b, 01c, 01d | none |
| 03_eda | 02_split | none |
| 04_features | 02_split | none |
| 05a–05d, 05f, 05g | 02_split, 04_features | none |
| 05e_train_distilbert | the 12 classification split CSVs (as a Kaggle Dataset or the 02_split notebook); **Internet on** | **GPU T4** |
| 06_ensemble | 02_split, all of 05a–05g | none |
| 07_evaluation | 02_split, 04_features, all of 05a–05g, 06_ensemble | GPU optional |

The notebooks detect Kaggle automatically: they read from `/kaggle/input` and write to `/kaggle/working`.

### Training DistilBERT on Kaggle

1. Upload the 12 classification split files from `outputs/02_split/` (all `<dataset>_train/val/test.csv`,
   not `liar_reg_*`) as one Kaggle Dataset.
2. Create the `misinfo_utils` Utility Script (step 1 above).
3. Import `05e_train_distilbert.ipynb`, then add the utility script and the dataset.
4. In **Session options**, set Accelerator to **GPU T4** and Internet to **On** (the GPU needs a
   phone-verified Kaggle account). Check that the log prints `Device: cuda`; if it prints `cpu`, cancel and
   fix the setting.
5. **Save Version → Save & Run All**. The last cell (Kaggle only) packs the results into one zip.
6. Download the zip from the version's **Output** tab and unzip it. For each dataset, copy the
   `distilbert__<dataset>/` folder and `pred__distilbert__<dataset>.csv` into
   `outputs/05_models/classification/<dataset>/` locally.

---

## What each stage produces

| Stage | Folder | Output files |
|---|---|---|
| 1 Preprocess | `outputs/01_preprocess/` | `liar_cleaned.csv`, `welfake_cleaned.csv`, `covid_cleaned.csv`, `fakenewsnet_cleaned.csv` |
| 2 Split | `outputs/02_split/` | `<dataset>_train/val/test.csv` × 4, `liar_reg_train/val/test.csv` |
| 3 EDA | `outputs/03_eda/` | none (tables and plots stay in the notebook) |
| 4 Features | `outputs/04_features/<key>_preprocessed/` | `features__<key>.joblib`, `X_tfidf__<key>__<split>.npz`, `X_dense__<key>__<split>.npy` (key = each dataset and `liar_reg`) |
| 5 Classification | `outputs/05_models/classification/<dataset>/` | `<model>__<dataset>.joblib` × 4, `distilbert__<dataset>/`, `pred__<model>__<dataset>.csv` × 5 |
| 5 Regression | `outputs/05_models/regression/liar/` | `dummy_reg__liar.joblib`, `ridge_reg__liar.joblib`, `lightgbm_reg__liar.joblib`, `pred__<model>__liar.csv` × 3 |
| 5 Clustering | `outputs/05_models/clustering/<dataset>/` | `kmeans__<dataset>.joblib`, `gmm__<dataset>.joblib`, `clusters__<dataset>.csv` (+ `clustering/clustering_k.json`) |
| 6 Ensemble | `outputs/06_ensemble/` | `final_model.json`, `blender.joblib`, `pred__final.csv`, `reg_final.json`, `pred__average_reg__liar.csv` |
| 7 Evaluation | `outputs/07_evaluation/` | `results_in_dataset.csv`, `results_cross_dataset.csv`, `results_final.csv`, `results_regression.csv`, … |

Checkpoints: 20 classification (Logistic Regression, Complement NB, Linear SVM, LightGBM, DistilBERT × 4
datasets), 3 regression (Dummy, Ridge, LightGBM regressor on LIAR) and 8 clustering (K-Means and Gaussian
Mixture × 4 datasets).

`outputs/04_features/` is not in the repository (large; rebuilt by Stage 4).

Later stages find files **by name** anywhere under `outputs/`. If two files share a name (for example an old
copy left directly in `05_models/`), the notebook prints a warning and uses the first one, so delete old
copies.

---

## 4. Predict new text

The trained models are included, so prediction works straight after cloning (no training needed). Run
this in the `misinfo` environment, for example in a new notebook in `notebooks/`:

```python
import torch                        # keep this first (see the Windows note)
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../src").resolve()))   # path to src/ from where you run this

import misinfo_utils as mu
mu.setup("07")                      # finds the saved files
model = mu.FinalModel()             # loads the 4 DistilBERT experts + the blender

texts = ["Drinking hot lemon water kills the coronavirus."]
print(model.predict(texts))         # 1 = misleading, 0 = reliable
print(model.predict_proba(texts))   # P(misleading) for each text
print(model.order)                  # ['liar', 'welfake', 'covid', 'fakenewsnet']
print(model.expert_probs(texts))    # each expert's P(misleading), same order
```

- Loading takes a while on CPU (four DistilBERT models). Load once, then predict many texts; prediction
  takes about 231 ms per text on a CPU.
- **Use the label, not the probability.** The blender's probabilities stay in a narrow range (about
  0.41–0.55), so the website shows the label.
- The model judges wording and style, not facts, so a label is a prompt to check further, not a verdict.

---

## Evidence check (extra feature)

The website also offers an optional evidence check that the user runs on request, after seeing the label.
It is a separate system from this pipeline: a guardrail screens the input, then the system searches a
ChromaDB store of four fact-checking guides (all-MiniLM-L6-v2 embeddings) and, if needed, DuckDuckGo, and a
Qwen-2.5-1.5B model (LoRA, 4-bit) writes a short judgement from the retrieved evidence. It is not trained
or evaluated by the notebooks in this repository.

---

## Results (test data)

| Item | Result |
|---|---|
| DistilBERT experts, each on its own dataset (macro F1) | LIAR 0.635, WELFake 0.980, COVID-19 0.958, FakeNewsNet 0.788 (mean 0.840) |
| Final ensemble, all 7,622 test texts | accuracy 0.764, precision 0.783, recall 0.709, macro F1 0.762, ROC-AUC 0.820 |
| Final ensemble vs best single expert across all four datasets | 0.689 vs 0.605 mean macro F1 |
| Ridge regression, LIAR score | MAE 1.254 (mean baseline 1.347), R² 0.086 |
| Clustering | K-Means beats Gaussian Mixture on silhouette and Davies-Bouldin for every dataset |

Full results, figures and discussion are in the group report.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Could not find ...` | The stage that makes that file has not run (run the stages in order), or on Kaggle the input notebook or dataset is not attached |
| `output_path() got an unexpected keyword argument 'subfolder'` | Old `misinfo_utils`: restart the kernel; on Kaggle, save a new version of the utility script |
| `WinError 127 ... shm.dll` | `import torch` as the first line, then restart the kernel |
| `torch can only be initialized once per process` | An earlier `import torch` failed: restart the kernel and read the first error |
| Model files are a few hundred bytes | Git LFS was not installed before cloning: `git lfs install`, then `git lfs pull` |
| Jupyter cannot find the `misinfo` environment | Run the `ipykernel install` line above, then choose the **misinfo** kernel |
| Jupyter opens in `C:\` and cannot see the project | Start it from the project folder: `cd /d D:\path\to\Misinformation-Detection-Model` then `jupyter lab` |

---

## References

- Patwa, P. et al. (2021) 'Fighting an infodemic: COVID-19 fake news dataset', *CONSTRAINT 2021*, Springer, pp. 21–29.
- Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019) 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter', arXiv:1910.01108.
- Shu, K., Mahudeswaran, D., Wang, S., Lee, D. and Liu, H. (2020) 'FakeNewsNet: a data repository with news content, social context, and spatiotemporal information for studying fake news on social media', *Big Data*, 8(3), pp. 171–188.
- Verma, P.K., Agrawal, P., Amorim, I. and Prodan, R. (2021) 'WELFake: word embedding over linguistic features for fake news detection', *IEEE Transactions on Computational Social Systems*, 8(4), pp. 881–893.
- Wang, W.Y. (2017) '"Liar, liar pants on fire": a new benchmark dataset for fake news detection', *Proceedings of ACL 2017 (Volume 2: Short Papers)*, pp. 422–426.
