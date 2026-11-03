"""
misinfo_utils.py — shared code for the misinformation detection pipeline.


Used by every notebook, on Kaggle and locally:
  * Kaggle: upload this file as a Utility Script named "misinfo_utils" and attach it
    to each notebook (File -> Add utility script). Then `import misinfo_utils` works.
  * Local:  keep it in  <project>/src/  . Each notebook adds that folder to sys.path.

Contents
  1. Paths      setup(), find_file()           -> same code works on Kaggle and locally
  2. Constants  label conventions, LIAR score map
  3. Cleaning   clean_text(), dedup_key()
  4. Stage 1    make_row_ids(), clean_and_deduplicate(), save_clean_dataset()
  5. Stage 2+   read_csv(), save_csv(), split_three_ways(), file-name helpers
  6. Plots      plot_label_counts()
  7. Stage 4    style_features(), FeatureBuilder, feature file helpers
  8. Stage 5    model names, train_classic_for_all(), prediction files, DistilBERT helpers
  9. Metrics    clf_scores(), reg_scores(), mean_dataset_f1()
 10. Final      FinalModel: loads the chosen experts + combiner and predicts any text
"""

import os
import re
import html
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# =============================================================================
# 1. PATHS
# =============================================================================

# Each stage writes into its own folder locally (on Kaggle everything goes to /kaggle/working).
STAGE_FOLDERS = {
    "01": "01_preprocess", "02": "02_split", "03": "03_eda", "04": "04_features",
    "05": "05_models", "06": "06_ensemble", "07": "07_evaluation",
}

PATHS = {}   # filled by setup(); used by find_file()


def is_kaggle():
    """True when running inside a Kaggle notebook (MISINFO_ENV can force either mode for testing)."""
    forced = os.environ.get("MISINFO_ENV", "").lower()
    if forced in ("kaggle", "local"):
        return forced == "kaggle"
    return os.path.isdir("/kaggle/input")


def find_project_dir():
    """Local project folder = the folder that contains src/misinfo_utils.py."""
    if os.environ.get("MISINFO_PROJECT_DIR"):
        return Path(os.environ["MISINFO_PROJECT_DIR"]).resolve()
    return Path(__file__).resolve().parent.parent


def setup(stage, seed=SEED):
    """
    Call once at the top of every notebook, e.g. paths = setup("01a").
    Returns a dict with:
      env        "kaggle" or "local"
      read_dirs  folders searched (recursively) for input files
      write_dir  where this notebook saves its outputs
      fig_dir    write_dir/figures (created only when a figure is saved)
    """
    stage_key = str(stage)[:2]
    if stage_key not in STAGE_FOLDERS:
        raise ValueError(f"Unknown stage {stage!r}; expected one starting with {list(STAGE_FOLDERS)}")

    if is_kaggle():
        env = "kaggle"
        read_dirs = [Path(os.environ.get("MISINFO_INPUT_DIR", "/kaggle/input"))]
        write_dir = Path(os.environ.get("MISINFO_OUTPUT_DIR", "/kaggle/working"))
    else:
        env = "local"
        project = find_project_dir()
        read_dirs = [project / "data" / "raw", project / "outputs"]
        write_dir = project / "outputs" / STAGE_FOLDERS[stage_key]

    fig_dir = write_dir / "figures"
    write_dir.mkdir(parents=True, exist_ok=True)

    PATHS.clear()
    PATHS.update({"env": env, "stage": str(stage), "read_dirs": read_dirs,
                  "write_dir": write_dir, "fig_dir": fig_dir})
    set_seed(seed)

    print(f"misinfo_utils | stage {stage} | running on {env}")
    print("  read from :", ", ".join(str(d) for d in read_dirs))
    print("  write to  :", write_dir)
    return PATHS


def set_seed(seed=SEED):
    """Same random numbers on every run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def list_input_files():
    """Every file under the read folders (useful to check inputs are attached)."""
    if not PATHS:
        raise RuntimeError("Call setup(...) first.")
    found = []
    for d in PATHS["read_dirs"]:
        if d.exists():
            found += [p for p in d.rglob("*") if p.is_file()]
    return sorted(found)


def find_file(*names, required=True):
    """
    Find an input file by NAME anywhere under the read folders, e.g. find_file("train.tsv").
    Several names = alternatives. Warns if the same name exists in more than one place,
    because then it is ambiguous which one is used.
    """
    wanted = {n.lower() for n in names}
    matches = [p for p in list_input_files() if p.name.lower() in wanted]
    if not matches:
        if required:
            raise FileNotFoundError(
                f"Could not find {names} under {[str(d) for d in PATHS['read_dirs']]}. "
                "On Kaggle: is the dataset/notebook added as an input? "
                "Locally: is the file in data/raw/ (or produced by an earlier stage)?")
        return None
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} files named {matches[0].name}; using {matches[0]}")
    return matches[0]


def find_dir(name, required=True):
    """Like find_file, but for a FOLDER (e.g. a saved DistilBERT model)."""
    matches = []
    for d in PATHS["read_dirs"]:
        if d.exists():
            matches += [p for p in d.rglob(name) if p.is_dir()]
    if not matches:
        if required:
            raise FileNotFoundError(f"Could not find folder {name!r} under the read folders.")
        return None
    return sorted(matches)[0]


def output_path(filename, figure=False, subfolder=None):
    """Full path for a file this notebook writes (optionally inside a subfolder)."""
    folder = PATHS["fig_dir"] if figure else PATHS["write_dir"]
    if subfolder:
        folder = folder / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


# =============================================================================
# 2. CONSTANTS
# =============================================================================

LABEL_NAMES = {0: "reliable", 1: "misleading"}      # same convention in every dataset
DATASETS = ["liar", "welfake", "covid", "fakenewsnet"]
SPLITS = ["train", "val", "test"]

# LIAR six ratings -> 0-5 truthfulness score (regression target)
LIAR_SCORE = {"pants-fire": 0, "false": 1, "barely-true": 2,
              "half-true": 3, "mostly-true": 4, "true": 5}
SCORE_NAMES = {v: k for k, v in LIAR_SCORE.items()}
LIAR_HALF_TRUE_SCORE = 3   # kept for regression, excluded from classification

# Columns every <dataset>_cleaned.csv has, in this order (same schema for all four datasets)
CLEAN_COLUMNS = ["row_id", "text", "label", "score", "source_dataset", "sub_source",
                 "n_words", "had_url"]


# =============================================================================
# 3. TEXT CLEANING
# =============================================================================

URL_RE = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+|\b\S+\.(?:com|org|net|co)/\S*)", re.I)
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")

# Phrases that reveal WHO published a text rather than WHAT it says.
# Removing them stops the models learning "Reuters = reliable" etc.
SOURCE_GIVEAWAYS = [
    r"^[^\n]{0,120}?\(Reuters\)\s*[-–—]\s*",    # "WASHINGTON (Reuters) - "
    r"\(Reuters\)",
    r"Reuters",
    r"\b21st Century Wire\b",
    r"\bFeatured image via\b[^\n.]*",
    r"\b(?:Photo|Image) (?:by|via|credit:?)\b[^\n.]*",
    r"\bGetty Images\b",
    r"tmsnrt\.rs\S*",          # Reuters short links, e.g. tmsnrt.rs/2abc
    r"21WIRE",                 # 21st Century Wire's short name
    r"Wikimedia Commons",      # image credit
    r"HollywoodLife",          # gossip site name
]
SOURCE_RE = [re.compile(p, re.I) for p in SOURCE_GIVEAWAYS]


def clean_text(text):
    """Remove HTML, links, hashtags, @mentions and source giveaways; tidy spaces.
    Case and stopwords are kept: TF-IDF lowercases itself and BERT needs the original text."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)            # &amp; -> &
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    for pattern in SOURCE_RE:
        text = pattern.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def dedup_key(text):
    """Lowercase letters and digits only, so near-identical copies count as duplicates."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def has_url(series):
    """True for texts containing a link (check BEFORE cleaning, which removes links)."""
    return series.astype(str).str.contains(r"https?://|www\.", regex=True)


# =============================================================================
# 4. PREPROCESSING STEPS SHARED BY THE FOUR DATASETS
# =============================================================================

def make_row_ids(n, dataset):
    """Stable IDs from the row's position in the RAW data, e.g. liar_000123."""
    return [f"{dataset}_{i:06d}" for i in range(n)]


def clean_and_deduplicate(df, min_words, target_col="label"):
    """
    Shared cleaning for one dataset. `df` needs: row_id, text, <target_col>.
    Steps (rows counted after each, for the report):
      1. drop rows with no target           (e.g. a rating not in the label map)
      2. clean the text, drop empty texts
      3. drop texts shorter than min_words
      4. drop texts that appear with DIFFERENT targets (label noise: all copies removed)
      5. drop exact/near duplicates (keep the first)
    target_col = "label" for most datasets, "score" for LIAR (so statements rated
    differently in different copies are removed).
    Returns (clean_df, log_dict).
    """
    log = {"loaded": len(df)}
    df = df.copy()

    df = df.dropna(subset=[target_col])
    log["after_missing_target"] = len(df)

    df["had_url"] = has_url(df["text"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    log["after_empty_text"] = len(df)

    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"] >= min_words]
    log["after_min_length"] = len(df)

    df["key"] = df["text"].map(dedup_key)
    n_targets = df.groupby("key")[target_col].nunique()
    conflicting = set(n_targets[n_targets > 1].index)
    df = df[~df["key"].isin(conflicting)]
    log["after_conflict_removal"] = len(df)

    df = df.drop_duplicates("key", keep="first").drop(columns="key")
    log["after_dedup"] = len(df)
    return df, log


def to_clean_schema(df, dataset, sub_source=None):
    """Put the columns in the shared order; fill the ones a dataset does not have."""
    out = df.copy()
    out["source_dataset"] = dataset
    if "sub_source" not in out.columns:
        out["sub_source"] = sub_source or dataset
    if "score" not in out.columns:
        out["score"] = pd.NA
    out["label"] = out["label"].astype("Int64")      # whole numbers; empty allowed (LIAR half-true)
    out["score"] = out["score"].astype("Int64")
    return out[CLEAN_COLUMNS].reset_index(drop=True)


def save_clean_dataset(df, dataset):
    """Write <dataset>_cleaned.csv (the only file Stage 1 saves); check the schema first."""
    missing = [c for c in CLEAN_COLUMNS if c not in df.columns]
    assert not missing, f"missing columns: {missing}"
    assert df["row_id"].is_unique, "row_id must be unique"
    assert df["text"].str.len().gt(0).all(), "empty texts left"
    labels = set(df["label"].dropna().unique())
    assert labels <= {0, 1}, f"labels must be 0/1, found {labels}"

    data_path = output_path(f"{dataset}_cleaned.csv")
    df.to_csv(data_path, index=False)
    print(f"Saved {len(df):,} rows -> {data_path}")
    return data_path


# =============================================================================
# 5. FILES AND SPLITS (Stage 2 onwards)
# =============================================================================

def cleaned_file(dataset):
    """Stage 1 output name, e.g. liar_cleaned.csv"""
    return f"{dataset}_cleaned.csv"


def split_file(dataset, split):
    """Stage 2 classification split, e.g. covid_train.csv"""
    return f"{dataset}_{split}.csv"


def reg_split_file(split):
    """Stage 2 regression split (LIAR, all six ratings), e.g. liar_reg_test.csv"""
    return f"liar_reg_{split}.csv"


def read_csv(filename):
    """Read an input CSV found by name; keeps label/score as whole numbers with empty allowed."""
    df = pd.read_csv(find_file(filename))
    for col in ("label", "score"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def save_csv(df, filename):
    """Write a CSV into this notebook's output folder."""
    path = output_path(filename)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):>7,} rows -> {path.name}")
    return path


def split_three_ways(df, strata, train=0.70, val=0.15, seed=SEED):
    """
    Stratified train / val / test split (default 70 / 15 / 15).
    `strata` = a column (Series) whose proportions are kept in all three parts.
    Returns a dict {"train": df, "val": df, "test": df}.
    """
    from sklearn.model_selection import train_test_split
    test = 1 - train - val
    train_df, rest = train_test_split(df, train_size=train, stratify=strata, random_state=seed)
    val_df, test_df = train_test_split(rest, test_size=test / (val + test),
                                       stratify=strata.loc[rest.index], random_state=seed)
    return {"train": train_df, "val": val_df, "test": test_df}


# =============================================================================
# 6. PLOTS
# =============================================================================

def plot_label_counts(frames, title, filename=None):
    """Side-by-side count plots, e.g. {"raw": raw_df, "clean": clean_df}. Rows without a label are skipped.
    Shown in the notebook; also saved as a figure only if `filename` is given."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 4), squeeze=False)
    for ax, (name, data) in zip(axes[0], frames.items()):
        labelled = data.dropna(subset=["label"]).copy()
        labelled["label_name"] = labelled["label"].astype(int).map(LABEL_NAMES)
        graph = sns.countplot(data=labelled, x="label_name", order=["reliable", "misleading"],
                              hue="label_name", palette={"reliable": "#4C9F70", "misleading": "#D1495B"},
                              legend=False, ax=ax)
        for container in graph.containers:
            graph.bar_label(container)
        graph.set_title(name)
        graph.set_xlabel("")
        for side in ("top", "right", "left"):
            graph.spines[side].set_visible(False)
    fig.suptitle(title)
    plt.tight_layout()
    if filename:
        path = output_path(filename, figure=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print("Saved figure:", path)
    plt.show()


# =============================================================================
# 7. FEATURES (Stage 4)
# =============================================================================

STYLE_FEATURE_NAMES = ["log_words", "log_chars", "exclamations", "questions", "caps_ratio",
                       "digit_ratio", "allcaps_word_ratio", "quote_ratio"]
FEATURE_KEYS = DATASETS + ["liar_reg"]      # one feature builder per dataset + one for LIAR regression


def style_features(texts):
    """8 numbers per text describing its writing style (not its words)."""
    rows = []
    for t in texts:
        n_chars = max(len(t), 1)
        words = t.split()
        letters = [c for c in t if c.isalpha()]
        rows.append([
            np.log1p(len(words)),                                                 # length in words
            np.log1p(n_chars),                                                    # length in characters
            t.count("!"),
            t.count("?"),
            sum(c.isupper() for c in letters) / max(len(letters), 1),             # capital-letter ratio
            sum(c.isdigit() for c in t) / n_chars,                                # digit ratio
            sum(w.isupper() and len(w) > 1 for w in words) / max(len(words), 1),  # ALL-CAPS words
            t.count('"') / n_chars,                                               # quotation marks
        ])
    return np.array(rows, dtype=float)


class FeatureBuilder:
    """
    Turns text into vectors. Fitted on ONE training part only, then used for any text.
      sparse(texts) -> TF-IDF vectors (one column per word / word pair)      for LogReg, NB, SVM, Ridge
      dense(texts)  -> 300 SVD numbers + 8 style numbers per text            for LightGBM, clustering
    """
    def __init__(self, svd_dims=300, max_features=200_000, seed=SEED):
        self.svd_dims, self.max_features, self.seed = svd_dims, max_features, seed

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler
        texts = list(texts)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                                     max_features=self.max_features)
        X = self.tfidf.fit_transform(texts)
        dims = max(2, min(self.svd_dims, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dims, random_state=self.seed).fit(X)
        self.scaler = StandardScaler().fit(style_features(texts))
        return self

    @property
    def n_svd(self):
        return self.svd.n_components

    def sparse(self, texts):
        return self.tfidf.transform(list(texts))

    def dense(self, texts, X_sparse=None):
        texts = list(texts)
        X_sparse = self.sparse(texts) if X_sparse is None else X_sparse
        return np.hstack([self.svd.transform(X_sparse), self.scaler.transform(style_features(texts))])


def builder_file(key):
    return f"features__{key}.joblib"


def tfidf_file(key, split):
    return f"X_tfidf__{key}__{split}.npz"


def dense_file(key, split):
    return f"X_dense__{key}__{split}.npy"


def load_builder(key):
    import joblib
    return joblib.load(find_file(builder_file(key)))


def load_tfidf(key, split):
    from scipy import sparse
    return sparse.load_npz(find_file(tfidf_file(key, split)))


def load_dense(key, split):
    return np.load(find_file(dense_file(key, split)))


def load_split(key, split):
    """Stage 2 rows for a dataset (or 'liar_reg') and split, in the same order as the feature matrices."""
    return read_csv(reg_split_file(split) if key == "liar_reg" else split_file(key, split))


# =============================================================================
# 8. MODELS AND PREDICTIONS (Stage 5)
# =============================================================================

CLF_MODELS = ["logreg", "complementnb", "linearsvm", "lightgbm", "distilbert"]
REG_MODELS = ["dummy_reg", "ridge_reg", "lightgbm_reg"]
MODEL_NAMES = {"logreg": "Logistic Regression", "complementnb": "Complement NB",
               "linearsvm": "Linear SVM", "lightgbm": "LightGBM", "distilbert": "DistilBERT",
               "dummy_reg": "Dummy (mean)", "ridge_reg": "Ridge", "lightgbm_reg": "LightGBM regressor",
               "average_reg": "Average (Ridge + LightGBM)"}
SPARSE_MODELS = {"logreg", "complementnb", "linearsvm", "dummy_reg", "ridge_reg"}   # use TF-IDF; others use dense


def checkpoint_file(model, trained_on):
    """e.g. linearsvm__covid.joblib"""
    return f"{model}__{trained_on}.joblib"


def bert_dir_name(trained_on):
    return f"distilbert__{trained_on}"


def pred_file(model, trained_on):
    """One file per checkpoint with its predictions on every dataset's val and test parts."""
    return f"pred__{model}__{trained_on}.csv"


def features_for(model, builder, texts):
    """The vectors a model expects: TF-IDF for SPARSE_MODELS, dense SVD + style for the rest."""
    return builder.sparse(texts) if model in SPARSE_MODELS else builder.dense(texts)


def prob_misleading(model, fitted, builder, texts):
    """P(misleading) for each text from a fitted classic model."""
    return fitted.predict_proba(features_for(model, builder, texts))[:, 1]


def train_classic_for_all(model, make_model, tune=None, datasets=DATASETS):
    """
    Stage 5 loop for one classic model type:
      for each dataset: fit on its training vectors (from Stage 4), save the checkpoint,
      then predict the val + test parts of ALL four datasets with that dataset's feature builder.
    make_model(params) -> an unfitted model.  tune = (estimator, param_grid) or None.
    Returns (val_f1 table: rows = trained on, columns = validated on, chosen params).
    """
    import time, joblib
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import f1_score

    texts = {(ds, s): load_split(ds, s) for ds in datasets for s in ("val", "test")}
    table, chosen = {}, {}
    for ds in datasets:
        t0 = time.time()
        builder = load_builder(ds)
        y_train = load_split(ds, "train")["label"].astype(int).values
        X_train = load_tfidf(ds, "train") if model in SPARSE_MODELS else load_dense(ds, "train")

        params = {}
        if tune is not None:
            estimator, grid = tune
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=cv, n_jobs=-1).fit(X_train, y_train)
            params = search.best_params_
        fitted = make_model(params).fit(X_train, y_train)
        joblib.dump(fitted, output_path(checkpoint_file(model, ds), subfolder=f"classification/{ds}"))
        chosen[ds] = params

        rows = []
        for (other, split), df in texts.items():
            p = prob_misleading(model, fitted, builder, df["text"])
            rows.append(pd.DataFrame({"row_id": df["row_id"], "tested_on": other, "split": split,
                                      "prob_misleading": p}))
        preds = pd.concat(rows, ignore_index=True)
        preds.to_csv(output_path(pred_file(model, ds), subfolder=f"classification/{ds}"), index=False)

        table[ds] = {}
        for other in datasets:
            mask = (preds["tested_on"] == other) & (preds["split"] == "val")
            table[ds][other] = f1_score(texts[(other, "val")]["label"].astype(int),
                                        (preds.loc[mask, "prob_misleading"] >= 0.5).astype(int), average="macro")
        print(f"{model:12s} trained on {ds:12s} {len(y_train):>6,} rows | params {params} | {time.time() - t0:.0f}s")
    val_f1 = pd.DataFrame(table).T.loc[datasets, datasets].round(4)
    val_f1.index.name, val_f1.columns.name = "trained on", "validated on"
    return val_f1, chosen


def load_predictions(model, trained_on, required=True):
    path = find_file(pred_file(model, trained_on), required=required)
    return None if path is None else pd.read_csv(path)


def bert_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_bert(trained_on):
    """Load a fine-tuned DistilBERT checkpoint folder saved by Stage 5."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    folder = find_dir(bert_dir_name(trained_on))
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).to(bert_device()).eval()
    return tokenizer, model


def bert_prob_misleading(tokenizer, model, texts, max_len=256, batch_size=64):
    """P(misleading) for each text from a DistilBERT checkpoint."""
    import torch
    out = []
    texts = list(texts)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt").to(model.device)
            out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


# =============================================================================
# 9. METRICS
# =============================================================================

def clf_scores(y, p, threshold=0.5):
    """Classification metrics from true labels and P(misleading)."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "macro_F1": f1_score(y, pred, average="macro"),
            "ROC_AUC": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan}


def reg_scores(y, p):
    """Regression metrics for the 0-5 score (predictions clipped to 0-5)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 5)
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p), "within_1_level_%": 100 * float(np.mean(np.abs(y - p) <= 1))}


def mean_dataset_f1(y, p, groups, threshold=0.5):
    """Macro F1 per dataset, then averaged, so every dataset counts equally."""
    from sklearn.metrics import f1_score
    y, p, groups = np.asarray(y).astype(int), np.asarray(p), np.asarray(groups)
    return float(np.mean([f1_score(y[groups == g], (p[groups == g] >= threshold).astype(int), average="macro")
                          for g in np.unique(groups)]))


# =============================================================================
# 10. FINAL MODEL (Stages 6-7, and later the website)
# =============================================================================

FINAL_CONFIG = "final_model.json"


class FinalModel:
    """
    The website's classifier: one expert per dataset (chosen in Stage 6) + a combiner.
        fm = FinalModel(); fm.predict_proba(["some text", ...]) -> P(misleading) per text
    Texts are cleaned with clean_text first, exactly as the training data was.
    """
    def __init__(self, bert_max_len=256):
        import json, joblib
        with open(find_file(FINAL_CONFIG)) as f:
            self.config = json.load(f)
        self.experts = self.config["experts"]            # {dataset: model}
        self.order = self.config["order"]                # dataset order of the combiner's inputs
        self.bert_max_len = bert_max_len
        self.parts = {}
        for ds, model in self.experts.items():
            if model == "distilbert":
                self.parts[ds] = ("bert", load_bert(ds))
            else:
                self.parts[ds] = ("classic", (joblib.load(find_file(checkpoint_file(model, ds))), load_builder(ds)))
        self.blender = joblib.load(find_file(self.config["blender_file"])) if self.config["combiner"] == "blender" else None

    def expert_probs(self, texts):
        """One column per expert (in self.order): each expert's P(misleading)."""
        texts = [clean_text(t) for t in texts]
        cols = []
        for ds in self.order:
            kind, obj = self.parts[ds]
            if kind == "bert":
                cols.append(bert_prob_misleading(*obj, texts, max_len=self.bert_max_len))
            else:
                fitted, builder = obj
                cols.append(prob_misleading(self.experts[ds], fitted, builder, texts))
        return np.column_stack(cols)

    def predict_proba(self, texts):
        P = self.expert_probs(texts)
        combiner = self.config["combiner"]
        if combiner == "blender":
            return self.blender.predict_proba(P)[:, 1]
        if combiner == "weighted":
            w = np.array([self.config["weights"][ds] for ds in self.order])
            return P @ (w / w.sum())
        return P.mean(axis=1)                               # soft vote

    def predict(self, texts, threshold=0.5):
        return (self.predict_proba(texts) >= threshold).astype(int)
