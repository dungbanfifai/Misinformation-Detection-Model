# Processed Datasets: Misinformation Detection

COS30049 Computing Technology Innovation Project, Assignment 2 (AI4Cyber: misinformation detection on social media).

This folder contains the processed data used to train and test our final models: a classifier that labels a
text as **reliable** or **misleading**, a regressor that predicts a LIAR-style **truthfulness score**
(0–5), and **topic clusters**. The data comes from four public datasets, cleaned and split by our pipeline
(Stages 1 and 2; see the project `README.md`).

## Contents

```
data/
├── liar_train.csv           liar_val.csv           liar_test.csv
├── welfake_train.csv        welfake_val.csv        welfake_test.csv
├── covid_train.csv          covid_val.csv          covid_test.csv
├── fakenewsnet_train.csv    fakenewsnet_val.csv    fakenewsnet_test.csv
├── liar_reg_train.csv       liar_reg_val.csv       liar_reg_test.csv
└── DATASET_README.md        this file
```

- **Classification files** (`<dataset>_train/val/test.csv`, 12 files): one set per dataset, used by all
  classifiers, the final ensemble and clustering.
- **Regression files** (`liar_reg_train/val/test.csv`, 3 files): LIAR with all six ratings, used to predict
  the 0–5 truthfulness score.

All files are UTF-8 CSV with a header row.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `row_id` | text | unique ID of the text, e.g. `liar_000123` (prefix = source dataset) |
| `text` | text | the cleaned text (see "How the data was processed") |
| `label` | 0 / 1 | **0 = reliable, 1 = misleading**; empty for LIAR half-true claims (regression files only) |
| `score` | 0–5 | LIAR only: 0 pants-fire, 1 false, 2 barely-true, 3 half-true, 4 mostly-true, 5 true; empty for other datasets |
| `source_dataset` | text | `liar`, `welfake`, `covid` or `fakenewsnet` |
| `sub_source` | text | for FakeNewsNet, the fact-checking site (`politifact` or `gossipcop`); otherwise the dataset name |
| `n_words` | integer | number of words in the cleaned text |
| `had_url` | true / false | whether the text contained a web link **before** cleaning (used for the shortcut check) |

The task uses `text` as input and `label` (classification) or `score` (regression) as the target. The other
columns are for analysis only.

## Size and balance

| Dataset | Kind of text | Train | Validation | Test | Total | % misleading | Median words |
|---|---|---|---|---|---|---|---|
| LIAR (classification) | short US political claims | 7,114 | 1,524 | 1,525 | 10,163 | 55.5 | 16 |
| WELFake | full news articles | 14,000 | 2,999 | 3,001 | 20,000 | 45.5 | 405 |
| COVID-19 | social-media posts | 7,195 | 1,542 | 1,543 | 10,280 | 46.7 | 24 |
| FakeNewsNet | news headlines | 7,243 | 1,552 | 1,553 | 10,348 | 49.4 | 11 |
| **Total (classification)** | | **35,552** | **7,617** | **7,622** | **50,791** | | |
| LIAR (regression) | short US political claims | 8,953 | 1,918 | 1,920 | 12,791 | | 16 |

LIAR ratings in the regression files: pants-fire 1,050, false 2,484, barely-true 2,107, half-true 2,628,
mostly-true 2,464, true 2,058 (mean score 2.72, standard deviation 1.56).

## Sources

| Dataset | Downloaded from | Original labels | How we mapped them |
|---|---|---|---|
| LIAR (Wang, 2017) | [Kaggle: doanquanvietnamca/liar-dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset) | six PolitiFact ratings | pants-fire, false, barely-true → 1; mostly-true, true → 0; half-true left out of classification. All six kept as `score` 0–5 |
| WELFake (Verma et al., 2021) | [Kaggle: saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) | 0 / 1 | kept as given. The documentation says 1 = real, but the data shows 0 = real (Reuters is named in 61.8% of label-0 and 1.8% of label-1 articles), so 0 = reliable, 1 = misleading |
| COVID-19 Constraint (Patwa et al., 2021) | [Kaggle: elvinagammed/covid19-fake-news-dataset-nlp](https://www.kaggle.com/datasets/elvinagammed/covid19-fake-news-dataset-nlp) | real / fake | real → 0, fake → 1. Only the labelled files were used (train, validation and `english_test_with_labels.csv`) |
| FakeNewsNet (Shu et al., 2020) | [GitHub: KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) | separate fake and real files | real → 0, fake → 1. Headlines only (the files have no article body) |

## How the data was processed

**Stage 1: cleaning (each dataset on its own)**

1. Loaded the raw files and built one text per row (WELFake: title + space + body).
2. Decoded HTML entities (e.g. `&amp;` → `&`), and removed web links, HTML tags, hashtags, @mentions and
   source names that give the label away (e.g. "Reuters" in WELFake).
3. Kept letter case and punctuation, because capitals and exclamation marks are signs of sensational writing.
   Stop words were not removed.
4. Removed rows with a missing label or empty text, texts that were too short (fewer than 5 words in WELFake
   and COVID-19, fewer than 3 in LIAR and FakeNewsNet), texts that appeared with both labels, and exact
   duplicates.

| Rows | LIAR | WELFake | COVID-19 | FakeNewsNet |
|---|---|---|---|---|
| Loaded | 12,836 | 72,134 | 10,700 | 23,196 |
| After cleaning | 12,791 | 63,313 | 10,280 | 21,221 |
| Removed | 45 (0.4%) | 8,821 (12.2%) | 420 (3.9%) | 1,975 (8.5%) |

**Stage 2: joining, balancing and splitting**

1. Searched for texts shared between datasets: 2 were found (both with the same label) and removed from the
   lower-priority dataset (priority: LIAR, COVID-19, FakeNewsNet, WELFake). The datasets are otherwise
   kept separate, because each trains its own expert model.
2. WELFake was randomly reduced from 63,312 to 20,000 articles, keeping its class ratio.
3. GossipCop (part of FakeNewsNet) was balanced by randomly reducing real headlines to 4,700, the number of
   fake ones.
4. Each dataset was split 70 / 15 / 15 into train, validation and test, stratified by label (FakeNewsNet by
   site and label), with random seed 42. LIAR was split once, stratified by score; the classification files
   are the regression files without half-true claims, so every claim is in the same split for both tasks.

**How the splits are used:** models are trained on `train`, every model choice is made on `val`, and `test`
is used once, for the final results.

## Known limitations

- **WELFake** still contains some publisher traces (e.g. "filessupport" from 21st Century Wire, "corrects"
  from Reuters correction notices), so scores on it are likely optimistic.
- **COVID-19** reliable posts are mostly official daily updates and are longer than misleading posts (median
  about 30 words against 17), which a model can partly use as a shortcut.
- **FakeNewsNet** has headlines only, about 11 words each.
- **LIAR** claims starting with "Says" are slightly more often misleading; this is how PolitiFact writes
  claims, so it was kept.
- The texts are in English and mostly from 2007–2021 (US politics, COVID-19 and celebrity news), so newer
  topics may be handled less well.

## Loading the data

```python
import pandas as pd

train = pd.read_csv("data/covid_train.csv")
X, y = train["text"], train["label"]                 # classification

reg = pd.read_csv("data/liar_reg_train.csv")
X_reg, y_reg = reg["text"], reg["score"]             # regression (0–5)
```

## References

Patwa, P., Sharma, S., Pykl, S., Guptha, V., Kumari, G., Akhtar, M.S., Ekbal, A., Das, A. and Chakraborty, T.
(2021) 'Fighting an infodemic: COVID-19 fake news dataset', in *Combating Online Hostile Posts in Regional
Languages during Emergency Situation (CONSTRAINT 2021)*, Springer, Cham, pp. 21–29.

Shu, K., Mahudeswaran, D., Wang, S., Lee, D. and Liu, H. (2020) 'FakeNewsNet: a data repository with news
content, social context, and spatiotemporal information for studying fake news on social media', *Big Data*,
8(3), pp. 171–188.

Verma, P.K., Agrawal, P., Amorim, I. and Prodan, R. (2021) 'WELFake: word embedding over linguistic features
for fake news detection', *IEEE Transactions on Computational Social Systems*, 8(4), pp. 881–893.

Wang, W.Y. (2017) '"Liar, liar pants on fire": a new benchmark dataset for fake news detection',
*Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short
Papers)*, Vancouver, pp. 422–426.
