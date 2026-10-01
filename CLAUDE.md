# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for a coral-bleaching regression paper (NECTEC internship; submission
`docs/papers/IUKM2026_CCGK_20260512.pdf`). It is a **notebook-driven experiment log**, not an application —
there is no library, no test suite, and `main.py` is an unused `uv init` stub. The deliverables are
the paper tables, `docs/review/HYPERPARAMETERS.md`, and `docs/review/REVIEWER_RESPONSE.md`/`REVIEWER_RESPONSE_TH.md`.

Target variable: percent of coral bleached at a reef sample. Metrics reported everywhere: R², RMSE, MAE.

## Environment / commands

`uv` + Python 3.12 (`.venv/` is ignored by the root `.gitignore`).
The PyTorch CUDA index is pinned in `pyproject.toml` with `index-strategy = "unsafe-best-match"`.

```bash
uv sync                                   # install deps (torch cu121 wheels)
uv run --with jupyterlab jupyter lab       # notebooks are the primary interface
cd start101/implement_03/new_data
uv run python best_pca.py 8               # PCA sweep + LazyPredict + Optuna, arg = n_components
```

No lint config, no tests. "Verification" in this repo means re-running the relevant notebook and
checking the printed K-fold table against the number in the paper / `docs/review/HYPERPARAMETERS.md`.

## Two datasets, two column vocabularies

The single most common source of confusion. Never assume a column name — check the CSV header.

| | Dataset 1 (old) | Dataset 2 (new, current) |
|---|---|---|
| Raw source | `PreviousDataset.csv` (9,665 rows) | `start101/implement_03/global_bleaching_environmental.csv` (41,361 rows) |
| Cleaned rows | 6,112 | 21,928 |
| Target column | `Average_Bleaching` | `Percent_Bleaching` |
| Depth column | `Depth` | `Depth_m` |
| Lives in | `implement_02/`, `implement_03/old_data/`, `start101/cleanData.csv` | `implement_03/new_data/`, `implement_04`–`implement_08` |

**Dataset 2 is the active scope.** Dataset-1 runs are largely not preserved (see the ⚠️ cells in
`docs/review/HYPERPARAMETERS.md`) — don't try to reproduce them, they were re-run in a lost session.

Each experiment folder carries its own cleaned `best_data.csv`, and every notebook loads it by
relative path (`pd.read_csv("../best_data.csv")` or `"best_data.csv"`). Editing one folder's CSV does
not affect the others — that duplication is deliberate, it freezes each experiment's input.

Cleaning that produced `best_data.csv` (from `implement_03/oneHot_16_month_new_data_Ken.ipynb`):
drop identifier/text columns → `df[df['SSTA'] > 0]` → `dropna()` → rename to Dataset-2 names →
3-period rolling means of SSTA/TSA per lat/lon → one-hot `Date_Month` into `Month_1..Month_12`.

## Feature blocks (paper notation)

Notebooks and the paper both use x₁/x₂/x₃; the notebooks spell them out as column lists:

- **x₃** — `paper_features = ['Latitude_Degrees','Longitude_Degrees','Depth_m','ClimSST','SSTA']`, the 5 features from `docs/papers/OriginalPaper.pdf`.
- **x₂** — `month_cols = ['Month_1'…'Month_12']`, one-hot month. Passed through **unscaled**; only x₃ goes through `SimpleImputer(median)` + `StandardScaler`. See `fit_pipeline`/`transform_pipeline`, copied into most notebooks.
- **x₁** — PCA components over the full numeric block (variance thresholds 0.75 / 0.85 / 0.95 depending on experiment; `pca_variance.png` in each folder is that folder's scree plot).

## Experiment folder map (`start101/`)

Numbered folders are chronological; higher number = later thinking. Do not "clean up" or renumber —
`docs/review/HYPERPARAMETERS.md` and the paper cite these paths.

- `implement_01/` — first PCA/baseline exploration, `task.md` is the original checklist.
- `implement_02/` — Dataset-1 baseline (RF), mean baseline, 5-component PCA, MLP.
- `implement_03/{old_data,new_data}/` — same experiments run on both datasets side by side. `new_data/` holds the paper's Table 1 baseline, Table 2 (`of_3Models.ipynb`) and Table 3 x₁+x₂+x₃ (`pca_feature_month.ipynb`).
- `implement_04/` — perturbation-aware late fusion (Table 5).
- `implement_05/` — Table 3 x₁+x₃ (`pca_feature.ipynb`), naive baseline, first classifier+regressor split.
- `implement_06/`, `implement_07/hypoA/` — single-component PCA and lat/lon-PCA hypotheses.
- `implement_08/2stage_{temporal,no_temporal}/` — **the proposed model** (Tables 1 and 4): with vs. without month features.

## The proposed model (two-stage hurdle)

Implemented in `implement_08/2stage_temporal/` (`1pca.ipynb` is the full pipeline; `classif.ipynb`
and `regression.ipynb` are the Optuna tuning runs for each stage).

1. `Status = Percent_Bleaching <= 1` — a boolean gate label, not exactly zero. 
2. Stage A: `RandomForestClassifier(class_weight='balanced')` predicts `Status`.
3. Stage B: ET / RF / HGB regressors trained on the non-gated rows, plus a `VotingRegressor` and a
   **per-row sensitivity-weighted** blend (`compute_sensitivity_weights_per_row`) that perturbs each
   physical feature by `perturbation_deltas` and weights each model by 1/(its response magnitude).
4. Rows the classifier flags stay 0; the rest get the regressor prediction. Scored with
   `StratifiedKFold(5, shuffle=True, random_state=42)` on `Status`.

`RANDOM_STATE = 42` throughout, except `implement_05/pca_feature.ipynb` whose `KFold` uses 67 —
keep it, changing it breaks agreement with the published x₁+x₃ numbers.

Stage A and Stage B fit **separate** imputer/scaler pairs (`imp_A/sc_A`, `imp_B/sc_B`) because they
train on different row subsets. Keep them paired with their model when adding a prediction path.

## Hyperparameters are the artifact

Tuned values are hard-coded as dicts in the notebooks (`rf_cls_param`, `et_param`, `hgb_param`, …)
and mirrored in `docs/review/HYPERPARAMETERS.md`, which maps every paper table cell to its source notebook.
**If you change a hyperparameter in a notebook, update `docs/review/HYPERPARAMETERS.md` in the same edit** —
that file is the reviewer-facing disclosure and a drift between them is a paper defect.
`docs/review/data_prepare.txt` is an earlier scratch copy of the same values.

Notebook comments and markdown are mixed Thai/English; keep the existing language of a cell when editing.

## `research/` and `cache/` — dead PyTorch branch

An earlier, abandoned line of work: a PyTorch MLP / hurdle model on Dataset 1
(`cache/MODEL_INFO.md`, `cache/PROGRESS.md`, `*.pth` checkpoints). It is not the paper's method and
is not referenced by `start101/`. Don't extend it or cite its numbers (RMSE ~7.5 there is on the
Dataset-1 target scale and is not comparable to the paper's ~12).
