# Coral Bleaching Prediction Research

This repository contains notebook experiments for predicting the percentage of coral bleaching at reef samples. The numbered `start101/` folders record the progression from baseline models to the two-stage model used in the [research manuscript](docs/papers/IUKM2026_CCGK_20260512.pdf). This is a research archive, not a packaged application; `main.py` is an unused project starter.

## Repository map

| Path | Contents |
| --- | --- |
| `start101/implement_01`–`implement_08` | Chronological notebooks, local input snapshots, and experiment plots. `implement_08/2stage_temporal/1pca.ipynb` contains the proposed pipeline. |
| `start101/implement_03/new_data/` | Baselines and PCA experiments on the newer dataset. |
| `research/`, `cache/` | Earlier PyTorch experiments and saved checkpoints on the older dataset; these results use a different target scale. |
| [`docs/papers/`](docs/papers/) | Manuscript, original reference paper, and earlier predictor report. |
| [`docs/review/`](docs/review/) | [Hyperparameters by paper table](docs/review/HYPERPARAMETERS.md) and reviewer responses in English and Thai. |
| [`docs/presentation/`](docs/presentation/) | Slides and speaking script. |
| `archive/maintenance/` | Preserved repository maintenance artifacts. |

The raw older dataset is `PreviousDataset.csv` (`Average_Bleaching` target). The newer raw dataset is `start101/implement_03/global_bleaching_environmental.csv` (`Percent_Bleaching` target). Experiment folders keep their own `best_data.csv` snapshots because notebooks read those files by relative path. Run a notebook from its own directory and keep these inputs beside it.

## Getting started

Use Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
cd start101/implement_03/new_data
uv run python best_pca.py 8
```

The example runs a PCA component sweep with eight components and can take substantial time. For notebooks, select the environment created by `uv sync` as the Jupyter kernel and open the notebook in its experiment directory. The project does not define a build, automated test suite, or lint command. To check a research change, rerun the affected notebook and compare its R², RMSE, and MAE with the manuscript and [hyperparameter record](docs/review/HYPERPARAMETERS.md).

## Paper results

The [hyperparameter record](docs/review/HYPERPARAMETERS.md) maps Tables 1–5 to source notebooks. In particular, `start101/implement_08/2stage_temporal/` holds the proposed model with month features, while `2stage_no_temporal/` holds the comparison without them. Some older-dataset runs were not preserved; the record marks those table cells explicitly.
