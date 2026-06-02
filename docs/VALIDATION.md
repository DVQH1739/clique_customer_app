# Validation checklist

Automated checks run at the end of `retail.run_from_xlsx()` (`pipelines/validate.py`).

## Data & features

- [x] `data/raw/online_retail_ii.xlsx` exists locally
- [x] All 8 `FEATURE_NAMES` present; no NaN/inf in profiles
- [x] `frequency >= MIN_FREQUENCY` (2) for every customer

## Preprocessing

- [x] `X_train_scaled.csv` and `X_test_scaled.csv` exist
- [x] Scaled values in [0, 1]
- [x] `random_state=42` for reproducible split

## CLIQUE algorithm

- [x] Apriori dense units: threshold `tau × n_samples`
- [x] Grid `xi ∈ {8,10,12}`, `tau ∈ {0.10…0.18}`
- [x] Selection: positive silhouette when possible, else composite `quality_score`
- [x] At least 2 non-noise clusters after fit

## Metrics

- [x] Intrinsic: silhouette, Davies–Bouldin, Calinski–Harabasz (non-noise points only)
- [x] `baseline_comparison.csv` CLIQUE row matches recomputed train silhouette (±0.01)
- [x] Test set: coverage, silhouette on assigned points

## Artifacts

- [x] 5 metrics CSVs non-empty
- [x] EDA, baseline comparison, subspace heatmap, cluster sizes PNGs present

Manual review: open `docs/RESULTS.md` and `results/figures/` after each full run.
