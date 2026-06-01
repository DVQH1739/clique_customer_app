# CLIQUE Customer Intelligence

Subspace clustering for customer profiles using the **CLIQUE** algorithm
([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)), with a
reproducible ML pipeline, baseline comparison, and a Streamlit app.

## Project structure

```
clique_customer_app/
├── config.py                 # Central paths, FEATURE_NAMES, RANDOM_STATE=42
├── app.py                    # Streamlit UI (4 pages)
├── clique/                   # Core library
│   ├── algorithm.py          # CLIQUE implementation
│   ├── metrics.py            # Intrinsic + supervised (F1/ROC/AUC) metrics
│   └── utils.py              # Data loading, cleaning, feature engineering, I/O
├── pipelines/                # Reproducible pipeline stages
│   ├── generate_data.py      # 1. Synthetic raw data (with ground-truth segments)
│   ├── preprocess.py         # 2. Log-transform, split, MinMax scale
│   ├── train.py              # 3. Grid search + fit + save model
│   └── evaluate.py           # 4. Metrics (CSV) + figures (PNG)
├── scripts/
│   ├── run_all.py            # Orchestrates the full pipeline
│   ├── verify_clique.py      # Smoke test
│   └── push_to_github.ps1
├── notebooks/training.ipynb  # EDA + orchestration
├── data/
│   ├── raw/                  # Raw inputs (synthetic CSV; real .xlsx ignored by git)
│   └── processed/            # Scaled features + label CSVs
├── models/                   # clique_model.pkl, scaler.pkl, profiles.pkl
└── results/
    ├── metrics/              # CSV: model_comparison, classification report, grid search
    └── figures/              # PNG: confusion matrix, ROC, comparisons, subspace grids
```

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt

# Run the full reproducible pipeline (data -> preprocess -> train -> evaluate)
python scripts/run_all.py

# Launch the dashboard
streamlit run app.py
```

Individual stages can also be run directly, e.g. `python pipelines/train.py`.

## Metrics

CLIQUE is unsupervised, so two metric families are reported (`clique/metrics.py`):

| Family | Metrics | Needs labels? |
|--------|---------|---------------|
| Intrinsic | silhouette, Davies-Bouldin, Calinski-Harabasz | No |
| Agreement | Adjusted Rand, NMI, homogeneity, completeness, V-measure | Yes |
| Classification | accuracy, precision/recall/F1 (macro & weighted), ROC-AUC (OvR) | Yes |

Classification-style metrics require ground-truth labels, which only the
**synthetic** dataset provides. Predicted clusters are mapped to segments by
**majority vote**; ROC-AUC uses **nearest-centroid softmax** scores. Because
majority-vote F1 rewards over-fragmentation, **model selection uses Adjusted Rand
Index** (tie-broken by silhouette), not F1.

## Reproducibility

`RANDOM_STATE = 42` is the single seed used for synthetic data generation,
`train_test_split`, and KMeans. DBSCAN, Agglomerative, and CLIQUE are
deterministic. All paths are defined once in `config.py`.

## Real data (optional)

Download [Online Retail II](https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip),
place `online_retail_ii.xlsx` in `data/raw/`, and run the last cell of
`notebooks/training.ipynb` (intrinsic metrics only — no ground truth).

## License

MIT
