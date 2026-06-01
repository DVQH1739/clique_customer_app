# CLIQUE Customer Intelligence

Subspace clustering for customer profiles using the **CLIQUE** algorithm
([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)), with a
reproducible ML pipeline, baseline comparison, and a Streamlit app.

## Project structure (4 folders)

```
clique_customer_app/
├── README.md · requirements.txt · DATA_CONTRACT.md
├── src/                      # ALL code
│   ├── config.py             # Central paths, FEATURE_NAMES, RANDOM_STATE=42
│   ├── app.py                # Streamlit UI (4 pages)
│   ├── clique/               # Core library
│   │   ├── algorithm.py      # CLIQUE implementation
│   │   ├── metrics.py        # Intrinsic + supervised metrics
│   │   └── utils.py          # Data loading, cleaning, feature engineering, I/O
│   ├── pipelines/            # 1 generate · 2 preprocess · 3 train · 4 evaluate
│   ├── scripts/              # run_all.py · verify_clique.py · push_to_github.ps1
│   └── notebooks/            # training.ipynb (EDA + orchestration)
├── data/
│   ├── raw/                  # Raw inputs (synthetic CSV; real .xlsx ignored by git)
│   └── processed/            # Scaled features + label CSVs
├── models/                   # clique_model.pkl, scaler.pkl, profiles.pkl
└── results/
    ├── metrics/              # CSV: model_comparison, classification report, grid, coverage
    └── figures/              # PNG: confusion matrix, metric comparison, heatmap, grids
```

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt

# Run the full reproducible pipeline (data -> preprocess -> train -> evaluate)
python src/scripts/run_all.py

# Launch the dashboard
streamlit run src/app.py
```

Individual stages can also be run directly, e.g. `python src/pipelines/train.py`.

## Metrics

CLIQUE is unsupervised, so two metric families are reported (`src/clique/metrics.py`):

| Family | Metrics | Needs labels? |
|--------|---------|---------------|
| Intrinsic | silhouette, Davies-Bouldin, Calinski-Harabasz | No |
| Agreement | Adjusted Rand, NMI, homogeneity, completeness, V-measure | Yes |
| Classification | accuracy, precision/recall/F1 (macro & weighted) | Yes |

Classification-style metrics require ground-truth labels, which only the
**synthetic** dataset provides. Predicted clusters are mapped to segments by
**majority vote**. Because majority-vote F1 rewards over-fragmentation, **model
selection uses Adjusted Rand Index** (tie-broken by silhouette), not F1.

> ROC/AUC was intentionally removed: for a hard clustering it has no well-defined
> score, and the centroid-based proxy produced a degenerate AUC of 1.0 for every
> algorithm — misleading rather than informative.

## Results summary (synthetic data)

| Algorithm | ARI | F1-macro | Silhouette | Note |
|-----------|----:|---------:|-----------:|------|
| KMeans (k=4) | **0.87** | 1.00 | **0.39** | best overall |
| Agglomerative (k=5) | 0.84 | 1.00 | 0.37 | strong |
| CLIQUE (xi=8, tau=0.08) | 0.34 | 0.99 | -0.10 | recovers segments but over-fragments |
| DBSCAN (eps=0.5) | 0.57 | 0.56 | 0.60 | merges two segments |

CLIQUE separates segments well after alignment (F1≈0.99) but its subspace clusters
over-fragment the space (low ARI/silhouette) — an honest, expected characteristic
of the algorithm on globular RFM-style data.

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
