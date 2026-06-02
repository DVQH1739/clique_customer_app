# CLIQUE Customer Intelligence

Subspace clustering for customer profiles using **CLIQUE**
([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)): reproducible pipelines,
baseline comparison, and a Streamlit app.

## Project structure

```
clique_customer_app/
├── README.md · requirements.txt · DATA_CONTRACT.md
├── docs/RESULTS.md              # Interpretation of metrics (Vietnamese)
├── src/
│   ├── config.py                # Paths, FEATURE_NAMES, RANDOM_STATE=42
│   ├── app.py                   # Streamlit UI
│   ├── clique/
│   │   ├── algorithm.py         # CLIQUE (3 phases)
│   │   ├── metrics.py           # Intrinsic + supervised metrics
│   │   └── io.py                # Model load/save, validation
│   └── pipelines/
│       ├── data.py              # Synthetic + Online Retail load/clean/features
│       ├── preprocess.py        # Log-transform, split, MinMaxScaler
│       ├── train.py             # Grid search + final fit
│       ├── benchmark.py         # Baselines, test eval, figures
│       └── run.py               # Single CLI entry point
├── data/raw/
├── data/processed/synthetic/    # Scaled features + labels (demo)
├── data/processed/retail/       # Online Retail II artifacts
├── models/synthetic/ · models/retail/
└── results/metrics/ · results/figures/
```

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt

# Synthetic demo (600 labeled customers) — full pipeline
python src/pipelines/run.py synthetic

# Online Retail II — place online_retail_ii.xlsx in data/raw/
python src/pipelines/run.py retail

# Smoke test CLIQUE implementation
python src/pipelines/run.py verify

# Dashboard
streamlit run src/app.py
```

Run individual stages, e.g. `python src/pipelines/run.py synthetic --step train`.

## Pipelines

| Mode | Data | Grid selection | Benchmark |
|------|------|----------------|-----------|
| `synthetic` | Generated CSV + ground truth | ARI → silhouette | Supervised + intrinsic + test metrics |
| `retail` | UCI Online Retail II | Silhouette + coverage constraints | Intrinsic baselines + test predict |

## Metrics

See `src/clique/metrics.py` and `docs/RESULTS.md`. Model selection on synthetic data uses **Adjusted Rand Index** (not F1), because majority-vote F1 rewards over-fragmentation.

## Reproducibility

`RANDOM_STATE = 42` in `config.py` for data generation, splits, and KMeans.

## License

MIT
