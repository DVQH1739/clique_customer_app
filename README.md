# CLIQUE — Customer Subspace Clustering

**CLIQUE** ([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)) on **Online Retail II** (`data/raw/online_retail_ii.xlsx`). Command-line pipeline only.

## Requirements
scikit-learn>=1.4
pandas>=2.1
numpy>=1.26
joblib>=1.3
openpyxl>=3.1
nbformat>=5.9
matplotlib>=3.8
seaborn>=0.13

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt

python src/pipelines/run.py retail

# Or explicit path
python src/pipelines/run.py retail --xlsx data/raw/online_retail_ii.xlsx

# Algorithm smoke test (random 8D data)
python src/pipelines/run.py verify
```

## Data layout

```
data/
  raw/online_retail_ii.xlsx   # Online Retail II (tracked in git)
  processed/               # CSV caches from pipeline
models/                    # clique_model.pkl, scaler.pkl, profiles.pkl (gitignored)
results/
  metrics/                 # 5 CSV files
  figures/                 # EDA + benchmark PNGs
```

## Pipeline stages

| Stage | Module | What it does |
|-------|--------|----------------|
| Load & clean | `data.py` | Excel → UK transactions, drop cancellations/invalid rows |
| Features | `data.py` | 8 RFM-style features per customer; `frequency >= 2` |
| Preprocess | `preprocess.py` | Winsorize 1–99%, log1p, train/test split, MinMaxScaler |
| Train | `train.py` + `clique/algorithm.py` | Grid `(xi, tau)`, CLIQUE fit, save model |
| Benchmark | `benchmark.py` | KMeans / DBSCAN / Agglomerative vs CLIQUE |
| Validate | `validate.py` | Data contract, metrics consistency, artifacts |

## Cluster quality (latest)

See `docs/RESULTS.md`. Selected config: **xi=12, tau=0.18** — train silhouette ~**0.049**, test ~**0.038**, 7 clusters, ~96% coverage.

## Documentation

- `DATA_CONTRACT.md` — feature schema and paths
- `docs/RESULTS.md` — metrics summary and output checklist
- `docs/VALIDATION.md` — validation rules

## License

MIT
