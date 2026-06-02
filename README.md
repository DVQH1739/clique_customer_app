# CLIQUE — Customer Subspace Clustering

**CLIQUE** ([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)) on **Online Retail II** (`data/raw/online_retail_ii.xlsx`). Command-line pipeline only.

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt

# Place workbook in data/raw/ (see data/raw/README.md)
python src/pipelines/run.py retail

# Or explicit path
python src/pipelines/run.py retail --xlsx data/raw/online_retail_ii.xlsx

# Algorithm smoke test (random 8D data)
python src/pipelines/run.py verify
```

## Data layout

```
data/
  raw/
    README.md              # how to obtain online_retail_ii.xlsx
    online_retail_ii.xlsx  # local only (gitignored)
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
