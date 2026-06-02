# CLIQUE — Customer Subspace Clustering

**CLIQUE** ([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)) on **Online Retail II** (`data/raw/online_retail_ii.xlsx`). Command-line pipeline only.

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt

# Primary: load Excel -> clean -> features -> train -> evaluate
python src/pipelines/run.py retail

# Explicit workbook path
python src/pipelines/run.py retail --xlsx data/raw/online_retail_ii.xlsx
```

Place `online_retail_ii.xlsx` in `data/raw/` (not CSV). Files under `data/processed/retail/` are **caches** written after each run.

## Project layout

```
src/
  clique/           # algorithm, metrics, io
  pipelines/
    data.py         # load xlsx, clean, 8 features
    preprocess.py   # winsorize, log1p, MinMaxScaler
    train.py        # grid search + CLIQUE fit
    benchmark.py    # baselines + test metrics
    retail.py       # end-to-end from xlsx
    run.py          # CLI entry
data/raw/online_retail_ii.xlsx
models/retail/      # clique_model.pkl, scaler.pkl, profiles.pkl
results/            # metrics CSV, figures PNG
```

## Cluster quality (retail)

Training uses an expanded `(xi, tau)` grid and a **composite score** (silhouette, coverage, Davies–Bouldin, penalize over-fragmentation). Preprocessing:

- UK customers only, `frequency >= 2` (repeat buyers)
- Winsorize features at 1st/99th percentile before log-transform

Optional demo with synthetic CSV + ground truth:

```bash
python src/pipelines/run.py synthetic
```

## License

MIT
