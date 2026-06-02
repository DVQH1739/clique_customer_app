# CLIQUE — Customer Subspace Clustering



**CLIQUE** ([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)) on **Online Retail II** (`data/raw/online_retail_ii.xlsx`). Command-line pipeline only.



## Quick start



```bash

cd clique_customer_app

python -m pip install -r requirements.txt



# Load Excel -> clean -> features -> train -> evaluate

python src/pipelines/run.py retail



# Explicit workbook path

python src/pipelines/run.py retail --xlsx data/raw/online_retail_ii.xlsx

```



Place `online_retail_ii.xlsx` in `data/raw/` (only raw file needed). Processed CSVs are written to `data/processed/` after each run.



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

data/

  raw/online_retail_ii.xlsx

  processed/        # clean_transactions, profiles, scaled splits

models/             # clique_model.pkl, scaler.pkl, profiles.pkl

results/            # metrics CSV, figures PNG

```



## Cluster quality



Training uses an expanded `(xi, tau)` grid and a **composite score** (silhouette, coverage, Davies–Bouldin, penalize over-fragmentation). Preprocessing:



- UK customers only, `frequency >= 2` (repeat buyers)

- Winsorize features at 1st/99th percentile before log-transform

Latest run (`docs/RESULTS.md`): **xi=12, tau=0.18** — train silhouette ~0.049, test ~0.038, 7 clusters, ~96% coverage.

## Outputs

After `run.py retail`: 5 CSVs in `results/metrics/`, figures in `results/figures/` (EDA, baselines, subspace heatmap, cluster sizes, grid plots). Models in `models/` (regenerated locally; `.pkl` gitignored).

## License



MIT

