# CLIQUE — Customer Subspace Clustering

Implementation of **CLIQUE** ([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)) with reproducible training pipelines and baseline comparison. No web UI — run from the command line or notebook.

## Project structure

```
clique_customer_app/
├── README.md · requirements.txt · DATA_CONTRACT.md
├── docs/RESULTS.md
├── src/
│   ├── config.py
│   ├── clique/              # algorithm · metrics · io
│   ├── pipelines/           # data · preprocess · train · benchmark · run.py
│   └── notebooks/training.ipynb
├── data/raw/ · data/processed/{synthetic,retail}/
├── models/{synthetic,retail}/
└── results/{metrics,figures}/
```

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt

# Synthetic demo (600 labeled customers)
python src/pipelines/run.py synthetic

# Online Retail II — place online_retail_ii.xlsx in data/raw/
python src/pipelines/run.py retail

# Smoke test
python src/pipelines/run.py verify
```

Stages: `--step data|preprocess|train|benchmark`

## Pipelines

| Mode | Data | Model selection | Output |
|------|------|-----------------|--------|
| `synthetic` | Generated CSV + ground truth | ARI → silhouette | Metrics + figures + `models/synthetic/` |
| `retail` | UCI Online Retail II | Silhouette + coverage | `models/retail/` + test predictions |

## License

MIT
