# CLIQUE Customer Intelligence

Subspace clustering for e-commerce customer profiles using the **CLIQUE** algorithm ([Agrawal et al., SIGMOD 1998](https://doi.org/10.1145/276304.276306)).

## Features

- CLIQUE implementation from scratch (`clique/algorithm.py`)
- Online Retail II data pipeline (load → clean → RFM-style features)
- Streamlit app: dashboard, batch profiling, single-customer lookup, subspace explorer
- Training notebook with grid search over `xi` and `tau`

## Quick start

```bash
cd clique_customer_app
python -m pip install -r requirements.txt
python data/generate_mock.py
python notebooks/train_quick.py
streamlit run app.py
```

## Project layout

| Path | Description |
|------|-------------|
| `clique/` | Core algorithm and utilities |
| `app.py` | Streamlit UI |
| `notebooks/training.ipynb` | Full training on Online Retail II |
| `data/` | Mock CSV generator + dataset (download separately) |
| `models/` | Saved `clique_model.pkl`, `scaler.pkl`, `profiles.pkl` |
| `DATA_CONTRACT.md` | Input CSV schema |

## Dataset

Download [Online Retail II](https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip) and place `online_retail_ii.xlsx` in `data/`, then run `notebooks/training.ipynb`.

## License

MIT
