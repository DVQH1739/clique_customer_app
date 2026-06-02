# Data Contract — CLIQUE Customer Profiles

## Raw input

| Path | Description |
|------|-------------|
| `data/raw/online_retail_ii.xlsx` | Online Retail II workbook (2 sheets), versioned in git. |

## Processed outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `clean_transactions.csv` | UK clean line items |
| `cancellations.csv` | Cancelled invoices (for return_rate) |
| `customer_profiles_raw.csv` | 8 features per customer (before scale) |
| `X_train_scaled.csv` | MinMax-scaled train split + `CustomerID` |
| `X_test_scaled.csv` | MinMax-scaled test split + `CustomerID` |

## Feature columns (order)

| # | Column | Description |
|---|--------|-------------|
| 1 | `recency` | Days since last purchase (snapshot = max date + 1 day) |
| 2 | `frequency` | Unique invoice count |
| 3 | `monetary` | Total revenue (Quantity × UnitPrice) |
| 4 | `avg_basket` | Mean revenue per invoice |
| 5 | `product_diversity` | Unique StockCode count |
| 6 | `return_rate` | Cancelled invoices / (cancelled + purchased) |
| 7 | `weekend_ratio` | Share of lines on Sat/Sun |
| 8 | `repeat_category_rate` | Share of lines in dominant StockCode prefix (2 chars) |

Optional: `CustomerID` (string).

## Constraints

- After scaling: features ∈ **[0.0, 1.0]**; no `NaN` or `inf`.
- Retail filter: `frequency >= 2` (repeat buyers).
- Scaler fit on **train only**; same transform applied to test.

## Pipeline

```
xlsx → clean → profiles → winsorize → log1p → split → MinMaxScaler → CLIQUE → metrics/figures
```

## Results (`results/`)

| CSV | Content |
|-----|---------|
| `metrics/retail_grid_search.csv` | Grid over xi, tau |
| `metrics/best_params.csv` | Selected params for the current objective |
| `metrics/baseline_comparison.csv` | CLIQUE vs baselines (train) |
| `metrics/cluster_descriptions.csv` | Subspace descriptions |
| `metrics/test_predictions.csv` | Test cluster assignments |
| `metrics/clique_subspace_coverage.csv` | Per-subspace coverage |

| PNG | Content |
|-----|---------|
| `figures/EDA_distributions.png` | Before/after transform |
| `figures/baseline_comparison.png` | Silhouette / DB / CH bars |
| `figures/subspace_heatmap.png` | 2D subspace cluster counts |
| `figures/cluster_sizes.png` | Cluster sizes |
| `figures/pareto_frontier.png` | Coverage vs silhouette trade-off |
| `figures/grid_*.png` | Top 2D dense-unit grids |
