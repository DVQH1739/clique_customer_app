# Data Contract — CLIQUE Customer Profiles



## Overview



Training and inference use **MinMax-scaled** profile vectors in **[0.0, 1.0]** per feature. Raw-scale profiles are log-transformed; the scaler is fit on the train split only.



## Feature columns (order)



| # | Column | Description |

|---|--------|-------------|

| 1 | `recency` | Days since last purchase |

| 2 | `frequency` | Unique invoice count |

| 3 | `monetary` | Total revenue |

| 4 | `avg_basket` | Monetary / frequency |

| 5 | `product_diversity` | Unique StockCode count |

| 6 | `return_rate` | Cancellation fraction |

| 7 | `weekend_ratio` | Fraction of weekend purchases |

| 8 | `repeat_category_rate` | Top-category loyalty ratio |



Optional: `CustomerID` (string).



## Constraints



- Scaled features ∈ **[0.0, 1.0]**; no `NaN` or `inf`.



## Paths (`config.py`)



| Path | Purpose |

|------|---------|

| `data/raw/online_retail_ii.xlsx` | Only raw input (Excel) |

| `data/processed/` | Clean transactions, profiles, scaled train/test CSVs |

| `models/` | `clique_model.pkl`, `scaler.pkl`, `profiles.pkl` |

| `results/metrics/` · `results/figures/` | Evaluation CSVs and PNGs |



## Pipeline



Source: **`data/raw/online_retail_ii.xlsx`**.



Flow: load xlsx → clean → 8 features → winsorize → log1p → MinMaxScaler → CLIQUE grid search → `models/`.



Customers with `frequency >= 2` only (repeat buyers). Cached CSVs under `data/processed/` are optional outputs.

