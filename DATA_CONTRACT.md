# Data Contract — CLIQUE Customer Profiles

## Overview

All model inputs must be **MinMax-scaled** customer profile vectors in **[0.0, 1.0]** per feature. Raw-scale profiles are log-transformed and scaled by the training pipeline before inference.

## File Format

| Property    | Value        |
|-------------|--------------|
| Encoding    | UTF-8        |
| Delimiter   | Comma (`,`)  |
| Header row  | Required     |
| Index column| Not required |

## Feature Column Order (strict)

Columns must appear in this exact order when present (additional ID columns allowed):

| # | Column name            | Description                                      |
|---|------------------------|--------------------------------------------------|
| 1 | `recency`              | Days since last purchase (higher = less recent)  |
| 2 | `frequency`            | Unique invoice count                             |
| 3 | `monetary`             | Total revenue                                    |
| 4 | `avg_basket`           | Monetary / frequency                             |
| 5 | `product_diversity`    | Unique StockCode count                           |
| 6 | `return_rate`          | Cancellation fraction                          |
| 7 | `weekend_ratio`        | Fraction of weekend purchases                  |
| 8 | `repeat_category_rate` | Loyalty to top product category                |

Optional identifier column: `CustomerID` (string).

## Value Constraints

- After scaling: every feature value ∈ **[0.0, 1.0]**
- No `NaN`, `inf`, or non-numeric values in feature columns
- Missing any feature column → **`ValueError`** at load time

## Pre-Scaled vs Raw Profiles

| Upload type              | Detection                          | Pipeline                          |
|--------------------------|------------------------------------|-----------------------------------|
| Pre-computed profile     | All 8 `FEATURE_NAMES` present      | `scaler.transform()` → `predict()`|
| Raw transactions         | `InvoiceNo` / `Invoice` / `StockCode`| `clean` → `profiles` → `scale` → `predict()` |

## Mock Test Files

| File                 | Purpose                                      |
|----------------------|----------------------------------------------|
| `data/test_clean.csv`     | 500 valid rows, values in [0, 1]        |
| `data/test_corrupted.csv` | 200 rows with missing cols, strings, NaN |

## Model Artifacts (`models/`)

| File               | Contents                    |
|--------------------|-----------------------------|
| `clique_model.pkl` | Fitted `CLIQUE` instance    |
| `scaler.pkl`       | Fitted `MinMaxScaler`       |
| `profiles.pkl`     | Cluster mean vectors (scaled)|

## Training Date / Version

Artifacts are versioned as `*_v1.pkl` and mirrored as unversioned `.pkl` for the Streamlit app.
