"""
Data ingestion: synthetic ground-truth profiles and Online Retail II transactions.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config

N_SYNTHETIC = 600

SEGMENT_RANGES: list[dict[str, tuple[float, float]]] = [
    {
        "recency": (1, 45),
        "frequency": (20, 60),
        "monetary": (4000, 20000),
        "product_diversity": (120, 320),
        "return_rate": (0.0, 0.08),
        "weekend_ratio": (0.05, 0.30),
        "repeat_category_rate": (0.50, 0.85),
    },
    {
        "recency": (200, 365),
        "frequency": (1, 6),
        "monetary": (50, 800),
        "product_diversity": (1, 40),
        "return_rate": (0.15, 0.45),
        "weekend_ratio": (0.40, 0.80),
        "repeat_category_rate": (0.05, 0.30),
    },
    {
        "recency": (40, 160),
        "frequency": (8, 20),
        "monetary": (900, 4000),
        "product_diversity": (40, 120),
        "return_rate": (0.03, 0.18),
        "weekend_ratio": (0.20, 0.55),
        "repeat_category_rate": (0.35, 0.65),
    },
]


def generate_synthetic() -> pd.DataFrame:
    """Create natural-scale profiles with ground-truth ``true_segment``."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n_per = N_SYNTHETIC // 3
    counts = [n_per, n_per, N_SYNTHETIC - 2 * n_per]

    chunks: list[pd.DataFrame] = []
    for seg_idx, (params, n) in enumerate(zip(SEGMENT_RANGES, counts)):
        data: dict[str, np.ndarray] = {}
        for feat, (lo, hi) in params.items():
            data[feat] = rng.uniform(lo, hi, size=n)
        data["avg_basket"] = data["monetary"] / np.maximum(data["frequency"], 1)
        df = pd.DataFrame(data)
        df["true_segment"] = seg_idx
        df["segment_name"] = config.SEGMENT_NAMES[seg_idx]
        chunks.append(df)

    out = pd.concat(chunks, ignore_index=True)
    out = out.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)
    out["CustomerID"] = [f"C{i:05d}" for i in range(len(out))]
    cols = ["CustomerID", *config.FEATURE_NAMES, "true_segment", "segment_name"]
    return out[cols]


def persist_synthetic() -> None:
    """Write synthetic raw CSV to ``data/raw/``."""
    config.ensure_dirs()
    raw = generate_synthetic()
    raw.to_csv(config.RAW_CUSTOMERS_CSV, index=False, encoding="utf-8")
    print(f"Wrote {config.RAW_CUSTOMERS_CSV} ({len(raw)} rows)")
    print("Segment counts:\n", raw["segment_name"].value_counts().to_string())


def resolve_retail_xlsx(explicit: str | None = None) -> Path:
    """Resolve Online Retail II workbook path."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path
    if config.ONLINE_RETAIL_XLSX.exists():
        return config.ONLINE_RETAIL_XLSX
    raise FileNotFoundError(
        f"Place online_retail_ii.xlsx in {config.RAW_DIR} or pass --xlsx PATH."
    )


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load Online Retail II Excel (2 sheets) or CSV; normalize column names."""
    filepath = str(filepath)
    if filepath.endswith((".xlsx", ".xls")):
        print("Reading sheet: Year 2009-2010 ...")
        df1 = pd.read_excel(
            filepath, sheet_name="Year 2009-2010", dtype={"Customer ID": str}
        )
        print(f"  -> {len(df1):,} rows")
        print("Reading sheet: Year 2010-2011 ...")
        df2 = pd.read_excel(
            filepath, sheet_name="Year 2010-2011", dtype={"Customer ID": str}
        )
        print(f"  -> {len(df2):,} rows")
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = pd.read_csv(filepath, encoding="latin-1", dtype={"CustomerID": str})

    df.rename(
        columns={
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
            "Customer ID": "CustomerID",
        },
        inplace=True,
    )
    return df


def clean_data(
    df: pd.DataFrame, filter_uk: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean transactions; return (clean_df, cancellations_df)."""
    n_original = len(df)
    print(f"\n{'=' * 50}")
    print("CLEANING TRANSACTIONS")
    print(f"{'=' * 50}")
    print(f"[0] Raw rows:              {n_original:>10,}")

    df = df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    cancellations_df = df[df["InvoiceNo"].str.startswith("C")].copy()
    df = df[~df["InvoiceNo"].str.startswith("C")].copy()
    print(
        f"[1] After drop cancellations:{len(df):>10,}  "
        f"({len(cancellations_df):,} cancelled)"
    )

    df = df[df["CustomerID"].notna() & (df["CustomerID"] != "")].copy()
    print(f"[2] After valid CustomerID: {len(df):>10,}")

    df = df[df["Quantity"] > 0].copy()
    print(f"[3] After Quantity > 0:     {len(df):>10,}")

    df = df[df["UnitPrice"] > 0].copy()
    print(f"[4] After UnitPrice > 0:    {len(df):>10,}")

    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    print(
        f"[5] After dedup:            {len(df):>10,}  "
        f"(removed {before_dedup - len(df):,})"
    )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[df["InvoiceDate"].notna()].copy()
    print(f"[6] After parse dates:      {len(df):>10,}")

    if filter_uk:
        df = df[df["Country"] == "United Kingdom"].copy()
        print(f"[7] After UK filter:        {len(df):>10,}")

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"\n{'=' * 50}")
    print("CLEAN SUMMARY:")
    print(f"  Rows:             {len(df):,}")
    print(f"  Unique customers: {df['CustomerID'].nunique():,}")
    if len(df) > 0:
        print(
            f"  Date range:       "
            f"{df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}"
        )
    print(f"  Cancellations:    {len(cancellations_df):,}")
    print(f"{'=' * 50}\n")

    return df, cancellations_df


def build_customer_profiles(
    clean_df: pd.DataFrame,
    cancellations_df: pd.DataFrame,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate eight behavioral features per CustomerID."""
    if snapshot_date is None:
        snapshot_date = clean_df["InvoiceDate"].max() + timedelta(days=1)
    print(f"Snapshot date: {snapshot_date.date()}")

    grouped = clean_df.groupby("CustomerID")
    recency = (snapshot_date - grouped["InvoiceDate"].max()).dt.days
    frequency = grouped["InvoiceNo"].nunique()
    monetary = grouped["Revenue"].sum()
    invoice_revenue = (
        clean_df.groupby(["CustomerID", "InvoiceNo"])["Revenue"].sum().reset_index()
    )
    avg_basket = invoice_revenue.groupby("CustomerID")["Revenue"].mean()

    product_diversity = grouped["StockCode"].nunique()

    cancel_counts = (
        cancellations_df[cancellations_df["CustomerID"].notna()]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
    )
    purchase_counts = grouped["InvoiceNo"].nunique()
    return_rate = cancel_counts / (cancel_counts + purchase_counts)
    return_rate = return_rate.fillna(0)

    work = clean_df.copy()
    work["is_weekend"] = work["InvoiceDate"].dt.dayofweek >= 5
    weekend_ratio = work.groupby("CustomerID")["is_weekend"].mean()

    work["category"] = work["StockCode"].astype(str).str[:2]
    cat_counts = work.groupby(["CustomerID", "category"]).size().reset_index(name="cnt")
    top_cat = cat_counts.loc[cat_counts.groupby("CustomerID")["cnt"].idxmax()]
    total_lines = work.groupby("CustomerID").size()
    repeat_category_rate = top_cat.set_index("CustomerID")["cnt"] / total_lines
    repeat_category_rate = repeat_category_rate.fillna(0)

    profiles = pd.DataFrame(
        {
            "CustomerID": recency.index,
            "recency": recency.values,
            "frequency": frequency.reindex(recency.index).values,
            "monetary": monetary.reindex(recency.index).values,
            "avg_basket": avg_basket.reindex(recency.index).fillna(0).values,
            "product_diversity": product_diversity.reindex(recency.index).values,
            "return_rate": return_rate.reindex(recency.index).fillna(0).values,
            "weekend_ratio": weekend_ratio.reindex(recency.index).fillna(0).values,
            "repeat_category_rate": repeat_category_rate.reindex(recency.index).fillna(0).values,
        }
    )
    profiles = profiles.fillna(0)
    assert not profiles[config.FEATURE_NAMES].isna().any().any()
    assert not np.isinf(profiles[config.FEATURE_NAMES].values).any()

    print("\n--- FEATURE STATS (before transform) ---")
    stats = profiles[config.FEATURE_NAMES].describe().T[["mean", "std", "min", "max"]]
    stats["skewness"] = profiles[config.FEATURE_NAMES].skew()
    stats["needs_log"] = stats["skewness"].abs() > 1
    print(stats.round(2).to_string())
    print(f"\nTotal customers: {len(profiles):,}")

    return profiles


def filter_retail_customers(profiles: pd.DataFrame) -> pd.DataFrame:
    """Keep repeat buyers; reduces noise from one-off purchasers."""
    min_freq = config.RETAIL_MIN_FREQUENCY
    before = len(profiles)
    out = profiles[profiles["frequency"] >= min_freq].copy()
    print(
        f"Filter frequency >= {min_freq}: {before:,} -> {len(out):,} customers"
    )
    return out.reset_index(drop=True)


def build_profiles_from_xlsx(
    xlsx_path: str | Path,
    *,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Load Online Retail II from Excel, clean, aggregate features.

    This is the canonical retail data entry (not CSV).
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    print("\n" + "=" * 70)
    print("STEP 1-2: LOAD XLSX, CLEAN, FEATURE ENGINEERING")
    print("=" * 70)

    raw_df = load_raw_data(xlsx_path)
    print(f"\nMerged rows: {len(raw_df):,}")
    clean_df, cancel_df = clean_data(raw_df, filter_uk=True)
    profiles = build_customer_profiles(clean_df, cancel_df)
    profiles = filter_retail_customers(profiles)

    if save_artifacts:
        config.ensure_dirs()
        clean_df.to_csv(config.CLEAN_TRANSACTIONS_CSV, index=False)
        cancel_df.to_csv(config.CANCELLATIONS_CSV, index=False)
        profiles.to_csv(config.CUSTOMER_PROFILES_RAW_CSV, index=False)
        print(
            f"Cached: {config.CLEAN_TRANSACTIONS_CSV.name}, "
            f"{config.CUSTOMER_PROFILES_RAW_CSV.name}"
        )

    return profiles


def guess_business_label(dims: list[str], profile: np.ndarray) -> str:
    """Heuristic business label from subspace and mean scaled profile."""
    dim_set = set(dims)
    if "frequency" in dim_set and "monetary" in dim_set:
        fi = config.FEATURE_NAMES.index("frequency")
        mi = config.FEATURE_NAMES.index("monetary")
        if profile[fi] > 0.6 and profile[mi] > 0.6:
            return "VIP — Mua nhiều, chi tiêu cao"
        if profile[fi] < 0.3:
            return "One-time buyer"
    if "recency" in dim_set and "weekend_ratio" in dim_set:
        return "Seasonal / Weekend shopper"
    if "repeat_category_rate" in dim_set and "frequency" in dim_set:
        return "Brand-loyal customer"
    if "product_diversity" in dim_set and "avg_basket" in dim_set:
        return "Explorer — Thích đa dạng sản phẩm"
    if "return_rate" in dim_set:
        ri = config.FEATURE_NAMES.index("return_rate")
        if profile[ri] > 0.4:
            return "High-return risk"
    return "General cluster"


def load_and_clean_retail(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper around ``build_profiles_from_xlsx``."""
    profiles = build_profiles_from_xlsx(xlsx_path, save_artifacts=True)
    clean_df = pd.read_csv(config.CLEAN_TRANSACTIONS_CSV)
    cancel_df = pd.read_csv(config.CANCELLATIONS_CSV)
    return clean_df, cancel_df, profiles
