"""
Streamlit app for CLIQUE customer subspace clustering.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from clique.algorithm import CLIQUE, NOISE_LABEL
from clique.io import apply_log_transform, load_model, validate_profile_csv
from pipelines.benchmark import run_baseline_comparison
from pipelines.data import build_customer_profiles, clean_data, load_raw_data

MODELS_DIR = config.SYNTHETIC_MODELS
DATA_DIR = config.DATA_DIR
FEATURE_NAMES = config.FEATURE_NAMES
FEATURE_DISPLAY_NAMES = config.FEATURE_DISPLAY_NAMES
LOG_TRANSFORM_COLS = config.LOG_TRANSFORM_COLS

MARKETING_TIPS: dict[int, str] = {
    0: "VIP retention: offer early access and loyalty rewards to protect high-value share.",
    1: "Win-back campaign: send personalized discounts to re-engage dormant buyers.",
    2: "Cross-sell: recommend complementary products from their favorite category.",
    3: "Weekend push: promote limited-time weekend bundles to boost off-peak visits.",
    4: "Quality focus: highlight product reviews and easy returns to reduce churn.",
    5: "Basket builder: suggest bundles to increase average order value.",
}
DEFAULT_TIP = "General nurture: send a personalized thank-you and category recommendations."


@st.cache_resource
def load_cached_model() -> tuple[CLIQUE, object, dict]:
    """Load model, scaler, and profiles once per session."""
    return load_model(str(MODELS_DIR))


def get_model_metadata(model: CLIQUE) -> dict:
    """Read model file mtime and hyperparameters."""
    model_path = MODELS_DIR / "clique_model.pkl"
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        train_date = mtime.strftime("%Y-%m-%d %H:%M")
    else:
        train_date = "Unknown"
    runtime = getattr(model, "_training_runtime_sec", None) or 0.0
    return {
        "xi": model.xi,
        "tau": model.tau,
        "training_date": train_date,
        "runtime_sec": f"{runtime:.2f}",
    }


def style_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Color-code best metric per column (green)."""
    styled = df.copy()

    def highlight_best(series: pd.Series, higher_better: bool) -> list[str]:
        valid = series.dropna()
        if valid.empty:
            return [""] * len(series)
        best_val = valid.max() if higher_better else valid.min()
        return [
            "background-color: #c8e6c9"
            if pd.notna(v) and v == best_val
            else ""
            for v in series
        ]

    return styled  # Streamlit st.dataframe doesn't take Styler easily; use column format


def render_dashboard(model: CLIQUE, scaler, profiles: dict) -> None:
    """Page A: metrics, heatmap, cluster sizes, baselines, model info."""
    st.header("Dashboard")

    labels = model.labels_
    if labels is None:
        st.warning("Model labels not available.")
        return

    n_clusters = len(set(labels)) - (1 if NOISE_LABEL in labels else 0)
    subspaces_with = len({c["subspace"] for c in model.clusters_})
    total_cov = sum(model.subspace_coverage_.values()) / max(len(model.subspace_coverage_), 1)
    noise_pct = 100.0 * (labels == NOISE_LABEL).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters", n_clusters)
    c2.metric("Subspaces with clusters", subspaces_with)
    c3.metric("Avg subspace coverage", f"{100 * total_cov:.1f}%")
    c4.metric("Noise %", f"{noise_pct:.1f}%")

    # Heatmap
    heat_df = model.get_subspace_heatmap_data()
    if not heat_df.empty:
        pivot = heat_df.pivot_table(
            index="dim_1", columns="dim_2", values="n_clusters", fill_value=0
        )
        fig_hm = px.imshow(
            pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            labels=dict(color="Clusters"),
            title="2D Subspace Cluster Counts",
            aspect="auto",
        )
        fig_hm.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No 2D subspace heatmap data yet.")

    # Cluster sizes
    sizes = [c["size"] for c in model.clusters_]
    ids = [c["id"] for c in model.clusters_]
    if sizes:
        fig_bar = px.bar(x=[str(i) for i in ids], y=sizes, labels={"x": "Cluster ID", "y": "Size"})
        fig_bar.update_layout(title="Cluster Sizes")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Baseline comparison
    st.subheader("Baseline Comparison")
    X_path = config.X_TRAIN_SCALED_CSV
    if X_path.exists():
        try:
            X = pd.read_csv(X_path).values
            baseline_df = run_baseline_comparison(X)
            st.dataframe(
                baseline_df.style.apply(
                    lambda s: [
                        "background-color: #c8e6c9"
                        if s.name == "silhouette"
                        and pd.notna(v)
                        and v == baseline_df["silhouette"].max()
                        else (
                            "background-color: #c8e6c9"
                            if s.name == "calinski_harabasz"
                            and pd.notna(v)
                            and v == baseline_df["calinski_harabasz"].max()
                            else (
                                "background-color: #c8e6c9"
                                if s.name == "davies_bouldin"
                                and pd.notna(v)
                                and v == baseline_df["davies_bouldin"].min()
                                else ""
                            )
                        )
                        for v in s
                    ],
                    axis=0,
                ),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Baseline comparison failed: {e}")
    else:
        st.info("Train the model to generate X_train_scaled.csv for baselines.")

    st.subheader("Model Info")
    meta = get_model_metadata(model)
    st.table(pd.DataFrame([meta]))


def render_batch_profiling(model: CLIQUE, scaler, profiles: dict) -> None:
    """Page B: CSV upload, predict, visualize, download."""
    st.header("Batch Profiling")
    uploaded = st.file_uploader("Upload CSV (transactions or customer profiles)", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to profile customers.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_df = pd.read_csv(uploaded, encoding="latin-1")

    is_raw_transactions = any(
        col in uploaded_df.columns for col in ["InvoiceNo", "Invoice", "StockCode"]
    )

    try:
        if is_raw_transactions:
            st.info("Mode detected: **Raw transactions** — running full feature pipeline.")
            clean_df, cancel_df = clean_data(uploaded_df, filter_uk=True)
            customer_profiles = build_customer_profiles(clean_df, cancel_df)
            customer_profiles = apply_log_transform(customer_profiles)
            validate_profile_csv(customer_profiles)
            X = scaler.transform(customer_profiles[FEATURE_NAMES].values)
            result_df = customer_profiles.copy()
        else:
            st.info("Mode detected: **Pre-computed customer profile**.")
            validate_profile_csv(uploaded_df)
            profiles_in = uploaded_df.copy()
            feat = profiles_in[FEATURE_NAMES].astype(float)
            # DATA_CONTRACT: values already in [0,1] after MinMaxScaler → predict directly
            if feat.min().min() >= 0.0 and feat.max().max() <= 1.0:
                st.caption("Values in [0, 1] — using as scaled features (no re-scaling).")
                X = feat.values
            else:
                st.caption("Raw-scale profiles — applying log-transform + MinMaxScaler.")
                profiles_in = apply_log_transform(profiles_in)
                X = scaler.transform(profiles_in[FEATURE_NAMES].values)
            result_df = profiles_in.copy()

        labels = model.predict(X)
        result_df["cluster_id"] = labels
        result_df["cluster_description"] = result_df["cluster_id"].apply(
            lambda cid: next(
                (c["description"] for c in model.clusters_ if c["id"] == cid),
                "Noise / unassigned",
            )
            if cid >= 0
            else "Noise / unassigned"
        )

        def row_color(row: pd.Series) -> list[str]:
            cid = row.get("cluster_id", NOISE_LABEL)
            if cid == NOISE_LABEL:
                return ["background-color: #eeeeee"] * len(row)
            colors = ["#e3f2fd", "#fff3e0", "#e8f5e9", "#fce4ec", "#f3e5f5"]
            color = colors[int(cid) % len(colors)]
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            result_df.style.apply(row_color, axis=1),
            use_container_width=True,
            height=400,
        )

        plot_df = result_df.copy()
        plot_df["cluster_label"] = plot_df["cluster_id"].astype(str)
        fig_splom = px.scatter_matrix(
            plot_df,
            dimensions=FEATURE_NAMES,
            color="cluster_label",
            title="Customer Feature Scatter Matrix",
        )
        fig_splom.update_layout(height=800)
        st.plotly_chart(fig_splom, use_container_width=True)

        st.subheader("Cluster Profile Cards")
        unique_clusters = sorted(set(labels) - {NOISE_LABEL})
        for cid in unique_clusters[:6]:
            mask = labels == cid
            st.markdown(f"**Cluster {cid}** ({mask.sum()} customers)")
            cols = st.columns(4)
            cluster_mean = profiles.get(cid, model.cluster_profiles_.get(cid))
            for i, feat in enumerate(FEATURE_NAMES):
                with cols[i % 4]:
                    cust_mean = X[mask, i].mean() if mask.any() else 0
                    cm = cluster_mean[i] if cluster_mean is not None else 0
                    st.metric(
                        FEATURE_DISPLAY_NAMES.get(feat, feat),
                        f"{cust_mean:.3f}",
                        delta=f"{cust_mean - cm:+.3f} vs cluster",
                    )

        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_out,
            file_name="batch_profiling_results.csv",
            mime="text/csv",
        )
    except ValueError as e:
        st.error(f"Validation error: {e}")
    except Exception as e:
        st.error(f"Profiling failed: {e}. Check CSV format against DATA_CONTRACT.md.")


def render_single_customer(model: CLIQUE, scaler, profiles: dict) -> None:
    """Page C: sliders, predict, radar, marketing tip, history."""
    st.header("Single Customer Profiler")

    if "customer_history" not in st.session_state:
        st.session_state.customer_history = []

    st.subheader("Enter customer features (raw-scale or pre-log values)")
    values: dict[str, float] = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURE_NAMES):
        with cols[i % 2]:
            values[feat] = st.slider(
                FEATURE_DISPLAY_NAMES.get(feat, feat),
                0.0,
                1.0,
                0.5,
                0.01,
                key=f"slider_{feat}",
            )

    if st.button("Predict cluster", type="primary"):
        try:
            row = pd.DataFrame([values])
            row_log = apply_log_transform(row)
            X = scaler.transform(row_log[FEATURE_NAMES].values)
            label = int(model.predict(X)[0])
            desc = next(
                (c["description"] for c in model.clusters_ if c["id"] == label),
                "Noise — no dense unit match",
            )
            st.success(f"**Cluster ID:** {label}")
            st.write(f"**Description:** {desc}")
            tip = MARKETING_TIPS.get(label % len(MARKETING_TIPS), DEFAULT_TIP)
            st.info(f"**Recommendation:** {tip}")

            cluster_mean = profiles.get(label, model.cluster_profiles_.get(label))
            if cluster_mean is not None:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=list(X[0]),
                        theta=[FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_NAMES],
                        fill="toself",
                        name="Customer",
                    )
                )
                fig.add_trace(
                    go.Scatterpolar(
                        r=list(cluster_mean),
                        theta=[FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_NAMES],
                        fill="toself",
                        name=f"Cluster {label} mean",
                        opacity=0.6,
                    )
                )
                fig.update_layout(title="Customer vs Cluster Profile", polar=dict(radialaxis=dict(range=[0, 1])))
                st.plotly_chart(fig, use_container_width=True)

            st.session_state.customer_history.insert(
                0,
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "cluster_id": label,
                    "description": desc[:80],
                },
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.subheader("Session History")
    if st.session_state.customer_history:
        st.dataframe(pd.DataFrame(st.session_state.customer_history), use_container_width=True)
    else:
        st.caption("No predictions yet this session.")


def render_subspace_explorer(
    model: CLIQUE, X_sample: np.ndarray, feature_names: list[str]
) -> None:
    """Page D: 2D grid plot and subspace ranking."""
    st.header("Subspace Explorer")
    fx = st.selectbox("Feature X", range(len(feature_names)), format_func=lambda i: feature_names[i])
    fy = st.selectbox(
        "Feature Y",
        range(len(feature_names)),
        index=min(1, len(feature_names) - 1),
        format_func=lambda i: feature_names[i],
    )
    if fx == fy:
        st.warning("Select two different features.")
        return

    try:
        fig = model.plot_grid_static(X_sample, fx, fy, feature_names)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plot failed: {e}")

    st.subheader("Subspace Ranking")
    rows = []
    for subspace, cov in model.subspace_coverage_.items():
        n_cl = sum(1 for c in model.clusters_ if c["subspace"] == subspace)
        dim_labels = ", ".join(feature_names[d] for d in subspace)
        rows.append({"subspace": dim_labels, "dimensions": len(subspace), "n_clusters": n_cl, "coverage": cov})
    rank_df = pd.DataFrame(rows).sort_values("coverage", ascending=False)
    st.dataframe(rank_df, use_container_width=True)


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="CLIQUE Customer Intelligence", layout="wide")
    st.title("CLIQUE Customer Subspace Clustering")

    try:
        model, scaler, profiles = load_cached_model()
    except FileNotFoundError:
        st.error(
            "No trained model found in `models/`. "
            "Run `notebooks/training.ipynb` or place clique_model.pkl, scaler.pkl, profiles.pkl."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    X_sample = None
    x_path = config.X_TRAIN_SCALED_CSV
    if x_path.exists():
        try:
            X_sample = pd.read_csv(x_path).values
        except Exception:
            X_sample = np.zeros((100, len(FEATURE_NAMES)))
    else:
        X_sample = np.random.default_rng(42).random((100, len(FEATURE_NAMES)))

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Batch Profiling", "Single Customer", "Subspace Explorer"],
    )

    if page == "Dashboard":
        render_dashboard(model, scaler, profiles)
    elif page == "Batch Profiling":
        render_batch_profiling(model, scaler, profiles)
    elif page == "Single Customer":
        render_single_customer(model, scaler, profiles)
    elif page == "Subspace Explorer":
        render_subspace_explorer(model, X_sample, FEATURE_NAMES)


if __name__ == "__main__":
    main()
