"""
eda.py
======
Exploratory Data Analysis helpers: summary statistics and interactive Plotly
figures. Every plotting function returns a plotly Figure so that the Streamlit
layer (app.py) stays free of plotting logic.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import TARGET

COLOR_MAP = {"Retained": "#2E86AB", "Churned": "#E4572E"}
_LABELS = {0: "Retained", 1: "Churned"}


def _with_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Status"] = out[TARGET].map(_LABELS)
    return out


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
        template="plotly_white",
    )
    return fig


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def key_indicators(df: pd.DataFrame) -> dict:
    """Headline numbers shown as metric cards."""
    return {
        "customers": int(len(df)),
        "churn_rate": float(df[TARGET].mean() * 100),
        "avg_age": float(df["age"].mean()),
        "avg_balance": float(df["balance"].mean()),
        "avg_salary": float(df["estimated_salary"].mean()),
        "avg_credit_score": float(df["credit_score"].mean()),
        "zero_balance_pct": float((df["balance"] == 0).mean() * 100),
        "active_pct": float(df["active_member"].mean() * 100),
    }


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe(): adds skewness and median to the usual statistics."""
    numeric = df.select_dtypes("number")
    stats = numeric.describe().T
    stats["median"] = numeric.median()
    stats["skew"] = numeric.skew()
    return stats.round(2)


def churn_by_group(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate (%) and customer count for every value of *column*."""
    grouped = (
        df.groupby(column)[TARGET]
        .agg(customers="count", churned="sum", churn_rate="mean")
        .reset_index()
    )
    grouped["churn_rate"] = (grouped["churn_rate"] * 100).round(2)
    return grouped


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def churn_distribution_pie(df: pd.DataFrame) -> go.Figure:
    counts = df[TARGET].map(_LABELS).value_counts().reset_index()
    counts.columns = ["Status", "Customers"]
    fig = px.pie(
        counts,
        names="Status",
        values="Customers",
        hole=0.45,
        color="Status",
        color_discrete_map=COLOR_MAP,
        title="Churn Distribution",
    )
    fig.update_traces(textinfo="percent+label+value")
    return _style(fig)


def balance_vs_churn_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        _with_label(df),
        x="Status",
        y="balance",
        color="Status",
        color_discrete_map=COLOR_MAP,
        points="outliers",
        title="Account Balance vs Churn",
        labels={"balance": "Balance"},
    )
    return _style(fig)


def age_vs_churn_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        _with_label(df),
        x="age",
        color="Status",
        barmode="overlay",
        nbins=40,
        opacity=0.7,
        marginal="box",
        color_discrete_map=COLOR_MAP,
        title="Age Distribution by Churn Status",
        labels={"age": "Age"},
    )
    return _style(fig, 480)


def numeric_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Violin plot of any numeric column split by churn status."""
    fig = px.violin(
        _with_label(df),
        x="Status",
        y=column,
        color="Status",
        box=True,
        color_discrete_map=COLOR_MAP,
        title=f"{column.replace('_', ' ').title()} vs Churn",
    )
    return _style(fig)


def churn_rate_bar(df: pd.DataFrame, column: str) -> go.Figure:
    """Churn rate (%) per category / discrete value of *column*."""
    data = churn_by_group(df, column)
    data[column] = data[column].astype(str)
    fig = px.bar(
        data,
        x=column,
        y="churn_rate",
        text="churn_rate",
        color="churn_rate",
        color_continuous_scale="OrRd",
        hover_data=["customers", "churned"],
        title=f"Churn Rate (%) by {column.replace('_', ' ').title()}",
        labels={"churn_rate": "Churn rate (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(range=[0, max(data["churn_rate"].max() * 1.25, 5)])
    return _style(fig, 380)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes("number").corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )
    return _style(fig, 600)


def scatter_age_balance(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        _with_label(df),
        x="age",
        y="balance",
        color="Status",
        opacity=0.55,
        color_discrete_map=COLOR_MAP,
        title="Age vs Balance (coloured by churn)",
    )
    return _style(fig, 480)


def outlier_comparison_box(raw: pd.DataFrame, clean: pd.DataFrame, column: str) -> go.Figure:
    """Side-by-side boxplots of a column before and after outlier capping."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=raw[column], name="Before cleaning", marker_color="#E4572E"))
    fig.add_trace(go.Box(y=clean[column], name="After cleaning", marker_color="#2E86AB"))
    fig.update_layout(title=f"Outlier Handling: {column.replace('_', ' ').title()}")
    return _style(fig, 380)
