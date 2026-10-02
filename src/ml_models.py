"""
ml_models.py
============
Model training, evaluation and real-time inference for churn prediction.

Models compared
---------------
* Logistic Regression   (interpretable baseline)
* Random Forest         (non-linear ensemble)
* Gradient Boosting     (extra boosted-tree benchmark)

Class imbalance (~20 % churners) is handled with ``class_weight='balanced'``
for the first two models.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data_loader import (
    apply_scaling,
    encode_single_customer,
    prepare_dataset,
)

RANDOM_STATE = 42
MODEL_COLORS = {
    "Logistic Regression": "#2E86AB",
    "Random Forest": "#E4572E",
    "Gradient Boosting": "#3BB273",
}


def build_models() -> Dict[str, object]:
    """Return fresh, un-fitted estimators."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3, random_state=RANDOM_STATE
        ),
    }


# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #
def train_and_evaluate(
    clean_df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> dict:
    """
    Train every model and evaluate it on the hold-out test set.

    Returns a dictionary ("artifacts") holding fitted models, metrics, confusion
    matrices, ROC data, feature importances and the preprocessing objects needed
    for inference.
    """
    prep = prepare_dataset(clean_df, test_size=test_size, random_state=random_state)
    X_train, X_test = prep["X_train_scaled"], prep["X_test_scaled"]
    y_train, y_test = prep["y_train"], prep["y_test"]

    models = build_models()
    rows, confusions, roc_data = [], {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
            }
        )
        confusions[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": rows[-1]["ROC-AUC"]}

    metrics = pd.DataFrame(rows).set_index("Model")
    best_model = metrics["F1-Score"].idxmax()

    rf = models["Random Forest"]
    importance = (
        pd.DataFrame(
            {"Feature": prep["feature_columns"], "Importance": rf.feature_importances_}
        )
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )

    lr = models["Logistic Regression"]
    coefficients = (
        pd.DataFrame({"Feature": prep["feature_columns"], "Coefficient": lr.coef_[0]})
        .sort_values("Coefficient")
        .reset_index(drop=True)
    )

    return {
        "models": models,
        "metrics": metrics,
        "confusion": confusions,
        "roc": roc_data,
        "importance": importance,
        "coefficients": coefficients,
        "best_model": best_model,
        "scaler": prep["scaler"],
        "feature_columns": prep["feature_columns"],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }


# --------------------------------------------------------------------------- #
# Real-time inference
# --------------------------------------------------------------------------- #
def predict_churn(artifacts: dict, model_name: str, customer: dict) -> dict:
    """
    Predict churn for a single customer.

    *customer* must contain: credit_score, country, gender, age, tenure, balance,
    products_number, credit_card, active_member, estimated_salary.
    """
    encoded = encode_single_customer(customer, artifacts["feature_columns"])
    scaled = apply_scaling(encoded, artifacts["scaler"])
    model = artifacts["models"][model_name]
    probability = float(model.predict_proba(scaled)[0, 1])
    label = int(probability >= 0.5)

    if probability < 0.3:
        risk = "Low"
    elif probability < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "probability": probability,
        "label": label,
        "verdict": "Will Churn" if label else "Will Stay",
        "risk_level": risk,
    }


# --------------------------------------------------------------------------- #
# Plotly figures
# --------------------------------------------------------------------------- #
def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white"
    )
    return fig


def metrics_comparison_bar(metrics: pd.DataFrame) -> go.Figure:
    long = (
        metrics.reset_index()
        .melt(id_vars="Model", var_name="Metric", value_name="Score")
        .round(4)
    )
    fig = px.bar(
        long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        color_discrete_map=MODEL_COLORS,
        title="Model Comparison",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(range=[0, 1.1])
    return _style(fig, 460)


def confusion_matrix_heatmap(cm: np.ndarray, model_name: str) -> go.Figure:
    labels = ["Retained (0)", "Churned (1)"]
    total = cm.sum()
    text = [
        [f"{cm[i, j]}<br>({cm[i, j] / total:.1%})" for j in range(2)] for i in range(2)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16),
            colorscale="Blues",
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
    )
    return _style(fig, 400)


def roc_curve_figure(roc_data: dict) -> go.Figure:
    fig = go.Figure()
    for name, d in roc_data.items():
        fig.add_trace(
            go.Scatter(
                x=d["fpr"],
                y=d["tpr"],
                mode="lines",
                name=f"{name} (AUC = {d['auc']:.3f})",
                line=dict(width=3, color=MODEL_COLORS.get(name)),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random guess",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05),
    )
    return _style(fig, 480)


def feature_importance_figure(importance: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest: Feature Importance",
        color="Importance",
        color_continuous_scale="Tealgrn",
    )
    fig.update_coloraxes(showscale=False)
    return _style(fig, 460)


def coefficient_figure(coefficients: pd.DataFrame) -> go.Figure:
    df = coefficients.copy()
    df["Direction"] = np.where(df["Coefficient"] > 0, "Raises churn risk", "Lowers churn risk")
    fig = px.bar(
        df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Raises churn risk": "#E4572E",
            "Lowers churn risk": "#2E86AB",
        },
        title="Logistic Regression: Standardised Coefficients",
    )
    return _style(fig, 460)


def probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#222"},
                "steps": [
                    {"range": [0, 30], "color": "#9BDEAC"},
                    {"range": [30, 60], "color": "#FFE08A"},
                    {"range": [60, 100], "color": "#F4978E"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
            },
        )
    )
    return _style(fig, 320)
