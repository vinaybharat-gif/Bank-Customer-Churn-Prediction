"""
app.py
======
Bank Customer Churn Prediction: Streamlit web application.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import eda
from src import ml_models as mm

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "Bank_Customer_Churn_Prediction.csv"

PAGES = [
    "🏠 Overview & Business Problem",
    "🧹 Data Cleaning & Raw Data",
    "📊 Exploratory Data Analysis (EDA)",
    "🤖 ML Model Performance & Comparison",
    "🔮 Live Customer Churn Predictor",
]

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached data / model helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading dataset...")
def get_raw_data(uploaded_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_bytes is not None:
        return dl.load_raw(io.BytesIO(uploaded_bytes))
    return dl.load_raw(DATA_PATH)


@st.cache_data(show_spinner="Cleaning data...")
def get_clean_data(raw: pd.DataFrame):
    return dl.clean_data(raw)


@st.cache_resource(show_spinner="Training models (first run only)...")
def get_artifacts(clean: pd.DataFrame) -> dict:
    return mm.train_and_evaluate(clean)


# --------------------------------------------------------------------------- #
# Sidebar: navigation + data source
# --------------------------------------------------------------------------- #
st.sidebar.title("🏦 Churn Analytics")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

with st.sidebar.expander("📁 Data source", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload a different CSV (optional)", type=["csv"],
        help="Default: data/Bank_Customer_Churn_Prediction.csv",
    )
uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

try:
    raw_df = get_raw_data(uploaded_bytes)
except FileNotFoundError:
    st.error(
        f"Dataset not found at `{DATA_PATH}`. Place **Bank_Customer_Churn_Prediction.csv** "
        "inside the `data/` folder or upload it from the sidebar."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001 - show any parsing / schema problem
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

clean_df, clean_report = get_clean_data(raw_df)
st.sidebar.divider()


# --------------------------------------------------------------------------- #
# Page 1: Overview
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.title("🏠 Bank Customer Churn Prediction")
    st.caption("An end-to-end data analytics & machine learning web application")

    st.subheader("Business Problem")
    st.markdown(
        """
        **Customer churn** occurs when a customer closes their account and leaves the bank.
        Acquiring a new customer typically costs several times more than retaining an existing
        one, so identifying customers who are *likely to leave* gives the bank a chance to
        intervene with targeted offers, better service or product bundles.

        **Objective:** build a classification model that predicts whether a customer will
        **churn (1)** or **stay (0)** from demographic and account information, and explain
        the main drivers of churn.
        """
    )

    ind = eda.key_indicators(clean_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{ind['customers']:,}")
    c2.metric("Churn Rate", f"{ind['churn_rate']:.1f}%")
    c3.metric("Average Age", f"{ind['avg_age']:.1f} yrs")
    c4.metric("Average Balance", f"{ind['avg_balance']:,.0f}")

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(eda.churn_distribution_pie(clean_df))
    with right:
        st.subheader("Project Workflow")
        st.markdown(
            """
            1. **Data Cleaning**: remove IDs, handle missing values, duplicates and outliers
            2. **EDA**: interactive Plotly visualisations and statistics
            3. **Preprocessing**: one-hot encoding + `StandardScaler`
            4. **Modelling**: Logistic Regression vs Random Forest (+ Gradient Boosting)
            5. **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC
            6. **Deployment**: live churn predictor for individual customers
            """
        )
        st.subheader("Tech Stack")
        st.markdown("`Python` · `Streamlit` · `Pandas` · `NumPy` · `Scikit-Learn` · `Plotly`")

    st.subheader("Data Dictionary")
    st.dataframe(
        pd.DataFrame(
            {
                "Column": [
                    "credit_score", "country", "gender", "age", "tenure", "balance",
                    "products_number", "credit_card", "active_member",
                    "estimated_salary", "churn",
                ],
                "Description": [
                    "Customer credit score",
                    "Country of residence (France / Germany / Spain)",
                    "Male / Female",
                    "Age in years",
                    "Years as a bank customer",
                    "Account balance",
                    "Number of bank products held",
                    "Has a credit card (1 = yes)",
                    "Is an active member (1 = yes)",
                    "Estimated annual salary",
                    "TARGET: 1 = churned, 0 = retained",
                ],
            }
        ),
        hide_index=True,
    )


# --------------------------------------------------------------------------- #
# Page 2: Data cleaning
# --------------------------------------------------------------------------- #
def page_cleaning() -> None:
    st.title("🧹 Data Cleaning & Raw Data")

    st.subheader("1. Raw Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing values", int(raw_df.isna().sum().sum()))
    c4.metric("Duplicate rows", int(raw_df.duplicated().sum()))
    st.dataframe(raw_df.head(100), height=300)

    with st.expander("Column types & missing values per column"):
        info = pd.DataFrame(
            {
                "dtype": raw_df.dtypes.astype(str),
                "missing": raw_df.isna().sum(),
                "unique values": raw_df.nunique(),
            }
        )
        st.dataframe(info)

    st.subheader("2. Cleaning Report")
    r = clean_report
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows before", f"{r['rows_before']:,}")
    c2.metric("Rows after", f"{r['rows_after']:,}")
    c3.metric("Duplicates removed", r["duplicates_removed"] + r["duplicate_customer_ids_removed"])
    c4.metric("Missing after cleaning", r["missing_after"])
    st.markdown(
        f"- **Dropped unused columns:** `{', '.join(r['dropped_columns']) or 'none'}`\n"
        f"- **Missing values imputed:** "
        f"{r['missing_before'] if r['missing_before'] else 'none found'} "
        "(median for numeric, mode for categorical)\n"
        f"- **Rows dropped for missing target:** {r['rows_dropped_missing_target']}\n"
        "- **Outliers:** capped using the IQR rule (Q1 − 1.5·IQR, Q3 + 1.5·IQR), "
        "so no rows are lost"
    )
    outlier_table = pd.DataFrame(r["outliers"]).T
    outlier_table.columns = ["Lower bound", "Upper bound", "Outliers capped"]
    st.dataframe(outlier_table)

    column = st.selectbox("Visualise outlier handling for:", dl.OUTLIER_COLUMNS)
    st.plotly_chart(eda.outlier_comparison_box(raw_df, clean_df, column))

    st.subheader("3. Cleaned Dataset")
    st.dataframe(clean_df.head(100), height=300)
    st.download_button(
        "⬇️ Download cleaned CSV",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="bank_churn_cleaned.csv",
        mime="text/csv",
    )

    st.subheader("4. Encoding & Feature Scaling")
    st.markdown(
        "`country` and `gender` are **one-hot encoded** (first category dropped); the "
        "continuous features are standardised with **`StandardScaler`** "
        "(fitted on the training split only, to avoid data leakage)."
    )
    prep = dl.prepare_dataset(clean_df)
    t1, t2 = st.tabs(["Encoded (unscaled)", "Encoded + scaled (training set)"])
    with t1:
        st.dataframe(prep["X_train"].head(50))
    with t2:
        st.dataframe(prep["X_train_scaled"].head(50).round(3))
    st.caption(
        f"Train / test split: {len(prep['X_train']):,} / {len(prep['X_test']):,} "
        "rows (80 / 20, stratified on churn)."
    )


# --------------------------------------------------------------------------- #
# Page 3: EDA
# --------------------------------------------------------------------------- #
def page_eda() -> None:
    st.title("📊 Exploratory Data Analysis")

    ind = eda.key_indicators(clean_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn rate", f"{ind['churn_rate']:.1f}%")
    c2.metric("Avg credit score", f"{ind['avg_credit_score']:.0f}")
    c3.metric("Active members", f"{ind['active_pct']:.1f}%")
    c4.metric("Zero-balance accounts", f"{ind['zero_balance_pct']:.1f}%")

    with st.expander("📋 Statistical summary", expanded=True):
        st.dataframe(eda.summary_statistics(clean_df))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Churn & Balance", "Age & Numeric Features", "Categorical Drivers", "Correlations"]
    )

    with tab1:
        a, b = st.columns(2)
        a.plotly_chart(eda.churn_distribution_pie(clean_df))
        b.plotly_chart(eda.balance_vs_churn_box(clean_df))

    with tab2:
        st.plotly_chart(eda.age_vs_churn_distribution(clean_df))
        a, b = st.columns(2)
        feature = a.selectbox(
            "Compare a numeric feature by churn status",
            [c for c in dl.NUMERIC_FEATURES if c != "age"],
        )
        a.plotly_chart(eda.numeric_distribution(clean_df, feature))
        b.plotly_chart(eda.scatter_age_balance(clean_df))

    with tab3:
        a, b = st.columns(2)
        a.plotly_chart(eda.churn_rate_bar(clean_df, "country"))
        b.plotly_chart(eda.churn_rate_bar(clean_df, "gender"))
        a, b = st.columns(2)
        a.plotly_chart(eda.churn_rate_bar(clean_df, "products_number"))
        b.plotly_chart(eda.churn_rate_bar(clean_df, "active_member"))

    with tab4:
        st.plotly_chart(eda.correlation_heatmap(clean_df))
        corr = clean_df.select_dtypes("number").corr()[dl.TARGET].drop(dl.TARGET)
        st.markdown(
            f"**Strongest positive correlation with churn:** `{corr.idxmax()}` "
            f"({corr.max():+.2f})  \n"
            f"**Strongest negative correlation with churn:** `{corr.idxmin()}` "
            f"({corr.min():+.2f})"
        )

    st.subheader("💡 Key Insights")
    country = eda.churn_by_group(clean_df, "country").sort_values("churn_rate").iloc[-1]
    gender = eda.churn_by_group(clean_df, "gender").sort_values("churn_rate").iloc[-1]
    active = eda.churn_by_group(clean_df, "active_member").set_index("active_member")["churn_rate"]
    older = clean_df[clean_df["age"] >= 50][dl.TARGET].mean() * 100
    younger = clean_df[clean_df["age"] < 40][dl.TARGET].mean() * 100
    st.markdown(
        f"- Customers in **{country['country']}** churn the most ({country['churn_rate']:.1f}%).\n"
        f"- **{gender['gender']}** customers churn more ({gender['churn_rate']:.1f}%).\n"
        f"- Customers aged 50+ churn at **{older:.1f}%** versus **{younger:.1f}%** for those under 40.\n"
        f"- Inactive members churn at **{active.get(0, float('nan')):.1f}%** versus "
        f"**{active.get(1, float('nan')):.1f}%** for active members."
    )


# --------------------------------------------------------------------------- #
# Page 4: Model performance
# --------------------------------------------------------------------------- #
def page_models() -> None:
    st.title("🤖 ML Model Performance & Comparison")
    art = get_artifacts(clean_df)

    st.markdown(
        f"Models are trained on **{art['train_size']:,}** customers and evaluated on a "
        f"hold-out set of **{art['test_size']:,}** customers (stratified 80/20 split). "
        "Class imbalance is handled with balanced class weights."
    )

    st.subheader("Metrics Comparison")
    metrics = art["metrics"]
    st.dataframe(
        metrics.style.format("{:.4f}").highlight_max(axis=0, color="#CDEFD2")
    )
    st.success(f"🏆 Best model by F1-Score: **{art['best_model']}**")
    st.plotly_chart(mm.metrics_comparison_bar(metrics))

    st.subheader("Confusion Matrix")
    model_name = st.selectbox(
        "Select model", list(art["models"].keys()),
        index=list(art["models"].keys()).index(art["best_model"]),
    )
    cm = art["confusion"][model_name]
    tn, fp, fn, tp = cm.ravel()
    a, b = st.columns([3, 2])
    a.plotly_chart(mm.confusion_matrix_heatmap(cm, model_name))
    with b:
        st.markdown("**How to read it**")
        st.markdown(
            f"- True Negatives (stayed, predicted stay): **{tn}**\n"
            f"- False Positives (stayed, predicted churn): **{fp}**\n"
            f"- False Negatives (churned, predicted stay): **{fn}**\n"
            f"- True Positives (churned, predicted churn): **{tp}**"
        )
        st.caption(
            "For churn, **recall** matters most: a false negative is a customer "
            "the bank fails to try to save."
        )

    st.subheader("ROC Curve")
    st.plotly_chart(mm.roc_curve_figure(art["roc"]))

    st.subheader("What drives churn?")
    a, b = st.columns(2)
    a.plotly_chart(mm.feature_importance_figure(art["importance"]))
    b.plotly_chart(mm.coefficient_figure(art["coefficients"]))


# --------------------------------------------------------------------------- #
# Page 5: Live predictor
# --------------------------------------------------------------------------- #
def page_predictor() -> None:
    st.title("🔮 Live Customer Churn Predictor")
    art = get_artifacts(clean_df)

    sb = st.sidebar
    sb.header("Customer Details")
    model_name = sb.selectbox(
        "Model", list(art["models"].keys()),
        index=list(art["models"].keys()).index(art["best_model"]),
    )
    credit_score = sb.slider("Credit score", 300, 900, int(clean_df["credit_score"].median()))
    country = sb.selectbox("Country", sorted(clean_df["country"].unique()))
    gender = sb.selectbox("Gender", sorted(clean_df["gender"].unique()))
    age = sb.slider("Age", 18, 100, int(clean_df["age"].median()))
    tenure = sb.slider("Tenure (years)", 0, 10, int(clean_df["tenure"].median()))
    balance = sb.number_input(
        "Account balance", min_value=0.0, max_value=1_000_000.0,
        value=float(round(clean_df["balance"].median(), 2)), step=1000.0,
    )
    products_number = sb.slider("Number of products", 1, 4, 1)
    credit_card = sb.radio("Has credit card?", ["Yes", "No"], horizontal=True)
    active_member = sb.radio("Active member?", ["Yes", "No"], horizontal=True)
    estimated_salary = sb.number_input(
        "Estimated salary", min_value=0.0, max_value=1_000_000.0,
        value=float(round(clean_df["estimated_salary"].median(), 2)), step=1000.0,
    )

    customer = {
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": int(credit_card == "Yes"),
        "active_member": int(active_member == "Yes"),
        "estimated_salary": estimated_salary,
    }
    result = mm.predict_churn(art, model_name, customer)

    left, right = st.columns([1, 1])
    with left:
        if result["label"] == 1:
            st.error(f"### ⚠️ Prediction: {result['verdict']}")
        else:
            st.success(f"### ✅ Prediction: {result['verdict']}")
        st.metric("Churn probability", f"{result['probability']:.1%}")
        st.metric("Risk level", result["risk_level"])
        st.caption(f"Model used: {model_name}. Decision threshold: 50%.")
    with right:
        st.plotly_chart(mm.probability_gauge(result["probability"]))

    st.subheader("Customer Profile")
    profile = pd.DataFrame([customer]).T.rename(columns={0: "Value"})
    profile["Value"] = profile["Value"].astype(str)
    st.dataframe(profile)

    if result["label"] == 1 or result["risk_level"] != "Low":
        st.subheader("Retention Suggestions")
        tips = []
        if active_member == "No":
            tips.append("Re-engage this inactive member with personalised offers or outreach.")
        if products_number == 1 or products_number >= 3:
            tips.append("Review the product mix: customers with 1 or 3+ products churn more often.")
        if country == "Germany":
            tips.append("Germany shows elevated churn: consider a regional retention campaign.")
        if age >= 45:
            tips.append("Offer age-appropriate products such as savings or retirement plans.")
        if not tips:
            tips.append("Schedule a proactive relationship-manager call.")
        for tip in tips:
            st.markdown(f"- {tip}")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if page == PAGES[0]:
    page_overview()
elif page == PAGES[1]:
    page_cleaning()
elif page == PAGES[2]:
    page_eda()
elif page == PAGES[3]:
    page_models()
else:
    page_predictor()
