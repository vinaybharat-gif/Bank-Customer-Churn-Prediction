# 🏦 Bank Customer Churn Prediction: Streamlit Analytics App

An end-to-end data analytics and machine learning web application that explores
bank customer data, compares classification models and predicts in real time
whether a customer will **churn** or **stay**.

**Tech stack:** Python · Streamlit · Pandas · NumPy · Scikit-Learn · Plotly

## Project Structure

```
bank_churn_app/
├── app.py                  # Streamlit UI and sidebar navigation
├── requirements.txt        # Python dependencies
├── README.md
├── make_zip.py             # Packs the project into project.zip
├── data/
│   └── Bank_Customer_Churn_Prediction.csv
└── src/
    ├── __init__.py
    ├── data_loader.py      # Loading, cleaning, encoding, scaling
    ├── eda.py              # Plotly EDA figures and summary statistics
    └── ml_models.py        # Training, evaluation, plots, live inference
```

## Setup and Run

Requires **Python 3.9 or newer**.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at <http://localhost:8501>. The first visit to the model pages takes a
few seconds while the models train; results are then cached.

If the CSV is not in `data/`, you can upload it from the sidebar (**Data source**).

## Application Pages

| Page | What it shows |
|------|---------------|
| 🏠 Overview & Business Problem | Problem statement, KPIs, workflow, data dictionary |
| 🧹 Data Cleaning & Raw Data | Raw data, missing/duplicate/outlier handling, encoding and scaling |
| 📊 Exploratory Data Analysis | Summary statistics, churn pie, balance box plot, age distribution, correlation heatmap, churn by category |
| 🤖 ML Model Performance | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curves, feature importance |
| 🔮 Live Customer Churn Predictor | Sidebar inputs give an instant prediction, probability gauge and retention tips |

## Methodology

1. **Cleaning:** drop `customer_id`, remove duplicates, impute missing values (median / mode),
   cap outliers with the IQR rule.
2. **Encoding:** one-hot encoding of `country` and `gender` (`drop_first=True`).
3. **Scaling:** `StandardScaler` on continuous features, fitted on the training split only.
4. **Split:** 80 / 20 stratified train/test split (`random_state=42`).
5. **Models:** Logistic Regression, Random Forest and Gradient Boosting, with class-weight
   balancing for the first two because only about 20 % of customers churn.
6. **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix, ROC curve.
   The best model is selected by F1-Score, a better fit than accuracy on imbalanced data.

## Packaging for Submission

```bash
python make_zip.py
```

This creates `project.zip` containing all project files.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: run `streamlit run app.py` from inside the project folder.
- **Port already in use**: `streamlit run app.py --server.port 8502`.
- **Different column names** (e.g. Kaggle's `Geography`, `Exited`): handled automatically.
