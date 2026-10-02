"""Compress the project into project.zip (run: python make_zip.py)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    "app.py", "requirements.txt", "README.md", "make_zip.py",
    "src/__init__.py", "src/data_loader.py", "src/eda.py", "src/ml_models.py",
    "data/Bank_Customer_Churn_Prediction.csv",
]

with zipfile.ZipFile(ROOT / "project.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in FILES:
        path = ROOT / f
        if path.exists():
            zf.write(path, arcname=f"bank_churn_app/{f}")
            print("added", f)
        else:
            print("skipped (not found):", f)
print("Created project.zip")
