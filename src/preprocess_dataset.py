import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "activity_logs.csv"
OUTPUT_FILE = ROOT / "data" / "processed_activity_logs.csv"
FEATURE_FILE = ROOT / "data" / "model_features.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("INSIDERWATCH AI")
print("FEATURE ENGINEERING AND PREPROCESSING")
print("-" * 45)

# Required columns
required_columns = [
    "employee_id",
    "login_hour",
    "failed_logins",
    "files_accessed",
    "files_downloaded",
    "usb_usage",
    "external_access",
    "label"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Handle missing values
numeric_columns = [
    "login_hour",
    "failed_logins",
    "files_accessed",
    "files_downloaded",
    "usb_usage",
    "external_access"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df["employee_id"] = df["employee_id"].fillna("UNKNOWN")

# Validate activity values
if not df["label"].isin([0, 1]).all():
    raise ValueError("Label must contain only 0 or 1")

# Feature engineering
# These thresholds are for our synthetic dataset only.

df["after_hours_login"] = (
    (df["login_hour"] < 7) |
    (df["login_hour"] >= 20)
).astype(int)

df["high_download_volume"] = (
    df["files_downloaded"] >= 100
).astype(int)

df["high_file_access"] = (
    df["files_accessed"] >= 100
).astype(int)

df["multiple_login_failures"] = (
    df["failed_logins"] >= 5
).astype(int)

df["usb_external_combination"] = (
    (df["usb_usage"] == 1) &
    (df["external_access"] == 1)
).astype(int)

# Features used by the AI model
feature_columns = [
    "login_hour",
    "failed_logins",
    "files_accessed",
    "files_downloaded",
    "usb_usage",
    "external_access",
    "after_hours_login",
    "high_download_volume",
    "high_file_access",
    "multiple_login_failures",
    "usb_external_combination"
]

# Check for invalid values
# Check for missing or infinite values
if not np.isfinite(df[feature_columns].to_numpy()).all():
    raise ValueError("Invalid or missing values found")

print("Data validation successful!")
# Save processed data, including labels for later evaluation
df.to_csv(OUTPUT_FILE, index=False)

# Save model features separately.
# Employee ID and label are deliberately excluded.
df[feature_columns].to_csv(FEATURE_FILE, index=False)

print("\nPreprocessing completed!")
print("Total records:", len(df))
print("Original columns:", len(required_columns))
print("Model features:", len(feature_columns))

print("\nEngineered feature preview:")
print(df[feature_columns].head())

print("\nFiles created:")
print(OUTPUT_FILE)
print(FEATURE_FILE)