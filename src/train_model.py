import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed_activity_logs.csv"
MODEL_FILE = ROOT / "models" / "insiderwatch_model.joblib"
REPORT_FILE = ROOT / "reports" / "model_predictions.csv"

# Create required directories
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)

print("=" * 50)
print("INSIDERWATCH AI - MODEL TRAINING")
print("=" * 50)

# Load processed dataset
df = pd.read_csv(DATA_FILE)

# Select model features
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

X = df[feature_columns]
y = df["label"].astype(int)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Create Isolation Forest model
model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42
)

# Train using activity features only
model.fit(X_train)

print("\nAI model training completed!")

# Predict test records
predictions = model.predict(X_test)

# Convert predictions:
# Isolation Forest: -1 = anomaly, 1 = normal
# Our labels: 1 = suspicious, 0 = normal

predicted_labels = (predictions == -1).astype(int)

# Evaluate model
print("\nMODEL EVALUATION")
print("-" * 40)

print(
    classification_report(
        y_test,
        predicted_labels,
        labels=[0, 1],
        target_names=["Normal", "Suspicious"],
        zero_division=0
    )
)

print("Accuracy:", round(
    accuracy_score(y_test, predicted_labels) * 100, 2
), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted_labels))

# Calculate anomaly scores
anomaly_scores = -model.decision_function(X_test)

# Save predictions
results = pd.DataFrame({
    "employee_id": df.loc[X_test.index, "employee_id"].values,
    "actual_label": y_test.values,
    "predicted_label": predicted_labels,
    "anomaly_score": anomaly_scores
})

results["detection_result"] = results[
    "predicted_label"
].map({
    0: "Normal",
    1: "Anomaly"
})

results.to_csv(REPORT_FILE, index=False)

# Save trained model
joblib.dump(model, MODEL_FILE)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Model:", MODEL_FILE)
print("Prediction report:", REPORT_FILE)

print("\nINSIDERWATCH AI MODEL READY!")