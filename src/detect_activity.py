import pandas as pd
import joblib

from pathlib import Path
from datetime import datetime

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "models" / "insiderwatch_model.joblib"
REPORT_FILE = ROOT / "reports" / "security_alerts.csv"

# ==========================================
# LOAD AI MODEL
# ==========================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

model = joblib.load(MODEL_FILE)

print("=" * 55)
print("             INSIDERWATCH AI")
print("          THREAT DETECTION ENGINE")
print("=" * 55)

# ==========================================
# EMPLOYEE ACTIVITY INPUT
# ==========================================

activity = {
    "employee_id": "EMP051",
    "login_hour": 3,
    "failed_logins": 7,
    "files_accessed": 200,
    "files_downloaded": 350,
    "usb_usage": 1,
    "external_access": 1
}

employee_id = activity["employee_id"]

# ==========================================
# INPUT VALIDATION
# ==========================================

if not 0 <= activity["login_hour"] <= 23:
    raise ValueError("Login hour must be between 0 and 23.")

for column in [
    "failed_logins",
    "files_accessed",
    "files_downloaded"
]:
    if activity[column] < 0:
        raise ValueError(f"{column} cannot be negative.")

for column in ["usb_usage", "external_access"]:
    if activity[column] not in [0, 1]:
        raise ValueError(f"{column} must be 0 or 1.")

# ==========================================
# FEATURE ENGINEERING
# ==========================================

activity["after_hours_login"] = int(
    activity["login_hour"] < 7 or
    activity["login_hour"] >= 20
)

activity["high_download_volume"] = int(
    activity["files_downloaded"] >= 100
)

activity["high_file_access"] = int(
    activity["files_accessed"] >= 100
)

activity["multiple_login_failures"] = int(
    activity["failed_logins"] >= 5
)

activity["usb_external_combination"] = int(
    activity["usb_usage"] == 1 and
    activity["external_access"] == 1
)

# ==========================================
# MODEL FEATURES
# ==========================================

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

features = pd.DataFrame(
    [[activity[col] for col in feature_columns]],
    columns=feature_columns
)

# ==========================================
# AI ANOMALY DETECTION
# ==========================================

prediction = model.predict(features)[0]

anomaly_score = -model.decision_function(features)[0]

if prediction == -1:
    ai_result = "ANOMALOUS ACTIVITY"
else:
    ai_result = "NORMAL ACTIVITY"

# ==========================================
# RISK SCORING
# ==========================================

risk_score = 0
reasons = []

if activity["after_hours_login"]:
    risk_score += 15
    reasons.append("Employee logged in outside normal hours")

if activity["multiple_login_failures"]:
    risk_score += 15
    reasons.append("Multiple failed login attempts detected")

if activity["high_file_access"]:
    risk_score += 15
    reasons.append("Unusually high number of files accessed")

if activity["high_download_volume"]:
    risk_score += 25
    reasons.append("High volume of file downloads detected")

if activity["usb_usage"] == 1:
    risk_score += 10
    reasons.append("USB device usage detected")

if activity["external_access"] == 1:
    risk_score += 5
    reasons.append("External network access detected")

if activity["usb_external_combination"]:
    risk_score += 15
    reasons.append("USB usage combined with external access")

risk_score = min(risk_score, 100)

# ==========================================
# SEVERITY CLASSIFICATION
# ==========================================

if risk_score >= 60:
    severity = "HIGH"

elif risk_score >= 30:
    severity = "MEDIUM"

else:
    severity = "LOW"

# ==========================================
# ALERT STATUS
# ==========================================

if prediction == -1 or risk_score >= 30:
    alert_status = "ALERT"
else:
    alert_status = "NORMAL"

if prediction == -1:
    reasons.append("AI model identified an anomalous activity pattern")

if not reasons:
    reasons.append("No configured suspicious activity rules triggered")

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nEMPLOYEE INFORMATION")
print("-" * 40)
print("Employee ID:", employee_id)

print("\nAI DETECTION RESULT")
print("-" * 40)
print("Prediction:", ai_result)
print("Anomaly Score:", round(float(anomaly_score), 4))

print("\nSECURITY RISK ANALYSIS")
print("-" * 40)
print("Risk Score:", risk_score, "/ 100")
print("Severity:", severity)
print("Alert Status:", alert_status)

print("\nREASONS FOR DETECTION")
print("-" * 40)

for number, reason in enumerate(reasons, start=1):
    print(f"{number}. {reason}")

# ==========================================
# CREATE SECURITY REPORT
# ==========================================

result = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "employee_id": employee_id,
    "ai_prediction": ai_result,
    "anomaly_score": round(float(anomaly_score), 4),
    "risk_score": risk_score,
    "severity": severity,
    "alert_status": alert_status,
    "detection_reasons": "; ".join(reasons),
    **activity
}

new_report = pd.DataFrame([result])

# ==========================================
# SAVE SECURITY REPORT
# ==========================================

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

if REPORT_FILE.exists():
    old_report = pd.read_csv(REPORT_FILE)
    combined_report = pd.concat(
        [old_report, new_report],
        ignore_index=True,
        sort=False
    )
else:
    combined_report = new_report

combined_report.to_csv(REPORT_FILE, index=False)

print("\n" + "=" * 55)
print("SECURITY REPORT SAVED SUCCESSFULLY!")
print("Report:", REPORT_FILE)
print("Total Report Records:", len(combined_report))
print("=" * 55)

print("\nINSIDERWATCH AI DETECTION COMPLETED!")