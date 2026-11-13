import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
df = pd.read_csv("data/activity_logs.csv")

# Create reports directory if missing
Path("reports").mkdir(exist_ok=True)

# Chart 1: Normal vs suspicious activities
activity_counts = df["label"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(
    ["Normal", "Suspicious"],
    [activity_counts.get(0, 0), activity_counts.get(1, 0)]
)
plt.title("Normal vs Suspicious Activities")
plt.xlabel("Activity Type")
plt.ylabel("Number of Records")
plt.savefig("reports/activity_classification.png")
plt.show()

# Chart 2: Login hour distribution
plt.figure(figsize=(8, 5))

for label, name in [(0, "Normal"), (1, "Suspicious")]:
    subset = df[df["label"] == label]
    plt.hist(
        subset["login_hour"],
        bins=24,
        range=(0, 24),
        alpha=0.5,
        label=name
    )

plt.title("Employee Login Hour Distribution")
plt.xlabel("Login Hour (0-23)")
plt.ylabel("Number of Records")
plt.legend()
plt.savefig("reports/login_distribution.png")
plt.show()

# Chart 3: File download comparison
plt.figure(figsize=(7, 5))

download_data = [
    df[df["label"] == 0]["files_downloaded"],
    df[df["label"] == 1]["files_downloaded"]
]

plt.boxplot(
    download_data,
    tick_labels=["Normal", "Suspicious"]
)

plt.title("File Download Behaviour")
plt.xlabel("Activity Type")
plt.ylabel("Files Downloaded")
plt.savefig("reports/download_comparison.png")
plt.show()

# Chart 4: Employees with most suspicious records
employee_risks = (
    df[df["label"] == 1]
    .groupby("employee_id")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(9, 5))
plt.bar(employee_risks.index, employee_risks.values)

plt.title("Top 10 Employees by Suspicious Record Count")
plt.xlabel("Employee ID")
plt.ylabel("Suspicious Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/employee_activity.png")
plt.show()

print("All visualizations generated successfully!")
print("Check the reports folder.")