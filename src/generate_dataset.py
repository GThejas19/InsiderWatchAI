import pandas as pd
import numpy as np

# Reproducible sample dataset
np.random.seed(42)

number_of_records = 1000

employees = [
    f"EMP{i:03d}" for i in range(1, 51)
]

data = []

for i in range(number_of_records):

    employee = np.random.choice(employees)

    # Normal employee activity
    login_hour = np.random.choice(
        range(8, 19)
    )

    failed_logins = np.random.randint(0, 3)

    files_accessed = np.random.randint(1, 30)

    files_downloaded = np.random.randint(0, 15)

    usb_usage = np.random.choice([0, 1], p=[0.9, 0.1])

    external_access = np.random.choice([0, 1], p=[0.95, 0.05])

    label = 0

    # Inject simulated suspicious activities
    if np.random.random() < 0.08:

        login_hour = np.random.choice([0, 1, 2, 3, 4])

        failed_logins = np.random.randint(5, 12)

        files_accessed = np.random.randint(100, 300)

        files_downloaded = np.random.randint(100, 500)

        usb_usage = 1

        external_access = 1

        label = 1

    data.append([
        employee,
        login_hour,
        failed_logins,
        files_accessed,
        files_downloaded,
        usb_usage,
        external_access,
        label
    ])

columns = [
    "employee_id",
    "login_hour",
    "failed_logins",
    "files_accessed",
    "files_downloaded",
    "usb_usage",
    "external_access",
    "label"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv("data/activity_logs.csv", index=False)

print("Dataset generated successfully!")
print("Total records:", len(df))
print("\nDataset preview:")
print(df.head())

print("\nActivity distribution:")
print(df["label"].value_counts())