import pandas as pd

# Load dataset
df = pd.read_csv("data/activity_logs.csv")

# 1. Display dataset information
print("\n===== INSIDERWATCH AI =====")
print("EMPLOYEE ACTIVITY ANALYSIS")

print("\n1. Dataset Shape:")
print(df.shape)

# 2. Display first five records
print("\n2. First Five Records:")
print(df.head())

# 3. Check missing values
print("\n3. Missing Values:")
print(df.isnull().sum())

# 4. Dataset statistics
print("\n4. Statistical Summary:")
print(df.describe())

# 5. Count normal and suspicious activities
print("\n5. Activity Classification:")
print(df["label"].value_counts())

# 6. Calculate suspicious activity percentage
suspicious = (df["label"] == 1).sum()
normal = (df["label"] == 0).sum()

total = len(df)

print("\n6. Activity Percentages:")
print("Normal:", round(normal / total * 100, 2), "%")
print("Suspicious:", round(suspicious / total * 100, 2), "%")

# 7. Display suspicious records
print("\n7. Suspicious Activities:")
print(df[df["label"] == 1].head(10))

# 8. Save analysis report
df.describe().to_csv("reports/dataset_summary.csv")

print("\nAnalysis completed successfully!")
print("Report saved in reports folder.")