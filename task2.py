import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create student performance dataset with duplicates and missing values
data = {
    "StudentID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 111],
    "Gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Female"],
    "StudyHours": [6.5, 4.0, 8.0, 3.5, 9.0, 5.0, 7.5, 2.0, 8.5, 6.0, 7.0, 7.0],
    "Attendance(%)": [90, 80, 95, 75, 98, 85, 92, 70, 96, 88, 91, 91],
    "Marks": [78, 65, np.nan, 55, 92, 70, 88, np.nan, 95, 76, 85, 85]
}

pd.DataFrame(data).to_csv("student_performance.csv", index=False)

# (a) Load CSV, print shape and info
df = pd.read_csv("student_performance.csv")
print("Shape:", df.shape)
print("\n--- Dataset Info ---")
df.info()

# (b) Missing values and duplicate rows check
print("\nMissing values in Marks:", df["Marks"].isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Clean data: drop duplicate and fill missing Marks with column median
df_clean = df.drop_duplicates().copy()
df_clean["Marks"].fillna(df_clean["Marks"].median(), inplace=True)
print("After cleaning:", df_clean.shape)

# (c) Plot Marks histogram and boxplot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df_clean["Marks"], bins=6, color='steelblue', edgecolor='black')
plt.title("Marks Histogram")
plt.xlabel("Marks")

plt.subplot(1, 2, 2)
plt.boxplot(df_clean["Marks"])
plt.title("Marks Boxplot")

plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.savefig("marks_boxplot.png")

# (d) Scatter plot StudyHours vs Marks & Correlation Matrix
plt.figure(figsize=(6, 4))
plt.scatter(df_clean["StudyHours"], df_clean["Marks"], color='purple')
plt.title("StudyHours vs Marks")
plt.xlabel("StudyHours")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")

corr_matrix = df_clean[["StudyHours", "Attendance(%)", "Marks"]].corr()
print("\nCorrelation matrix:\n", corr_matrix)

# (e) Categorical summary by Gender
gender_summary = df_clean.groupby("Gender")[["Marks", "Attendance(%)"]].mean()
print("\nAverage Marks and Attendance by Gender:\n", gender_summary)

print("\nSaved marks_histogram.png, marks_boxplot.png, studyhours_vs_marks.png")