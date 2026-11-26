# getting the required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TASK 1.1 READING THE DATA
# settign the variable to hold the directory of the output files
output_dir = "OutputCharts"
# validating that the file does exsit 
os.makedirs(output_dir, exist_ok=True)
# setting a  standard style for the charts
sns.set_theme(style="whitegrid",palette="pastel")

# getting the file path of the csv file
file_path = os.path.join("ReadingData", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Loading the data from the csv file
df = pd.read_csv(file_path)

#TASK 1.2 INSEPECTION OF DATA
# conducting an inspection of the data to understand the structure and the data types of the columns
print("\n")
print("=" * 50)
print("DATA DIMENSIONS")
# getting the total number of rows in the data frame
print(f"Total Rows of Customers: {df.shape[0]}")
# getting the total number of columns in the data frame
print(f"Total Columns (Features + Target): {df.shape[1]}")
print("=" * 50)

print("\n")
print("=" * 50)
# getting header for the data type insepction 
print("COLUMN NAMES,DATA TYPES AND NULL VALUES")
print("=" * 50)
df.info()

print("\n")
print("=" * 50)
# getting a peak at the first 5 rows of the data frame
print("FIRST 5 ROWS OF THE DATA FRAME")
print("=" * 50)
print(df.head())

# TASK 1.3 DATA CLEANING
print("\n" + "=" * 50)
print("CHECKING FOR DUPLICATE VALUES")
print("=" * 50)
# checking if there are any duplicates
no_of_duplicates = df.duplicated().sum()
# Checks if any customerID appears more than once 
duplicate_ids = df["customerID"].duplicated().sum()
print(f"Exact Duplicate Rows: {no_of_duplicates}")
print(f"Duplicate customerIDs: {duplicate_ids}")

print("\n" + "=" * 50)
print("CHECKING FOR UNIQUE VALUES")
print("=" * 50)
# looping through each collumn 
for col in df.columns:
    # getting the unique values in each column
    unique_values = df[col].nunique()
    if unique_values < 10:
        print(f"Column '{col}' has {unique_values} unique values: {df[col].unique()}")
    else:
        print(f"Column '{col}' has {unique_values} unique values.")

print("\n" + "=" * 50)
print("CHECKING FOR NULL VALUES IN TOTALCHARGES")
print("=" * 50)
# Creates a temporary numeric column; 'errors="coerce"' converts blank spaces (" ") into NaN
df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
# Counts how many hidden missing values (NaN) were revealed by the numeric conversion
hidden_nulls = df["TotalCharges_num"].isnull().sum()
print(f"Hidden missing values found in TotalCharges: {hidden_nulls}")
# Displays those exact rows to prove they all belong to brand-new customers with tenure == 0
print(df[df["TotalCharges_num"].isnull()][["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])
# Overwrites the string 'TotalCharges' column with the clean numeric version for stats and plotting
df["TotalCharges"] = df["TotalCharges_num"]
# Drops the temporary helper column so the DataFrame structure stays clean
df.drop(columns=["TotalCharges_num"], inplace=True)

# TASK 1.4 TARGET CLASS DISTRIBUTION
print("\n" + "=" * 50)
print("TARGET CLASS DISTRIBUTION (CHURN)")
print("=" * 50)
# counting the number of customers who churned vs stayed
churn_counts = df["Churn"].value_counts()
# calculating the percentage of each class to check for class imbalance
churn_pct = df["Churn"].value_counts(normalize=True) * 100

# looping through both classes ('No' and 'Yes') to print counts and percentages
for label in churn_counts.index:
    print(f"Churn = {label}: {churn_counts[label]} customers ({churn_pct[label]:.2f}%)")

# TASK 1.5 NUMERICAL SUMMARY STATISTICS
print("\n" + "=" * 50)
print("NUMERICAL FEATURES SUMMARY STATISTICS")
print("=" * 50)
# listing the three continuous numerical columns in the dataset
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
# printing count, mean, std, min, quartiles, and max rounded to 2 decimal places
print(df[num_cols].describe().round(2))

# TASK 1.6 visualizing the distribution of numerical features
print("\n" + "=" * 50)
print("GENERATING AND SAVING EDA CHARTS")
print("=" * 50)

# setting the figure size for the bar chart
plt.figure(figsize=(6, 4))
# drawing a countplot to compare 'No' vs 'Yes' churn counts
ax = sns.countplot(data=df, x="Churn", hue="Churn", legend=False)
# adding exact customer count labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)
# adding title and axis labels
plt.title("Customer Churn Class Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Customer Churned?")
plt.ylabel("Number of Customers")
# expanding y-axis height slightly so bar labels fit cleanly
plt.ylim(0, churn_counts.max() * 1.12)
plt.tight_layout()
# saving the chart inside the OutputCharts folder
plt.savefig(os.path.join(output_dir, "01_churn_distribution.png"), dpi=300)
plt.close()

# selecting four key categorical features linked to customer churn
cat_features = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
# creating a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# flattening the 2D grid into a 1D list for easy looping
axes = axes.flatten()

# looping through each categorical feature to plot churn breakdown
for idx, col in enumerate(cat_features):
    sns.countplot(data=df, x=col, hue="Churn", ax=axes[idx])
    axes[idx].set_title(f"Churn Breakdown by {col}", fontsize=11, fontweight="bold")
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Customer Count")
    # tilting x-axis labels slightly so long category names do not overlap
    axes[idx].tick_params(axis="x", rotation=15)

plt.tight_layout()
# saving the 4-panel categorical chart inside OutputCharts
plt.savefig(os.path.join(output_dir, "02_categorical_churn_drivers.png"), dpi=300)
plt.close()