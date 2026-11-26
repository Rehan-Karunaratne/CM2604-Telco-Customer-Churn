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
