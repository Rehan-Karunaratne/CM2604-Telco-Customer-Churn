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


