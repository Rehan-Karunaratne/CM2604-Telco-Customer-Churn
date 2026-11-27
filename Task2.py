# importing standard libraries for file paths, numerical arrays, tables, and charts
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing scikit-learn tools for splitting, cross-validation, pipelines, scaling, and encoding
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# importing the three exact classification algorithms required by the coursework brief
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# importing evaluation metrics to measure and compare binary classification performance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# setting the folder name where evaluation plots will be saved
output_dir = "OutputCharts"
# creating the folder if it does not already exist so saving never crashes
os.makedirs(output_dir, exist_ok=True)
# setting a consistent visual theme for all evaluation charts
sns.set_theme(style="whitegrid", palette="pastel")


# TASK 2.1 DATA LOADING AND DATASET-LEVEL CLEANING
print("\n" + "=" * 50)
print("TASK 2.1 DATA LOADING AND CLEANING")
print("=" * 50)

# building a cross-platform path to the CSV inside ReadingData
file_path = os.path.join("ReadingData", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
# loading the raw dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# dropping 'customerID' because it is a unique random ID with zero predictive value
df.drop(columns=["customerID"], inplace=True)

# converting 'TotalCharges' from string to float; blank spaces become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# filling the 11 NaN values with 0.0 (verified in Task 1 as tenure == 0 new customers)
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

# converting the target column 'Churn' from text ('Yes'/'No') to binary integers (1/0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# separating input features (X) from the target variable (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Features shape (X): {X.shape}")
print(f"Target shape (y)  : {y.shape}")


# TASK 2.2 STRATIFIED TRAIN-TEST SPLIT
print("\n" + "=" * 50)
print("TASK 2.2 STRATIFIED TRAIN-TEST SPLIT (80% TRAIN / 20% TEST)")
print("=" * 50)

# splitting data into 80% training and 20% testing BEFORE scaling or encoding
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")



# TASK 2.3 PREPROCESSING PIPELINE (SCALING & ONE-HOT ENCODING)
print("\n" + "=" * 50)
print("TASK 2.3 BUILDING PREPROCESSING PIPELINE")
print("=" * 50)

# defining the three continuous numerical columns that need feature scaling
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
# selecting all remaining categorical columns that need One-Hot Encoding
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# combining StandardScaler (for numerical) and OneHotEncoder (for categorical) into one transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

print(f"Numerical features scaled   ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical features encoded ({len(categorical_cols)}): {len(categorical_cols)} columns")


# HELPER FUNCTION TO EVALUATE MODELS CONSISTENTLY
def evaluate_model(model, X_test_data, y_test_data, model_name, setting_name):
    # generating binary predictions (0 = Stay, 1 = Churn) on the unseen test set
    y_pred = model.predict(X_test_data)
    # generating predicted probabilities for class 1 ('Churn') needed to compute ROC-AUC
    y_prob = model.predict_proba(X_test_data)[:, 1]

    # returning a dictionary containing all key binary classification metrics
    return {
        "Model": model_name,
        "Setting": setting_name,
        "Accuracy": round(accuracy_score(y_test_data, y_pred), 4),
        "Precision": round(precision_score(y_test_data, y_pred), 4),
        "Recall": round(recall_score(y_test_data, y_pred), 4),
        "F1-Score": round(f1_score(y_test_data, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test_data, y_prob), 4),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# creating a master list to store results from both experimental settings
results_list = []


# TASK 2.4 EXPERIMENTAL SETTING 1: BASELINE MODELS (DEFAULT HYPERPARAMETERS)
print("\n" + "=" * 50)
print("TASK 2.4 EXPERIMENTAL SETTING 1: BASELINE MODELS")
print("=" * 50)

# defining the three required classifiers using their default scikit-learn settings
baseline_models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

# looping through each baseline model to train and evaluate it on the test set
for name, clf in baseline_models.items():
    # chaining the preprocessor and classifier into one Pipeline to prevent data leakage
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    # fitting the preprocessing and model on the 80% training data only
    pipe.fit(X_train, y_train)
    # evaluating the fitted baseline model on the unseen 20% test data
    res = evaluate_model(pipe, X_test, y_test, name, "Setting 1: Baseline")
    # saving the metrics dictionary into our master results list
    results_list.append(res)
    print(
        f"[Baseline] {name:20s} | Accuracy: {res['Accuracy']:.4f} | "
        f"Precision: {res['Precision']:.4f} | Recall: {res['Recall']:.4f} | "
        f"F1: {res['F1-Score']:.4f} | ROC-AUC: {res['ROC-AUC']:.4f}"
    )

# TASK 2.5 HYPERPARAMETER TUNING (GRIDSEARCHCV) & EXPERIMENTAL SETTING 2
print("\n" + "=" * 50)
print("TASK 2.5 HYPERPARAMETER TUNING & SETTING 2: TUNED MODELS")
print("=" * 50)

# defining 5-fold stratified cross-validation so every fold keeps the 26.54% churn ratio
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# defining hyperparameter search grids tailored to each of the three required algorithms
param_grids = {
    "KNN": {
        # testing odd neighbor counts (avoids ties), uniform vs distance weighting, and Euclidean vs Manhattan distance
        "classifier__n_neighbors": [5, 11, 17, 23],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    },
    "Logistic Regression": {
        # testing regularization strength C, L1 vs L2 penalty, and balanced class weights for churn imbalance
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear"],
        "classifier__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        # testing number of trees, maximum tree depth (controls overfitting), min samples to split, and class weighting
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [6, 10, None],
        "classifier__min_samples_split": [2, 10],
        "classifier__class_weight": [None, "balanced"],
    },
}

# looping through each algorithm to run GridSearchCV and evaluate the best tuned model
for name, clf in baseline_models.items():
    print(f"\nTuning {name}...")
    # creating a fresh pipeline so preprocessing fits independently inside each CV fold
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    # running exhaustive grid search across 5 stratified folds optimizing F1-score
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        cv=cv_strategy,
        scoring="f1",
        n_jobs=-1,
    )
    # fitting grid search strictly on the 80% training set (never touching X_test)
    grid.fit(X_train, y_train)
    # extracting the winning pipeline with the best hyperparameter combination
    best_pipe = grid.best_estimator_

    # printing the optimal hyperparameters and best cross-validated F1-score
    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Best 5-Fold CV F1-Score: {grid.best_score_:.4f}")

    # evaluating the tuned model on the unseen 20% test set (Experimental Setting 2)
    res = evaluate_model(best_pipe, X_test, y_test, name, "Setting 2: Tuned")
    # appending Setting 2 results to our master comparison list
    results_list.append(res)

# TASK 2.6 FINAL COMPARISON TABLE & EVALUATION CHARTS
print("\n" + "=" * 50)
print("TASK 2.6 FINAL COMPARISON: SETTING 1 (BASELINE) VS SETTING 2 (TUNED)")
print("=" * 50)

# converting the results list into a clean DataFrame and dropping raw prediction arrays for display
results_df = pd.DataFrame(results_list).drop(columns=["y_pred", "y_prob"])
# printing the full side-by-side comparison table across all models and both settings
print(results_df.to_string(index=False))



# creating a 1x3 grid of subplots to display confusion matrices for all 3 tuned models
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
# filtering the master results list to isolate Setting 2 (Tuned) models
tuned_results = [r for r in results_list if r["Setting"] == "Setting 2: Tuned"]

# looping through each tuned model to plot its 2x2 confusion matrix
for idx, res in enumerate(tuned_results):
    # computing True Negatives, False Positives, False Negatives, and True Positives on X_test
    cm = confusion_matrix(y_test, res["y_pred"])
    # drawing a heatmap with exact integer counts inside each cell
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted: No", "Predicted: Yes"],
        yticklabels=["Actual: No", "Actual: Yes"],
        ax=axes[idx],
    )
    axes[idx].set_title(f"{res['Model']} (Tuned)", fontsize=11, fontweight="bold")

plt.tight_layout()
# saving the confusion matrices figure inside OutputCharts at 300 DPI
plt.savefig(os.path.join(output_dir, "05_tuned_confusion_matrices.png"), dpi=300)
plt.close()



# creating a 1x2 subplot to compare ROC curves between Setting 1 and Setting 2
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# looping through both experimental settings to plot all 3 models on each panel
for setting_idx, setting_label in enumerate(["Setting 1: Baseline", "Setting 2: Tuned"]):
    # filtering results belonging to the current experimental setting
    subset = [r for r in results_list if r["Setting"] == setting_label]
    for res in subset:
        # computing False Positive Rate (FPR) and True Positive Rate (TPR) across probability thresholds
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[setting_idx].plot(
            fpr, tpr, lw=2, label=f"{res['Model']} (AUC = {res['ROC-AUC']:.4f})"
        )
    # plotting the diagonal dashed line representing a random 50/50 guesser (AUC = 0.50)
    axes[setting_idx].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[setting_idx].set_title(f"ROC Curves - {setting_label}", fontsize=11, fontweight="bold")
    axes[setting_idx].set_xlabel("False Positive Rate")
    axes[setting_idx].set_ylabel("True Positive Rate (Recall)")
    axes[setting_idx].legend(loc="lower right")

plt.tight_layout()
# saving the ROC curve comparison chart inside OutputCharts at 300 DPI
plt.savefig(os.path.join(output_dir, "06_roc_curves_comparison.png"), dpi=300)
plt.close()

print(f"\nTask 2 Complete! Evaluation charts saved inside '{output_dir}/'.")