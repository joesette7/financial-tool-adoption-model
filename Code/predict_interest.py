# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from corr_heatmap import generate_corr_heatmap
from learning_curves import generate_learning_curves
from roc_curves import generate_roc_curves
from misclassifications import generate_misclassification_diagram
from feat_analysis_log_reg import generate_feat_analysis_log_reg
from feat_analysis_svm import generate_svm_feat_analysis

# ──────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS DATA
# ──────────────────────────────────────────────────────────────────────────────

data = pd.read_csv("./Data/pop_sample.csv")

# Drop target feature that's too predictive (avoid leakage)
data = data.drop(columns='Interest_AI_Financial_Management')

# One-hot encode categorical variables
categorical_columns = [
    'Age', 'Education', 'Employment', 'Set_Financial_Goals', 'Online_Banking_Usage',
    'Physical_Bank_Visits', 'App_Usage_Weekly', 'Use_Budgeting_Tools', 
    'Interest_Personalized_Insights_Recommendations', 'Interest_Budget_Automation', 
    'Data_Security_Concerns', 'AI_Concerns'
]
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

# ──────────────────────────────────────────────────────────────────────────────
# SCALE ONLY CONTINUOUS FEATURES
# ──────────────────────────────────────────────────────────────────────────────
numeric_features = ['Satisfaction_Financial_Management', 'Comfort_with_Tech']
scaler = StandardScaler()
data_encoded[numeric_features] = scaler.fit_transform(data[numeric_features])

# Optional: export encoded data for inspection
# data_encoded.to_csv("./Data/encoded_data.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# FEATURES / TARGET
# ──────────────────────────────────────────────────────────────────────────────
generate_corr_heatmap(data_encoded)  # visualize correlations

target_columns = [c for c in data_encoded.columns if 'Interest_Budget_Automation' in c]
X = data_encoded.drop(columns=target_columns)
y = data_encoded['Interest_Budget_Automation_Yes']  # binary target

# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# (CV done on TRAIN only; evaluate on VAL and TEST)
# ──────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Logistic Regression Grid Search
log_base = LogisticRegression(
    solver='liblinear',          # supports l1 and l2
    class_weight='balanced',
    max_iter=5000,
    random_state=42
)
log_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
log_search = GridSearchCV(
    estimator=log_base,
    param_grid=log_grid,
    scoring='f1',                # positive class = 1
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)
log_search.fit(X_train, y_train)
best_log = log_search.best_estimator_
print("\n[LogReg] Best Params:", log_search.best_params_)
print("[LogReg] Best CV F1:", round(log_search.best_score_, 4))

# ---- SVM Grid Search
svc_base = SVC(
    probability=True,            # needed for ROC curves
    class_weight='balanced',
    random_state=42
)
svc_grid = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01]  # ignored when kernel='linear'
}
svc_search = GridSearchCV(
    estimator=svc_base,
    param_grid=svc_grid,
    scoring='f1',                # positive class = 1
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)
svc_search.fit(X_train, y_train)
best_svm = svc_search.best_estimator_
print("\n[SVM] Best Params:", svc_search.best_params_)
print("[SVM] Best CV F1:", round(svc_search.best_score_, 4))

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION: VALIDATION & TEST
# ──────────────────────────────────────────────────────────────────────────────
# Logistic Regression
y_val_pred = best_log.predict(X_val)
y_test_pred = best_log.predict(X_test)

print("\nValidation Set Classification Report (Logistic Regression, tuned):")
print(classification_report(y_val, y_val_pred))

print("\nTest Set Classification Report (Logistic Regression, tuned):")
print(classification_report(y_test, y_test_pred))

# SVM
y_val_pred_svm = best_svm.predict(X_val)
y_test_pred_svm = best_svm.predict(X_test)

print("\nValidation Set Classification Report (SVM, tuned):")
print(classification_report(y_val, y_val_pred_svm, zero_division=0))

print("\nTest Set Classification Report (SVM, tuned):")
print(classification_report(y_test, y_test_pred_svm, zero_division=0))

# Class balance check
print("\nClass Distribution:")
print("Training:", np.bincount(y_train))
print("Validation:", np.bincount(y_val))
print("Testing:", np.bincount(y_test))

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS & VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
# Feature importance (LogReg)
generate_feat_analysis_log_reg(best_log, X)

# Feature analysis (SVM)
generate_svm_feat_analysis(best_svm, X, y, X.columns)

# ROC curves comparing tuned models
generate_roc_curves(best_log, best_svm, X_val, y_val)

# Misclassification comparison on TEST
generate_misclassification_diagram(y_test, y_test_pred, y_test_pred_svm)

# Learning curves (train vs. val) using tuned models
generate_learning_curves(best_log, best_svm, X_train, y_train, X_val, y_val)
