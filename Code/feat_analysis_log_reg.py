import numpy as np
import matplotlib.pyplot as plt
import os

def generate_feat_analysis_log_reg(estimator, X):
    # Feature Analysis
    feature_importance = np.mean(np.abs(estimator.coef_), axis=0)  # Average absolute coefficients across classes

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = X.columns[sorted_indices]
    sorted_importance = feature_importance[sorted_indices]

    # Plot sorted feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance, align="center")
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Analysis (Logisitic Regression)')
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Invert y-axis to display highest importance at the top
    name = "log_reg_feature_analysis.png"
    filepath = os.path.join("./Visuals", name)
    plt.savefig(filepath)