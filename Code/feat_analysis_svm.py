from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_svm_feat_analysis(svm_model, X, y, feature_names):
    """
    Generate feature analysis for an SVM model using permutation importance.
    """
    # Calculate permutation importance
    perm_importance = permutation_importance(svm_model, X, y, scoring="accuracy", n_repeats=30, random_state=42)
    
    # Extract feature importance and sort by importance
    importance = perm_importance.importances_mean
    indices = np.argsort(importance)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_feature_names, sorted_importance, align="center")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Analysis (SVM)")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    name = "svm_feature_analysis.png"
    filepath = os.path.join("./Visuals", name)
    plt.savefig(filepath)