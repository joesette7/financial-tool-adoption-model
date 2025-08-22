import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def generate_roc_curves(log_reg, svm_model, X_val, y_val):
    #Plots ROC curves for multiple models (e.g., SVM and Logistic Regression).
    
    models = {'Logistic Regression': log_reg, 'SVM': svm_model}
    plt.figure(figsize=(8, 6))
    for model_name, model in models.items():
        # Get prediction probabilities
        if hasattr(model, "predict_proba"):  # For Logistic Regression
            y_prob = model.predict_proba(X_val)[:, 1]
        else:  # For SVM (decision function needs to be scaled)
            y_prob = model.decision_function(X_val)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Plot the diagonal line (no-skill classifier)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    name = "roc_curves.png"
    filepath = os.path.join("./Visuals", name)
    plt.savefig(filepath)