from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_misclassification_diagram(y_test, y_test_pred, y_test_pred_svm):
    # Get misclassified indices for Logistic Regression
    log_reg_misclassified = set(np.where(y_test != y_test_pred)[0])

    # Get misclassified indices for SVM
    svm_misclassified = set(np.where(y_test != y_test_pred_svm)[0])

    # Venn diagram of misclassifications
    plt.figure(figsize=(8, 6))
    venn = venn2([log_reg_misclassified, svm_misclassified], 
                set_labels=("Logistic Regression", "SVM"),
                set_colors=('blue', 'orange'))
    plt.title("Venn Diagram of Misclassifications")
    name = "misclassification_venn_diagram.png"
    filepath = os.path.join("./Visuals", name)
    plt.savefig(filepath)