from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from calc_costs_svm import calculate_svm_costs
from calc_costs_log_reg import calculate_log_reg_costs
import os

def generate_learning_curves(log_reg, svm_model, X_train, y_train, X_val, y_val):
    
    # Plot learning curves for training and validation costs.
    
    
    training_costs_log_reg = []
    validation_costs_log_reg = []
    training_costs_svm = []
    validation_costs_svm = []

    for size in np.linspace(0.1, 1.0, 10):
        if size == 1.0:
            X_partial, y_partial = X_train, y_train  # Use the entire dataset for training
        else:
            X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=float(size), random_state=42)
    
        # Logistic Regression Costs
        log_reg.fit(X_partial, y_partial)
        training_cost_log_reg, validation_cost_log_reg = calculate_log_reg_costs(log_reg, X_partial, y_partial, X_val, y_val)
        training_costs_log_reg.append(training_cost_log_reg)
        validation_costs_log_reg.append(validation_cost_log_reg)

        # SVM Costs
        svm_model.fit(X_partial, y_partial)
        training_cost_svm, validation_cost_svm = calculate_svm_costs(svm_model, X_partial, y_partial, X_val, y_val)
        training_costs_svm.append(training_cost_svm)
        validation_costs_svm.append(validation_cost_svm)
    
    # Plot the learning curve for Logistic Regression
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0.1, 1.0, 10), training_costs_log_reg, label="Training Cost (Logistic Regression)", marker='o')
    plt.plot(np.linspace(0.1, 1.0, 10), validation_costs_log_reg, label="Validation Cost (Logistic Regression)", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Cost (Log Loss)")
    plt.title("Learning Curve (Logistic Regression)")
    plt.legend()
    plt.grid()
    name = "learning_curve_log_reg.png"
    filepath = os.path.join("./Visuals", name)
    plt.savefig(filepath)

    # Plot the learning curve for SVM
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0.1, 1.0, 10), training_costs_svm, label="Training Cost (SVM)", marker='o')
    plt.plot(np.linspace(0.1, 1.0, 10), validation_costs_svm, label="Validation Cost (SVM)", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Cost (Log Loss)")
    plt.title("Learning Curve (SVM)")
    plt.legend()
    plt.grid()
    name = "learning_curve_svm.png"
    filepath = os.path.join("./Visuals", name)
    plt.savefig(filepath)