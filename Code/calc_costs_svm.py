from sklearn.metrics import log_loss

def calculate_svm_costs(svm_model, X_partial, y_partial, X_val, y_val):
    # Calculate training cost
    y_partial_pred_prob_svm = svm_model.predict_proba(X_partial)
    training_cost_svm = log_loss(y_partial, y_partial_pred_prob_svm)

    # Calculate validation cost
    y_val_pred_prob_svm = svm_model.predict_proba(X_val)
    validation_cost_svm = log_loss(y_val, y_val_pred_prob_svm)

    # Return training and validation cost
    return training_cost_svm, validation_cost_svm