from sklearn.metrics import log_loss

def calculate_log_reg_costs(log_reg, X_partial, y_partial, X_val, y_val):
    
    #Calculate training and validation costs for Logistic Regression.
    
    # Calculate training cost
    y_partial_pred_prob_log_reg = log_reg.predict_proba(X_partial)
    training_cost_log_reg = log_loss(y_partial, y_partial_pred_prob_log_reg)

    # Calculate validation cost
    y_val_pred_prob_log_reg = log_reg.predict_proba(X_val)
    validation_cost_log_reg = log_loss(y_val, y_val_pred_prob_log_reg)

    return training_cost_log_reg, validation_cost_log_reg

