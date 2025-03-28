from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    
    Parameters:
    - model: Trained model
    - X_test: Test images
    - y_test: Test labels
    
    Returns:
    - Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Classification report
    report = classification_report(y_test, y_pred_binary)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm
    }