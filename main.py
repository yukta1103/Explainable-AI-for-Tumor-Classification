import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.load_data import load_dataset
from src.data.preprocessing import create_data_augmentation, preprocess_data
from src.models.architectures import create_transfer_learning_model
from src.models.train import train_model
from src.evaluation.metrics import evaluate_model
from src.explainability.grad_cam import generate_grad_cam
from src.explainability.shap_explanation import calculate_shap_values

def main():
    # Data loading
    data_path = 'path/to/dataset'
    X, y = load_dataset(data_path)
    
    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Data augmentation
    datagen = create_data_augmentation()
    
    # Model creation
    model = create_transfer_learning_model('resnet')
    
    # Training
    history = train_model(
        model, 
        X_train, y_train, 
        X_test, y_test, 
        datagen
    )
    
    # Evaluation
    evaluation = evaluate_model(model, X_test, y_test)
    print(evaluation['classification_report'])
    
    # Explainability
    grad_cam_result = generate_grad_cam(model, X_test[0], 'last_conv_layer')
    shap_values = calculate_shap_values(model, X_test)
    
    # Optional: Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Grad-CAM Visualization')
    plt.imshow(grad_cam_result)
    plt.subplot(1, 2, 2)
    plt.title('Original Image')
    plt.imshow(X_test[0])
    plt.show()

if __name__ == '__main__':
    main()