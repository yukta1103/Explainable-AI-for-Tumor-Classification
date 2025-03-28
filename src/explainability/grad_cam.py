import numpy as np
import tf_explain

def generate_grad_cam(model, img, layer_name):
    """
    Generate Grad-CAM heatmap for model explanation
    
    Parameters:
    - model: Trained Keras model
    - img: Input image
    - layer_name: Last convolutional layer name
    
    Returns:
    - Grad-CAM heatmap
    """
    grad_cam = tf_explain.core.GradCAM()
    explanation = grad_cam.explain(
        validation_data=(np.array([img]), np.array([0])), 
        model=model, 
        layer_name=layer_name
    )
    return explanation