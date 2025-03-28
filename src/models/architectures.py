import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, VGG16

def create_transfer_learning_model(base_model_name='resnet'):
    """
    Create a transfer learning model for tumor classification
    
    Parameters:
    - base_model_name: Choose between 'resnet' or 'vgg'
    
    Returns:
    - Compiled Keras model
    """
    # Select base model
    if base_model_name.lower() == 'resnet':
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
    else:
        base_model = VGG16(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model