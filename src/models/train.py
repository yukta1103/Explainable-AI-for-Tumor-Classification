def train_model(model, X_train, y_train, X_val, y_val, datagen, epochs=50):
    """
    Train the model with data augmentation
    
    Parameters:
    - model: Compiled Keras model
    - X_train: Training images
    - y_train: Training labels
    - X_val: Validation images
    - y_val: Validation labels
    - datagen: Data augmentation generator
    - epochs: Number of training epochs
    
    Returns:
    - Training history
    """
    # Train model with data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(X_val, y_val),
        # Optional: Early stopping
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
        ]
    )
    
    return history