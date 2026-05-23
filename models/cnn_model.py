"""
CNN model architecture for sign language recognition.

This module implements a deep convolutional neural network specifically designed
for recognizing hand signs in 28x28 grayscale images (Sign Language MNIST).
"""

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from typing import Tuple


def create_cnn_model(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 24) -> Sequential:
    """
    Create and compile CNN model for sign language recognition.

    Architecture:
    - Input: 28x28x1 grayscale images
    - Block 1: Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
    - Block 2: Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
    - Block 3: Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
    - Dense layers: 512 (Dropout 0.5) -> 256 (Dropout 0.3) -> 24 classes (Softmax)

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (default: 24 for A-X letters)

    Returns:
        tensorflow.keras.Sequential: Compiled model ready for training

    Example:
        >>> model = create_cnn_model(input_shape=(28, 28, 1), num_classes=24)
        >>> model.summary()
        >>> predictions = model.predict(images)  # Shape: (batch_size, 24)
    """
    model = Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling2D((2, 2), name='max_pool_1'),
        layers.Dropout(0.25, name='dropout_1'),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_4'),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.MaxPooling2D((2, 2), name='max_pool_2'),
        layers.Dropout(0.25, name='dropout_2'),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_5'),
        layers.BatchNormalization(name='batch_norm_5'),
        layers.MaxPooling2D((2, 2), name='max_pool_3'),
        layers.Dropout(0.25, name='dropout_3'),

        # Flatten and Dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='batch_norm_6'),
        layers.Dropout(0.5, name='dropout_4'),

        layers.Dense(256, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_5'),

        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='SignLanguageCNN')

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary(model: Sequential) -> str:
    """
    Get detailed model architecture summary.

    Args:
        model: Keras Sequential model

    Returns:
        str: Model summary as string

    Example:
        >>> model = create_cnn_model()
        >>> summary = get_model_summary(model)
        >>> print(summary)
    """
    model.summary()
    return "Model summary printed to console"


def get_model_config(model: Sequential) -> dict:
    """
    Get model configuration for serialization.

    Args:
        model: Keras Sequential model

    Returns:
        dict: Model configuration

    Example:
        >>> config = get_model_config(model)
    """
    return {
        'name': model.name,
        'layers': len(model.layers),
        'parameters': model.count_params(),
        'optimizer': model.optimizer.get_config() if model.optimizer else None,
        'loss': model.loss,
        'metrics': model.metrics_names
    }
