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


# ==================== HYBRID MODEL FOR LANDMARKS ====================

def create_hybrid_model(input_features: int = 63, num_classes: int = 28) -> Sequential:
    """
    Create Hybrid CNN-Dense model for sign language landmarks recognition.

    Architecture for 3D landmark coordinates (21 landmarks × 3 coords = 63 features):
    - Input: (63,) - flattened 21 landmarks with x, y, z coordinates
    - Reshape: (21, 3, 1) - structured as 21 time steps, 3D coordinates
    - Conv1D blocks: Extract spatial patterns from sequential landmarks
    - Dense layers: Final classification

    Layer structure:
    - Input (63,)
    - Reshape (21, 3, 1)
    - Conv1D(32, 3) + BatchNorm + MaxPool(2) + Dropout(0.3)
    - Conv1D(64, 3) + BatchNorm + MaxPool(2) + Dropout(0.3)
    - Flatten
    - Dense(128) + BatchNorm + Dropout(0.4)
    - Dense(64) + Dropout(0.3)
    - Dense(num_classes, Softmax)

    Args:
        input_features: Number of input features (default: 63 for 21 landmarks × 3)
        num_classes: Number of output classes (default: 28 for A-Z + del + space)

    Returns:
        tensorflow.keras.Sequential: Compiled model ready for training

    Example:
        >>> model = create_hybrid_model(input_features=63, num_classes=28)
        >>> model.summary()
        >>> predictions = model.predict(landmarks)  # Shape: (batch_size, 28)
    """
    model = Sequential([
        # Input and reshape
        layers.Input(shape=(input_features,)),
        layers.Reshape((21, 3), name='reshape_landmarks'),

        # Conv1D Block 1: Extract local patterns
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv1d_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.MaxPooling1D(pool_size=2, name='max_pool_1'),
        layers.Dropout(0.3, name='dropout_1'),

        # Conv1D Block 2: Extract more complex patterns
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling1D(pool_size=2, name='max_pool_2'),
        layers.Dropout(0.3, name='dropout_2'),

        # Flatten to 1D
        layers.Flatten(name='flatten'),

        # Dense Block 1: Feature integration
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Dropout(0.4, name='dropout_3'),

        # Dense Block 2: Classification
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_4'),

        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='SignLanguageHybridModel')

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_hybrid_model_v2(input_features: int = 63, num_classes: int = 28) -> Sequential:
    """
    Alternative Hybrid model with more Conv1D layers for deeper feature extraction.

    More aggressive feature extraction with 3 Conv1D blocks.

    Args:
        input_features: Number of input features (default: 63)
        num_classes: Number of output classes (default: 28)

    Returns:
        tensorflow.keras.Sequential: Compiled model

    Example:
        >>> model = create_hybrid_model_v2(input_features=63, num_classes=28)
    """
    model = Sequential([
        # Input and reshape
        layers.Input(shape=(input_features,)),
        layers.Reshape((21, 3), name='reshape_landmarks'),

        # Conv1D Block 1
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv1d_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(0.25, name='dropout_1'),

        # Conv1D Block 2
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.MaxPooling1D(pool_size=2, name='max_pool_1'),
        layers.Dropout(0.25, name='dropout_2'),

        # Conv1D Block 3
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv1d_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.MaxPooling1D(pool_size=2, name='max_pool_2'),
        layers.Dropout(0.3, name='dropout_3'),

        # Flatten
        layers.Flatten(name='flatten'),

        # Dense layers
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='batch_norm_4'),
        layers.Dropout(0.5, name='dropout_4'),

        layers.Dense(128, activation='relu', name='dense_2'),
        layers.Dropout(0.3, name='dropout_5'),

        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='SignLanguageHybridModelV2')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_dense_model(input_features: int = 63, num_classes: int = 28) -> Sequential:
    """
    Pure Dense model for landmarks (alternative to Hybrid/Conv1D).

    For comparison - a simpler architecture with only Dense layers.

    Args:
        input_features: Number of input features (default: 63)
        num_classes: Number of output classes (default: 28)

    Returns:
        tensorflow.keras.Sequential: Compiled model

    Example:
        >>> model = create_dense_model(input_features=63, num_classes=28)
    """
    model = Sequential([
        # Input
        layers.Input(shape=(input_features,)),

        # Dense blocks
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Dropout(0.4, name='dropout_1'),

        layers.Dense(128, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(0.3, name='dropout_2'),

        layers.Dense(64, activation='relu', name='dense_3'),
        layers.Dropout(0.2, name='dropout_3'),

        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='SignLanguageDenseModel')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
