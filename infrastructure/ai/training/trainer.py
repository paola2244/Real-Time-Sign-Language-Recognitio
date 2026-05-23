"""
Model training utilities.

This module implements training pipelines, callbacks, and model checkpointing
for the sign language CNN.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from ..models.model_factory import ModelFactory
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class ModelTrainer:
    """Handles model training and checkpointing."""

    @staticmethod
    def create_callbacks(
        model_dir: str = 'trained_models',
        patience: int = 5
    ) -> list:
        """
        Create training callbacks.

        Args:
            model_dir: Directory to save models
            patience: Patience for early stopping

        Returns:
            list: Keras callbacks

        Example:
            >>> callbacks = ModelTrainer.create_callbacks()
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                str(model_dir / 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        logger.info(f"Created callbacks for model checkpointing")
        return callbacks

    @staticmethod
    def train(
        model: tf.keras.Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train model on dataset.

        Args:
            model: Keras model to train
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of callbacks
            verbose: Verbosity level

        Returns:
            History: Training history

        Example:
            >>> history = ModelTrainer.train(model, X_train, y_train, X_val, y_val)
        """
        if callbacks is None:
            callbacks = ModelTrainer.create_callbacks()

        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        logger.info("Training completed")
        return history

    @staticmethod
    def train_with_augmentation(
        model: tf.keras.Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        datagen,
        epochs: int = 20,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train model with data augmentation.

        Args:
            model: Keras model
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            datagen: ImageDataGenerator for augmentation
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of callbacks
            verbose: Verbosity level

        Returns:
            History: Training history

        Example:
            >>> history = ModelTrainer.train_with_augmentation(
            ...     model, X_train, y_train, X_val, y_val,
            ...     datagen, epochs=30
            ... )
        """
        if callbacks is None:
            callbacks = ModelTrainer.create_callbacks()

        logger.info(f"Starting training with augmentation: {epochs} epochs")

        # Create augmented generator
        train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)

        history = model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

        logger.info("Training with augmentation completed")
        return history

    @staticmethod
    def get_history_dict(history: tf.keras.callbacks.History) -> Dict:
        """
        Convert History object to dictionary.

        Args:
            history: Keras History object

        Returns:
            dict: History as dictionary

        Example:
            >>> history_dict = ModelTrainer.get_history_dict(history)
            >>> print(history_dict.keys())
        """
        return history.history

    @staticmethod
    def evaluate(
        model: tf.keras.Sequential,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 1
    ) -> Tuple[float, float]:
        """
        Evaluate model on test set.

        Args:
            model: Keras model
            X_test: Test images
            y_test: Test labels
            verbose: Verbosity level

        Returns:
            Tuple[float, float]: (loss, accuracy)

        Example:
            >>> loss, acc = ModelTrainer.evaluate(model, X_test, y_test)
            >>> print(f"Test accuracy: {acc:.4f}")
        """
        logger.info("Evaluating model on test set")

        loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)

        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        return loss, accuracy

    @staticmethod
    def get_predictions(
        model: tf.keras.Sequential,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Get model predictions.

        Args:
            model: Keras model
            X: Images to predict
            batch_size: Batch size

        Returns:
            np.ndarray: Predictions (N, num_classes)

        Example:
            >>> predictions = ModelTrainer.get_predictions(model, X_test)
        """
        return model.predict(X, batch_size=batch_size, verbose=0)
