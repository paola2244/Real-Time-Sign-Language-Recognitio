"""
Inference agent for CNN model predictions.

This module handles low-level model inference, executing predictions
on preprocessed images and returning probability distributions.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple
from ..models.model_factory import ModelFactory
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class InferenceAgent:
    """
    Agent responsible for executing CNN model inference.

    Attributes:
        model: TensorFlow/Keras model
        model_path: Path to the saved model
    """

    def __init__(self, model_path: str):
        """
        Initialize inference agent with a trained model.

        Args:
            model_path: Path to the trained model file

        Raises:
            RuntimeError: If model cannot be loaded

        Example:
            >>> agent = InferenceAgent('trained_models/best_model.keras')
            >>> probabilities = agent.predict(preprocessed_image)
        """
        self.model_path = model_path
        self.model = ModelFactory.load_model_from_path(model_path)

        if self.model is None:
            logger.error(f"Failed to load model from {model_path}")
            raise RuntimeError(f"Model not found or cannot be loaded: {model_path}")

        logger.info(f"InferenceAgent initialized with model: {model_path}")

    def predict(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Execute model inference on preprocessed input.

        Supports both:
        - Image tensors: (1, 28, 28, 1) for CNN model
        - Landmark tensors: (1, 63) for Hybrid/Dense models

        Args:
            input_tensor: Preprocessed tensor

        Returns:
            Tuple[np.ndarray, bool]:
                - Probability array (shape: num_classes)
                - Success flag (True if prediction succeeded)

        Example:
            >>> image_tensor = np.random.rand(1, 28, 28, 1).astype(np.float32)
            >>> probs, success = agent.predict(image_tensor)
            >>> if success:
            ...     top_class = np.argmax(probs)
            ...     confidence = probs[top_class]
        """
        try:
            # Validate input shape - accept both 2D (landmarks) and 4D (images)
            if len(input_tensor.shape) not in [2, 4]:
                logger.warning(f"Invalid tensor shape: {input_tensor.shape}. Expected (batch, features) or (batch, height, width, channels)")
                return np.array([]), False

            # Run prediction
            predictions = self.model.predict(input_tensor, verbose=0)

            # Return probabilities for the first (and only) image in batch
            return predictions[0], True

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return np.array([]), False

    def batch_predict(self, image_tensors: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Execute model inference on multiple images.

        Args:
            image_tensors: Batch of preprocessed tensors (batch_size, 28, 28, 1)

        Returns:
            Tuple[np.ndarray, bool]:
                - Predictions array (shape: batch_size, num_classes)
                - Success flag

        Example:
            >>> batch = np.random.rand(32, 28, 28, 1).astype(np.float32)
            >>> predictions, success = agent.batch_predict(batch)
            >>> if success:
            ...     top_classes = np.argmax(predictions, axis=1)
        """
        try:
            predictions = self.model.predict(image_tensors, verbose=0)
            return predictions, True

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return np.array([]), False

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            dict: Model information including layers, parameters, etc.

        Example:
            >>> info = agent.get_model_info()
            >>> print(f"Total parameters: {info['total_parameters']}")
        """
        return {
            'model_name': self.model.name,
            'total_layers': len(self.model.layers),
            'total_parameters': int(self.model.count_params()),
            'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'model_path': self.model_path
        }

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, bool]:
        """
        Evaluate model on test set.

        Args:
            x_test: Test images (batch_size, 28, 28, 1)
            y_test: Test labels (batch_size, num_classes) - one-hot encoded

        Returns:
            Tuple[float, float, bool]:
                - Test loss
                - Test accuracy
                - Success flag

        Example:
            >>> loss, accuracy, success = agent.evaluate(x_test, y_test)
            >>> if success:
            ...     print(f"Test accuracy: {accuracy:.4f}")
        """
        try:
            loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
            return loss, accuracy, True

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return 0.0, 0.0, False

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"InferenceAgent(model_path='{self.model_path}')"
