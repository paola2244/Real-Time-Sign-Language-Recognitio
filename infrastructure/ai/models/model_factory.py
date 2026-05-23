"""
Factory for creating and loading CNN models.

This module provides utilities for creating new models, loading trained models,
and checking model availability.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
from .cnn_model import create_cnn_model
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class ModelFactory:
    """Factory for managing CNN models."""

    @staticmethod
    def create_new_model(
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 24
    ) -> tf.keras.Sequential:
        """
        Create a new untrained CNN model.

        Args:
            input_shape: Input image shape (default: 28x28x1)
            num_classes: Number of output classes (default: 24)

        Returns:
            tf.keras.Sequential: Newly created model

        Example:
            >>> model = ModelFactory.create_new_model()
            >>> model.summary()
        """
        logger.info(f"Creating new CNN model with {num_classes} classes")
        model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
        logger.info(f"Model created with {model.count_params()} parameters")
        return model

    @staticmethod
    def load_model_from_path(model_path: str) -> Optional[tf.keras.Sequential]:
        """
        Load a trained model from file.

        Supports .keras (TensorFlow native), .h5 (HDF5), and .pb (SavedModel) formats.
        Busca en múltiples ubicaciones si la ruta es relativa.

        Args:
            model_path: Path to the saved model file

        Returns:
            tf.keras.Sequential: Loaded model, or None if loading fails
        """
        try:
            model_path = Path(model_path)

            # Si es una ruta absoluta, úsala directamente
            if model_path.is_absolute():
                paths_to_try = [model_path]
            else:
                # Si es relativa, intenta desde múltiples ubicaciones
                # __file__ = .../infrastructure/ai/models/model_factory.py
                # Necesitamos 4 niveles arriba para llegar a la raíz del proyecto
                project_root = Path(__file__).resolve().parent.parent.parent.parent

                paths_to_try = [
                    model_path,  # Directorio actual (cwd)
                    Path.cwd() / model_path,  # Desde cwd explícitamente
                    project_root / model_path,  # Desde raíz del proyecto
                ]

                logger.debug(f"Buscando modelo en: {model_path}")
                logger.debug(f"Rutas a intentar: {paths_to_try}")

            # Intenta encontrar el archivo en las ubicaciones posibles
            found_path = None
            for path in paths_to_try:
                if path.exists():
                    found_path = path
                    break

            if found_path is None:
                logger.error(f"Model file not found: {model_path}")
                logger.debug(f"Búsqueda en: {paths_to_try}")
                return None

            logger.info(f"Loading model from: {found_path}")
            model = tf.keras.models.load_model(str(found_path))
            logger.info(f"Model loaded successfully with {model.count_params()} parameters")
            return model

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None

    @staticmethod
    def check_model_exists(model_path: str) -> bool:
        """
        Check if a trained model file exists.

        Args:
            model_path: Path to the model file

        Returns:
            bool: True if model exists, False otherwise

        Example:
            >>> exists = ModelFactory.check_model_exists('trained_models/best_model.keras')
            >>> if not exists:
            ...     print("Model not found, training required")
        """
        exists = Path(model_path).exists()
        logger.debug(f"Model existence check for {model_path}: {exists}")
        return exists

    @staticmethod
    def save_model(model: tf.keras.Sequential, save_path: str) -> bool:
        """
        Save a trained model to file.

        Args:
            model: Model to save
            save_path: Path where to save the model

        Returns:
            bool: True if save was successful, False otherwise

        Example:
            >>> model = ModelFactory.create_new_model()
            >>> success = ModelFactory.save_model(model, 'trained_models/best_model.keras')
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving model to: {save_path}")
            model.save(str(save_path))
            logger.info(f"Model saved successfully ({save_path.stat().st_size / (1024*1024):.2f} MB)")
            return True

        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            return False

    @staticmethod
    def get_model_info(model_path: str) -> Optional[dict]:
        """
        Get information about a saved model without loading it fully.

        Args:
            model_path: Path to the model file

        Returns:
            dict: Model information, or None if file doesn't exist

        Example:
            >>> info = ModelFactory.get_model_info('trained_models/best_model.keras')
            >>> if info:
            ...     print(f"Model size: {info['file_size_mb']} MB")
        """
        model_path = Path(model_path)

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None

        try:
            file_size = model_path.stat().st_size
            return {
                'path': str(model_path),
                'exists': True,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_path}: {str(e)}")
            return None
