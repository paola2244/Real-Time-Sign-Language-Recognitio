"""
Dataset loader for Sign Language MNIST.

This module handles loading the Sign Language MNIST dataset from CSV files,
splitting it into train/test sets, and providing batch iteration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class DatasetLoader:
    """Loader for Sign Language MNIST dataset."""

    @staticmethod
    def load_csv(csv_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load sign language MNIST from CSV file.

        CSV format: first column is label, remaining 784 columns are pixel values (0-255).

        Args:
            csv_path: Path to CSV file

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Images array (N, 784)
                - Labels array (N,)
                Or None if loading fails

        Example:
            >>> images, labels = DatasetLoader.load_csv('data/raw/sign_mnist_train.csv')
            >>> print(f"Loaded {len(images)} images")
        """
        try:
            logger.info(f"Loading dataset from {csv_path}")
            df = pd.read_csv(csv_path)

            labels = df.iloc[:, 0].values.astype(np.int32)
            images = df.iloc[:, 1:].values.astype(np.uint8)

            logger.info(f"Loaded {len(images)} images, {len(np.unique(labels))} classes")
            return images, labels

        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None

    @staticmethod
    def load_and_preprocess(
        csv_path: str,
        normalize: bool = True
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load CSV and preprocess images.

        Args:
            csv_path: Path to CSV file
            normalize: Normalize pixel values to [0, 1]

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Preprocessed images (N, 28, 28, 1)
                - Labels (N,)
                Or None if loading fails

        Example:
            >>> images, labels = DatasetLoader.load_and_preprocess('data/raw/sign_mnist_train.csv')
            >>> print(images.shape, images.dtype)  # (N, 28, 28, 1) float32
        """
        result = DatasetLoader.load_csv(csv_path)

        if result is None:
            return None

        images, labels = result

        # Reshape to 28x28
        images = images.reshape(-1, 28, 28)

        # Normalize if requested
        if normalize:
            images = images.astype(np.float32) / 255.0
        else:
            images = images.astype(np.float32)

        # Add channel dimension
        images = np.expand_dims(images, axis=-1)

        logger.info(f"Preprocessing complete: {images.shape}, dtype: {images.dtype}")
        return images, labels

    @staticmethod
    def load_train_test(
        train_csv: str,
        test_csv: str,
        normalize: bool = True
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load both train and test datasets.

        Args:
            train_csv: Path to training CSV
            test_csv: Path to testing CSV
            normalize: Normalize pixel values

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - X_train (N_train, 28, 28, 1)
                - y_train (N_train,)
                - X_test (N_test, 28, 28, 1)
                - y_test (N_test,)
                Or None if loading fails

        Example:
            >>> X_train, y_train, X_test, y_test = DatasetLoader.load_train_test(
            ...     'data/raw/sign_mnist_train.csv',
            ...     'data/raw/sign_mnist_test.csv'
            ... )
        """
        logger.info("Loading train and test datasets...")

        train_result = DatasetLoader.load_and_preprocess(train_csv, normalize)
        test_result = DatasetLoader.load_and_preprocess(test_csv, normalize)

        if train_result is None or test_result is None:
            return None

        return train_result[0], train_result[1], test_result[0], test_result[1]

    @staticmethod
    def get_dataset_info(csv_path: str) -> Optional[dict]:
        """
        Get information about dataset.

        Args:
            csv_path: Path to CSV file

        Returns:
            dict: Dataset statistics or None

        Example:
            >>> info = DatasetLoader.get_dataset_info('data/raw/sign_mnist_train.csv')
            >>> print(f"Classes: {info['num_classes']}")
        """
        result = DatasetLoader.load_csv(csv_path)

        if result is None:
            return None

        images, labels = result
        unique_labels = np.unique(labels)

        return {
            'num_samples': len(images),
            'image_size': (28, 28),
            'num_pixels': 784,
            'num_classes': len(unique_labels),
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'min_pixel': images.min(),
            'max_pixel': images.max(),
            'mean_pixel': images.mean(),
            'std_pixel': images.std()
        }

    @staticmethod
    def split_dataset(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into train and validation.

        Args:
            X: Features array
            y: Labels array
            test_size: Fraction for validation
            random_state: Random seed

        Returns:
            Tuple: (X_train, X_val, y_train, y_val)

        Example:
            >>> X_train, X_val, y_train, y_val = DatasetLoader.split_dataset(X, y, test_size=0.2)
        """
        np.random.seed(random_state)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        split_idx = int(len(X) * (1 - test_size))

        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    # ==================== LANDMARKS DATASET LOADER ====================

    @staticmethod
    def load_landmarks_csv(csv_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load Sign Language Landmarks from CSV file.

        CSV format: 21 landmarks × 3 coordinates (x,y,z) + 1 label column
        Total: 63 features + 1 label = 64 columns
        Example columns: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label

        Args:
            csv_path: Path to landmarks CSV file

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Features array (N, 63) - landmark coordinates
                - Labels array (N,) - letter labels
                Or None if loading fails

        Example:
            >>> landmarks, labels = DatasetLoader.load_landmarks_csv('datasets/landmarks/asl_landmarks_final.csv')
            >>> print(landmarks.shape)  # (N, 63)
        """
        try:
            logger.info(f"Loading landmarks dataset from {csv_path}")
            df = pd.read_csv(csv_path)

            # Separar features y labels
            labels = df.iloc[:, -1].values  # Last column is label
            landmarks = df.iloc[:, :-1].values.astype(np.float32)  # All but last are features

            logger.info(f"Loaded {len(landmarks)} samples, {landmarks.shape[1]} features, {len(np.unique(labels))} classes")
            logger.info(f"Feature range: [{landmarks.min():.4f}, {landmarks.max():.4f}]")
            return landmarks, labels

        except Exception as e:
            logger.error(f"Error loading landmarks CSV: {str(e)}")
            return None

    @staticmethod
    def normalize_landmarks(
        landmarks: np.ndarray,
        standardize: bool = True
    ) -> np.ndarray:
        """
        Normalize landmark coordinates.

        Args:
            landmarks: Features array (N, 63)
            standardize: If True, standardize to mean=0, std=1. If False, scale to [-1, 1]

        Returns:
            Normalized landmarks (N, 63)

        Example:
            >>> landmarks_norm = DatasetLoader.normalize_landmarks(landmarks)
        """
        if standardize:
            # Standardization: (x - mean) / std
            mean = landmarks.mean(axis=0, keepdims=True)
            std = landmarks.std(axis=0, keepdims=True)
            return (landmarks - mean) / (std + 1e-8)
        else:
            # Min-Max scaling to [-1, 1]
            min_val = landmarks.min(axis=0, keepdims=True)
            max_val = landmarks.max(axis=0, keepdims=True)
            return 2 * (landmarks - min_val) / (max_val - min_val + 1e-8) - 1

    @staticmethod
    def load_landmarks_and_preprocess(
        csv_path: str,
        normalize: bool = True,
        standardize: bool = True
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load landmarks CSV and preprocess.

        Args:
            csv_path: Path to landmarks CSV
            normalize: Normalize landmark coordinates
            standardize: If True, standardize. If False, scale to [-1, 1]

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Preprocessed landmarks (N, 63)
                - Labels (N,)
                Or None if loading fails

        Example:
            >>> landmarks, labels = DatasetLoader.load_landmarks_and_preprocess('datasets/landmarks/asl_landmarks_final.csv')
            >>> print(landmarks.shape)  # (N, 63)
        """
        result = DatasetLoader.load_landmarks_csv(csv_path)

        if result is None:
            return None

        landmarks, labels = result

        # Normalize if requested
        if normalize:
            landmarks = DatasetLoader.normalize_landmarks(landmarks, standardize=standardize)

        logger.info(f"Preprocessing complete: {landmarks.shape}, dtype: {landmarks.dtype}")
        return landmarks, labels

    @staticmethod
    def load_landmarks_train_test(
        landmarks_csv: str,
        test_size: float = 0.2,
        normalize: bool = True,
        standardize: bool = True
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load landmarks dataset and split into train/test.

        Args:
            landmarks_csv: Path to landmarks CSV file
            test_size: Fraction for test set
            normalize: Normalize landmarks
            standardize: If True, standardize. If False, scale to [-1, 1]

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - X_train (N_train, 63)
                - y_train (N_train,)
                - X_test (N_test, 63)
                - y_test (N_test,)
                Or None if loading fails

        Example:
            >>> X_train, y_train, X_test, y_test = DatasetLoader.load_landmarks_train_test(
            ...     'datasets/landmarks/asl_landmarks_final.csv'
            ... )
        """
        logger.info("Loading landmarks dataset...")

        result = DatasetLoader.load_landmarks_and_preprocess(landmarks_csv, normalize, standardize)

        if result is None:
            return None

        landmarks, labels = result

        # Split dataset
        X_train, X_test, y_train, y_test = DatasetLoader.split_dataset(
            landmarks, labels, test_size=test_size, random_state=42
        )

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test

    @staticmethod
    def get_landmarks_info(csv_path: str) -> Optional[dict]:
        """
        Get information about landmarks dataset.

        Args:
            csv_path: Path to landmarks CSV file

        Returns:
            dict: Dataset statistics or None

        Example:
            >>> info = DatasetLoader.get_landmarks_info('datasets/landmarks/asl_landmarks_final.csv')
            >>> print(f"Classes: {info['num_classes']}")
        """
        result = DatasetLoader.load_landmarks_csv(csv_path)

        if result is None:
            return None

        landmarks, labels = result
        unique_labels = np.unique(labels)

        return {
            'num_samples': len(landmarks),
            'num_features': landmarks.shape[1],
            'num_landmarks': landmarks.shape[1] // 3,
            'num_classes': len(unique_labels),
            'classes': sorted(unique_labels.tolist()),
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'feature_min': landmarks.min(),
            'feature_max': landmarks.max(),
            'feature_mean': landmarks.mean(),
            'feature_std': landmarks.std()
        }
