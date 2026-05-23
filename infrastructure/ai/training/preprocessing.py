"""
Data preprocessing and augmentation utilities.

This module provides functions for normalizing data, applying augmentation,
and preparing datasets for training.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class DataPreprocessor:
    """Handles data preprocessing and augmentation."""

    @staticmethod
    def normalize_data(
        X: np.ndarray,
        X_val: np.ndarray = None,
        X_test: np.ndarray = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Normalize image data to [0, 1] range.

        Args:
            X: Training images
            X_val: Validation images (optional)
            X_test: Test images (optional)

        Returns:
            Tuple: Normalized arrays

        Example:
            >>> X_train, X_val, X_test = DataPreprocessor.normalize_data(X_train, X_val, X_test)
        """
        X = X.astype(np.float32) / 255.0

        if X_val is not None:
            X_val = X_val.astype(np.float32) / 255.0

        if X_test is not None:
            X_test = X_test.astype(np.float32) / 255.0

        logger.info(f"Data normalized: X_train [{X.min():.4f}, {X.max():.4f}]")

        if X_val is not None and X_test is not None:
            return X, X_val, X_test
        elif X_val is not None:
            return X, X_val
        else:
            return X

    @staticmethod
    def create_augmentation_generator(
        rotation_range: int = 10,
        zoom_range: float = 0.1,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        fill_mode: str = 'nearest'
    ) -> ImageDataGenerator:
        """
        Create ImageDataGenerator for data augmentation.

        Args:
            rotation_range: Rotation range in degrees
            zoom_range: Zoom range
            horizontal_flip: Whether to flip horizontally
            vertical_flip: Whether to flip vertically
            fill_mode: Fill mode for augmentation

        Returns:
            ImageDataGenerator: Configured augmentation generator

        Example:
            >>> aug_gen = DataPreprocessor.create_augmentation_generator()
            >>> train_generator = aug_gen.flow(X_train, y_train, batch_size=32)
        """
        return ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            fill_mode=fill_mode
        )

    @staticmethod
    def apply_augmentation(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        augmentation_config: dict = None
    ):
        """
        Apply data augmentation and create generator.

        Args:
            X: Training images
            y: Training labels
            batch_size: Batch size for generator
            augmentation_config: Augmentation parameters

        Returns:
            Generator for batches of augmented data

        Example:
            >>> gen = DataPreprocessor.apply_augmentation(X_train, y_train, batch_size=32)
            >>> for X_batch, y_batch in gen:
            ...     break
        """
        if augmentation_config is None:
            augmentation_config = {}

        aug_gen = DataPreprocessor.create_augmentation_generator(**augmentation_config)

        logger.info(f"Created augmentation generator with batch_size={batch_size}")

        return aug_gen.flow(X, y, batch_size=batch_size, shuffle=True)

    @staticmethod
    def prepare_training_data(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        normalize: bool = True,
        apply_augmentation: bool = True
    ) -> Tuple:
        """
        Complete preprocessing pipeline.

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            normalize: Normalize pixel values
            apply_augmentation: Apply data augmentation

        Returns:
            Tuple: Processed data or generators

        Example:
            >>> result = DataPreprocessor.prepare_training_data(X_train, y_train, X_val, y_val)
        """
        # Normalize
        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_val = X_val.astype(np.float32) / 255.0

        # Augmentation
        if apply_augmentation:
            aug_config = {
                'rotation_range': 10,
                'zoom_range': 0.1,
                'horizontal_flip': True
            }
            train_gen = DataPreprocessor.apply_augmentation(X_train, y_train, batch_size=32, augmentation_config=aug_config)
            logger.info("Training pipeline ready with augmentation")
            return train_gen, X_val, y_val
        else:
            logger.info("Training pipeline ready without augmentation")
            return X_train, X_val, y_val, y_train

    @staticmethod
    def get_augmentation_pipeline() -> ImageDataGenerator:
        """
        Get standard augmentation pipeline for training.

        Returns:
            ImageDataGenerator: Standard augmentation configuration

        Example:
            >>> pipeline = DataPreprocessor.get_augmentation_pipeline()
        """
        return DataPreprocessor.create_augmentation_generator(
            rotation_range=15,
            zoom_range=0.15,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest'
        )

    # ==================== LANDMARKS AUGMENTATION ====================

    @staticmethod
    def augment_landmarks(
        landmarks: np.ndarray,
        noise_std: float = 0.02,
        rotation_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        flip_horizontal: bool = False
    ) -> np.ndarray:
        """
        Apply augmentation to a single landmarks sample.

        Args:
            landmarks: Landmarks array (63,) - 21 landmarks × 3 coordinates (x,y,z)
            noise_std: Standard deviation of Gaussian noise
            rotation_range: Rotation range in degrees around centroid
            scale_range: (min_scale, max_scale) for scaling
            flip_horizontal: Whether to flip horizontally

        Returns:
            Augmented landmarks (63,)

        Example:
            >>> aug_landmarks = DataPreprocessor.augment_landmarks(landmarks)
        """
        # Reshape to (21, 3)
        landmarks = landmarks.reshape(21, 3).copy()

        # Calculate centroid (use only x, y for centroid)
        centroid = landmarks[:, :2].mean(axis=0)

        # 1. Rotation around centroid (only x, y affected)
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            angle_rad = np.radians(angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Translate to origin
            xy = landmarks[:, :2] - centroid

            # Rotate
            rotated_xy = np.zeros_like(xy)
            rotated_xy[:, 0] = xy[:, 0] * cos_a - xy[:, 1] * sin_a
            rotated_xy[:, 1] = xy[:, 0] * sin_a + xy[:, 1] * cos_a

            # Translate back
            landmarks[:, :2] = rotated_xy + centroid

        # 2. Scaling
        scale = np.random.uniform(scale_range[0], scale_range[1])
        # Scale around centroid
        landmarks[:, :2] = (landmarks[:, :2] - centroid) * scale + centroid
        # Z coordinate scaling
        landmarks[:, 2] = landmarks[:, 2] * scale

        # 3. Gaussian noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, landmarks.shape)
            landmarks = landmarks + noise

        # 4. Horizontal flip (mirror x coordinates)
        if flip_horizontal:
            # Mirror x coordinates
            landmarks[:, 0] = -landmarks[:, 0]
            # Some landmarks might need index swapping for proper left-right symmetry
            # For now, just flip the coordinates

        return landmarks.flatten().astype(np.float32)

    @staticmethod
    def augment_landmarks_batch(
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: int = 3,
        **augment_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create augmented versions of landmarks dataset.

        Args:
            X: Landmarks array (N, 63)
            y: Labels array (N,)
            augmentation_factor: Number of augmented copies per sample
            **augment_kwargs: Arguments for augment_landmarks()

        Returns:
            Tuple[np.ndarray, np.ndarray]: Augmented (X, y)

        Example:
            >>> X_aug, y_aug = DataPreprocessor.augment_landmarks_batch(X_train, y_train, augmentation_factor=2)
        """
        augmented_X = [X]
        augmented_y = [y]

        for _ in range(augmentation_factor):
            X_aug = np.array([
                DataPreprocessor.augment_landmarks(sample, **augment_kwargs)
                for sample in X
            ])
            augmented_X.append(X_aug)
            augmented_y.append(y.copy())

        X_augmented = np.vstack(augmented_X)
        y_augmented = np.hstack(augmented_y)

        logger.info(f"Created augmented dataset: {X.shape} -> {X_augmented.shape}")

        return X_augmented, y_augmented

    @staticmethod
    def create_landmarks_augmentation_generator(
        noise_std: float = 0.02,
        rotation_range: float = 10.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        flip_probability: float = 0.3
    ):
        """
        Create a simple augmentation generator for landmarks.

        Returns a Python generator that yields augmented batches.

        Args:
            noise_std: Gaussian noise standard deviation
            rotation_range: Rotation range in degrees
            scale_range: Scaling range
            flip_probability: Probability of horizontal flip

        Returns:
            Generator function

        Example:
            >>> gen = DataPreprocessor.create_landmarks_augmentation_generator()
            >>> batch_gen = gen(X_train, y_train, batch_size=32)
        """
        def augmentation_generator(X, y, batch_size=32, shuffle=True):
            """Generate augmented batches."""
            indices = np.arange(len(X))

            while True:
                if shuffle:
                    np.random.shuffle(indices)

                for start in range(0, len(X), batch_size):
                    end = min(start + batch_size, len(X))
                    batch_idx = indices[start:end]

                    X_batch = X[batch_idx].copy()
                    y_batch = y[batch_idx].copy()

                    # Augment each sample
                    for i in range(len(X_batch)):
                        flip = np.random.random() < flip_probability
                        X_batch[i] = DataPreprocessor.augment_landmarks(
                            X_batch[i],
                            noise_std=noise_std,
                            rotation_range=rotation_range,
                            scale_range=scale_range,
                            flip_horizontal=flip
                        )

                    yield X_batch, y_batch

        return augmentation_generator

    @staticmethod
    def apply_landmarks_augmentation(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        augmentation_config: dict = None
    ):
        """
        Apply landmarks augmentation and create generator.

        Args:
            X: Training landmarks (N, 63)
            y: Training labels (N,)
            batch_size: Batch size
            augmentation_config: Augmentation parameters

        Returns:
            Generator for batches of augmented landmarks

        Example:
            >>> gen = DataPreprocessor.apply_landmarks_augmentation(X_train, y_train)
            >>> for X_batch, y_batch in gen:
            ...     break
        """
        if augmentation_config is None:
            augmentation_config = {
                'noise_std': 0.02,
                'rotation_range': 10.0,
                'scale_range': (0.9, 1.1),
                'flip_probability': 0.3
            }

        aug_gen_func = DataPreprocessor.create_landmarks_augmentation_generator(**augmentation_config)
        gen = aug_gen_func(X, y, batch_size=batch_size, shuffle=True)

        logger.info(f"Created landmarks augmentation generator with batch_size={batch_size}")

        return gen

    @staticmethod
    def prepare_landmarks_training_data(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        apply_augmentation: bool = True,
        augmentation_config: dict = None
    ) -> Tuple:
        """
        Complete preprocessing pipeline for landmarks.

        Args:
            X_train: Training landmarks (N, 63)
            y_train: Training labels
            X_val: Validation landmarks
            y_val: Validation labels
            apply_augmentation: Apply data augmentation
            augmentation_config: Augmentation parameters

        Returns:
            Tuple: (train_generator, X_val, y_val) or (X_train, y_train, X_val, y_val)

        Example:
            >>> result = DataPreprocessor.prepare_landmarks_training_data(
            ...     X_train, y_train, X_val, y_val
            ... )
        """
        if augmentation_config is None:
            augmentation_config = {
                'noise_std': 0.02,
                'rotation_range': 10.0,
                'scale_range': (0.9, 1.1),
                'flip_probability': 0.3
            }

        if apply_augmentation:
            train_gen = DataPreprocessor.apply_landmarks_augmentation(
                X_train, y_train,
                batch_size=32,
                augmentation_config=augmentation_config
            )
            logger.info("Landmarks training pipeline ready with augmentation")
            return train_gen, X_val, y_val
        else:
            logger.info("Landmarks training pipeline ready without augmentation")
            return X_train, y_train, X_val, y_val
