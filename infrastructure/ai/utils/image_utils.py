"""
Image processing utilities for sign language recognition.

This module provides functions for loading, preprocessing, normalizing,
and transforming images for the CNN model.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file using OpenCV.

    Args:
        image_path: Path to the image file

    Returns:
        numpy.ndarray: Image array in BGR format, or None if loading fails

    Example:
        >>> image = load_image("path/to/image.jpg")
        >>> if image is not None:
        ...     print(image.shape)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        logger.debug(f"Loaded image: {image_path} with shape {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.

    Args:
        image: Input image in BGR format

    Returns:
        numpy.ndarray: Grayscale image

    Example:
        >>> gray = convert_to_grayscale(image)
        >>> print(gray.shape)  # (height, width)
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_pixels(image: np.ndarray, max_value: float = 255.0) -> np.ndarray:
    """
    Normalize pixel values to 0-1 range.

    Args:
        image: Input image
        max_value: Maximum pixel value (typically 255.0)

    Returns:
        numpy.ndarray: Normalized image with values in [0, 1]

    Example:
        >>> normalized = normalize_pixels(image)
        >>> print(normalized.min(), normalized.max())  # 0.0, 1.0
    """
    return image.astype(np.float32) / max_value


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to specified dimensions.

    Args:
        image: Input image
        size: Target size as (width, height)

    Returns:
        numpy.ndarray: Resized image

    Example:
        >>> resized = resize_image(image, (28, 28))
        >>> print(resized.shape)
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """
    Complete preprocessing pipeline for model input.

    Converts to grayscale, resizes to target size, and normalizes pixel values.

    Args:
        image: Input image (BGR or grayscale)
        target_size: Target dimensions (default: 28x28 for MNIST)

    Returns:
        numpy.ndarray: Preprocessed image (grayscale, normalized, resized)

    Example:
        >>> processed = preprocess_image(image, (28, 28))
        >>> print(processed.shape, processed.dtype)
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)

    # Resize
    resized = resize_image(gray, target_size)

    # Normalize
    normalized = normalize_pixels(resized)

    return normalized


def add_channel_dimension(image: np.ndarray) -> np.ndarray:
    """
    Add channel dimension for CNN input.

    Converts (height, width) to (height, width, 1) for grayscale images.

    Args:
        image: 2D image array

    Returns:
        numpy.ndarray: 3D array with added channel dimension

    Example:
        >>> with_channel = add_channel_dimension(image)
        >>> print(with_channel.shape)  # (28, 28, 1)
    """
    if len(image.shape) == 2:
        return np.expand_dims(image, axis=-1)
    return image


def prepare_for_inference(image: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """
    Prepare image tensor for CNN inference.

    Applies full preprocessing and adds batch dimension.

    Args:
        image: Input image (BGR or grayscale)
        target_size: Target size for model

    Returns:
        numpy.ndarray: Tensor ready for model.predict() - shape (1, 28, 28, 1)

    Example:
        >>> tensor = prepare_for_inference(image)
        >>> predictions = model.predict(tensor)
    """
    # Preprocess
    processed = preprocess_image(image, target_size)

    # Add channel dimension
    with_channel = add_channel_dimension(processed)

    # Add batch dimension
    batch_tensor = np.expand_dims(with_channel, axis=0)

    return batch_tensor


def draw_bounding_box(
    image: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: Input image
        x, y: Top-left corner coordinates
        w, h: Width and height
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        numpy.ndarray: Image with drawn bounding box

    Example:
        >>> boxed = draw_bounding_box(image, 10, 10, 100, 100)
    """
    return cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def draw_text(
    image: np.ndarray,
    text: str,
    x: int, y: int,
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw text on image.

    Args:
        image: Input image
        text: Text to draw
        x, y: Position coordinates
        font_scale: Font size scale
        color: BGR color
        thickness: Text thickness

    Returns:
        numpy.ndarray: Image with drawn text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(image, text, (x, y), font, font_scale, color, thickness)


# ==================== LANDMARKS UTILITIES ====================

def normalize_landmarks(
    landmarks: np.ndarray,
    standardize: bool = True
) -> np.ndarray:
    """
    Normalize landmark coordinates.

    Args:
        landmarks: Landmarks array (63,) or (21, 3)
        standardize: If True, standardize to mean=0, std=1. If False, min-max to [-1, 1]

    Returns:
        numpy.ndarray: Normalized landmarks with same shape as input

    Example:
        >>> normalized = normalize_landmarks(landmarks)
        >>> print(normalized.mean(), normalized.std())  # ~0, ~1
    """
    original_shape = landmarks.shape

    # Reshape to 2D if needed
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)
        was_flat = True
    else:
        was_flat = False

    if standardize:
        # Standardization: (x - mean) / std
        mean = landmarks.mean(axis=0, keepdims=True)
        std = landmarks.std(axis=0, keepdims=True)
        normalized = (landmarks - mean) / (std + 1e-8)
    else:
        # Min-Max scaling to [-1, 1]
        min_val = landmarks.min(axis=0, keepdims=True)
        max_val = landmarks.max(axis=0, keepdims=True)
        normalized = 2 * (landmarks - min_val) / (max_val - min_val + 1e-8) - 1

    # Restore original shape
    if was_flat:
        normalized = normalized.flatten()

    return normalized.astype(np.float32)


def landmarks_to_tensor(
    landmarks: np.ndarray,
    normalize: bool = True,
    add_batch_dim: bool = True
) -> np.ndarray:
    """
    Convert landmarks to model input tensor.

    Args:
        landmarks: Landmarks array (63,) or (21, 3)
        normalize: Normalize landmarks
        add_batch_dim: Add batch dimension for model.predict()

    Returns:
        numpy.ndarray: Tensor ready for model - (1, 63) if add_batch_dim=True, else (63,)

    Example:
        >>> tensor = landmarks_to_tensor(landmarks)
        >>> predictions = model.predict(tensor)  # Shape: (1, 28)
    """
    # Ensure shape is (63,)
    if landmarks.ndim == 2:
        landmarks = landmarks.flatten()

    # Normalize
    if normalize:
        landmarks = normalize_landmarks(landmarks, standardize=True)

    # Add batch dimension if requested
    if add_batch_dim:
        landmarks = np.expand_dims(landmarks, axis=0)

    return landmarks.astype(np.float32)


def extract_hand_roi_from_landmarks(
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int],
    padding: float = 0.1
) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box from hand landmarks.

    Args:
        landmarks: Landmarks array (21, 3) with normalized coordinates [0-1]
        frame_shape: Frame dimensions (height, width)
        padding: Padding as fraction of bbox size

    Returns:
        Tuple[int, int, int, int]: (x, y, w, h) bounding box coordinates

    Example:
        >>> x, y, w, h = extract_hand_roi_from_landmarks(landmarks, (480, 640))
    """
    # Reshape if needed
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)

    # Get x, y coordinates (ignore z depth)
    xy = landmarks[:, :2]

    # Convert from normalized [0-1] to pixel coordinates
    height, width = frame_shape
    xy_pixels = xy * np.array([width, height])

    # Calculate bounding box
    x_min, y_min = xy_pixels.min(axis=0)
    x_max, y_max = xy_pixels.max(axis=0)

    w = x_max - x_min
    h = y_max - y_min

    # Add padding
    pad_x = w * padding
    pad_y = h * padding

    x = max(0, int(x_min - pad_x))
    y = max(0, int(y_min - pad_y))
    w = int(w + 2 * pad_x)
    h = int(h + 2 * pad_y)

    # Ensure within frame bounds
    w = min(w, width - x)
    h = min(h, height - y)

    return x, y, w, h


def draw_landmarks_on_image(
    image: np.ndarray,
    landmarks: np.ndarray,
    frame_shape: Optional[Tuple[int, int]] = None,
    circle_color: Tuple[int, int, int] = (0, 255, 0),
    line_color: Tuple[int, int, int] = (255, 0, 0),
    circle_radius: int = 5,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw hand landmarks and connections on image.

    Args:
        image: Input image to draw on
        landmarks: Landmarks array (21, 3) with normalized coordinates [0-1]
        frame_shape: Frame shape (height, width) - if None, use image.shape[:2]
        circle_color: BGR color for landmark points
        line_color: BGR color for connections
        circle_radius: Radius of landmark circles
        line_thickness: Thickness of connection lines

    Returns:
        numpy.ndarray: Image with drawn landmarks

    Example:
        >>> annotated = draw_landmarks_on_image(image, landmarks)
    """
    if frame_shape is None:
        frame_shape = image.shape[:2]

    # Reshape if needed
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)

    # Convert from normalized to pixel coordinates
    height, width = frame_shape
    xy_pixels = (landmarks[:, :2] * np.array([width, height])).astype(np.int32)

    # Draw hand connections (standard MediaPipe hand topology)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    for start_idx, end_idx in connections:
        if start_idx < len(xy_pixels) and end_idx < len(xy_pixels):
            start = tuple(xy_pixels[start_idx])
            end = tuple(xy_pixels[end_idx])
            cv2.line(image, start, end, line_color, line_thickness)

    # Draw landmark points
    for i, point in enumerate(xy_pixels):
        cv2.circle(image, tuple(point), circle_radius, circle_color, -1)

    return image
