"""
Frame extraction and ROI processing for hand detection.

This module handles extraction and normalization of hand regions of interest
from video frames for model inference.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from ..utils.image_utils import preprocess_image, add_channel_dimension, normalize_pixels, normalize_landmarks, landmarks_to_tensor
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class FrameExtractor:
    """
    Extracts and preprocesses hand regions from video frames.

    Handles ROI extraction, normalization, and tensor preparation for
    the CNN model.
    """

    @staticmethod
    def extract_roi(frame: np.ndarray, hand_landmarks) -> Optional[np.ndarray]:
        """
        Extract region of interest (hand) from frame using MediaPipe landmarks.

        Args:
            frame: OpenCV frame (BGR)
            hand_landmarks: MediaPipe hand landmarks object

        Returns:
            np.ndarray: Extracted hand ROI, or None if extraction fails

        Example:
            >>> roi = FrameExtractor.extract_roi(frame, hand_landmarks)
            >>> if roi is not None:
            ...     print(f"ROI shape: {roi.shape}")
        """
        try:
            if hand_landmarks is None or not hand_landmarks.landmark:
                return None

            h, w, _ = frame.shape
            landmarks = hand_landmarks.landmark

            # Get bounding box from landmarks
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Add padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size == 0:
                logger.warning("Extracted ROI is empty")
                return None

            logger.debug(f"Extracted ROI: {roi.shape}")
            return roi

        except Exception as e:
            logger.error(f"ROI extraction error: {str(e)}")
            return None

    @staticmethod
    def normalize_roi(roi: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Normalize ROI to target size and grayscale.

        Args:
            roi: Region of interest image
            target_size: Target output size (height, width)

        Returns:
            np.ndarray: Normalized ROI

        Example:
            >>> normalized = FrameExtractor.normalize_roi(roi, (28, 28))
            >>> print(normalized.shape, normalized.dtype)  # (28, 28) float32
        """
        try:
            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            # Resize
            resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)

            # Normalize to 0-1
            normalized = normalize_pixels(resized)

            logger.debug(f"ROI normalized to {normalized.shape}")
            return normalized

        except Exception as e:
            logger.error(f"ROI normalization error: {str(e)}")
            return np.zeros(target_size, dtype=np.float32)

    @staticmethod
    def get_preprocessed_tensor(roi: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Get preprocessed tensor ready for model inference.

        Normalizes ROI and adds batch and channel dimensions.

        Args:
            roi: Region of interest image
            target_size: Target size

        Returns:
            np.ndarray: Tensor of shape (1, 28, 28, 1)

        Example:
            >>> tensor = FrameExtractor.get_preprocessed_tensor(roi)
            >>> predictions = model.predict(tensor)  # Shape (1, 24)
        """
        # Normalize
        normalized = FrameExtractor.normalize_roi(roi, target_size)

        # Add channel dimension
        with_channel = add_channel_dimension(normalized)

        # Add batch dimension
        batch_tensor = np.expand_dims(with_channel, axis=0)

        logger.debug(f"Prepared tensor shape: {batch_tensor.shape}")
        return batch_tensor

    @staticmethod
    def extract_hand_region(frame: np.ndarray, hand_landmarks) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract hand region with bounding box coordinates.

        Args:
            frame: OpenCV frame
            hand_landmarks: MediaPipe landmarks

        Returns:
            Optional[Tuple]:
                - Hand ROI image
                - Bounding box (x_min, y_min, x_max, y_max)
                Or None if extraction fails

        Example:
            >>> result = FrameExtractor.extract_hand_region(frame, landmarks)
            >>> if result:
            ...     roi, bbox = result
            ...     x_min, y_min, x_max, y_max = bbox
        """
        try:
            if hand_landmarks is None or not hand_landmarks.landmark:
                return None

            h, w, _ = frame.shape
            landmarks = hand_landmarks.landmark

            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Add padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size == 0:
                return None

            bbox = (x_min, y_min, x_max, y_max)
            return roi, bbox

        except Exception as e:
            logger.error(f"Hand region extraction error: {str(e)}")
            return None

    @staticmethod
    def get_hand_centroid(hand_landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """
        Get centroid of hand from landmarks.

        Args:
            hand_landmarks: MediaPipe landmarks
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Optional[Tuple[int, int]]: (x, y) centroid coordinates, or None

        Example:
            >>> centroid = FrameExtractor.get_hand_centroid(landmarks, w, h)
        """
        try:
            if hand_landmarks is None or not hand_landmarks.landmark:
                return None

            landmarks = hand_landmarks.landmark

            x_coords = [lm.x * frame_width for lm in landmarks]
            y_coords = [lm.y * frame_height for lm in landmarks]

            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))

            return (centroid_x, centroid_y)

        except Exception as e:
            logger.error(f"Centroid calculation error: {str(e)}")
            return None

    @staticmethod
    def get_hand_size(hand_landmarks, frame_width: int, frame_height: int) -> Optional[float]:
        """
        Get size of hand as bounding box area.

        Args:
            hand_landmarks: MediaPipe landmarks
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Optional[float]: Hand area, or None if cannot calculate

        Example:
            >>> area = FrameExtractor.get_hand_size(landmarks, w, h)
        """
        try:
            if hand_landmarks is None or not hand_landmarks.landmark:
                return None

            landmarks = hand_landmarks.landmark

            x_coords = [lm.x * frame_width for lm in landmarks]
            y_coords = [lm.y * frame_height for lm in landmarks]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            width = x_max - x_min
            height = y_max - y_min
            area = width * height

            return float(area)

        except Exception as e:
            logger.error(f"Hand size calculation error: {str(e)}")
            return None

    # ==================== LANDMARKS EXTRACTION ====================

    @staticmethod
    def extract_landmarks(hand_landmarks) -> Optional[np.ndarray]:
        """
        Extract landmark coordinates from MediaPipe hand landmarks.

        Args:
            hand_landmarks: MediaPipe hand landmarks object

        Returns:
            np.ndarray: Landmarks array (21, 3) with x, y, z coordinates, or None if extraction fails

        Example:
            >>> landmarks = FrameExtractor.extract_landmarks(hand_landmarks)
            >>> if landmarks is not None:
            ...     print(landmarks.shape)  # (21, 3)
        """
        try:
            if hand_landmarks is None or not hand_landmarks.landmark:
                return None

            # Extract coordinates for all 21 landmarks
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.append([lm.x, lm.y, lm.z])

            landmarks_array = np.array(landmarks_list, dtype=np.float32)
            logger.debug(f"Extracted landmarks: shape {landmarks_array.shape}")
            return landmarks_array

        except Exception as e:
            logger.error(f"Landmarks extraction error: {str(e)}")
            return None

    @staticmethod
    def get_landmarks_tensor(hand_landmarks, normalize: bool = True, add_batch_dim: bool = True) -> Optional[np.ndarray]:
        """
        Get landmarks as tensor ready for model inference.

        Args:
            hand_landmarks: MediaPipe hand landmarks object
            normalize: Normalize landmarks (standardization)
            add_batch_dim: Add batch dimension for model.predict()

        Returns:
            np.ndarray: Tensor of shape (1, 63) if add_batch_dim=True, else (63,)

        Example:
            >>> tensor = FrameExtractor.get_landmarks_tensor(hand_landmarks)
            >>> predictions = model.predict(tensor)
        """
        try:
            # Extract landmarks
            landmarks = FrameExtractor.extract_landmarks(hand_landmarks)
            if landmarks is None:
                return None

            # Flatten to (63,)
            landmarks_flat = landmarks.flatten().astype(np.float32)

            # Normalize if requested
            if normalize:
                landmarks_flat = normalize_landmarks(landmarks_flat, standardize=True)

            # Add batch dimension if requested
            if add_batch_dim:
                landmarks_flat = np.expand_dims(landmarks_flat, axis=0)

            logger.debug(f"Prepared landmarks tensor shape: {landmarks_flat.shape}")
            return landmarks_flat

        except Exception as e:
            logger.error(f"Landmarks tensor preparation error: {str(e)}")
            return None

    @staticmethod
    def extract_landmarks_and_roi(frame: np.ndarray, hand_landmarks) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract both landmarks and ROI image from frame.

        Useful for dual-stream models or visualization.

        Args:
            frame: OpenCV frame (BGR)
            hand_landmarks: MediaPipe hand landmarks object

        Returns:
            Optional[Tuple]:
                - Landmarks array (21, 3)
                - ROI image
                Or None if extraction fails

        Example:
            >>> result = FrameExtractor.extract_landmarks_and_roi(frame, landmarks)
            >>> if result:
            ...     landmarks, roi = result
        """
        try:
            landmarks = FrameExtractor.extract_landmarks(hand_landmarks)
            roi = FrameExtractor.extract_roi(frame, hand_landmarks)

            if landmarks is None or roi is None:
                return None

            return landmarks, roi

        except Exception as e:
            logger.error(f"Combined extraction error: {str(e)}")
            return None
