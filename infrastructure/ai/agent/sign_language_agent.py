"""
Main sign language agent that orchestrates all AI components.

This module implements the primary agent that coordinates model inference,
confidence management, and prediction buffering for real-time sign language
recognition.
"""

import json
import numpy as np
import cv2
from typing import Optional, Tuple, List
from pathlib import Path
from .inference_agent import InferenceAgent
from .confidence_manager import ConfidenceManager
from .prediction_manager import PredictionManager
from ..utils.image_utils import prepare_for_inference, landmarks_to_tensor, normalize_landmarks
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class SignLanguageAgent:
    """
    Main agent for sign language recognition and translation.

    Orchestrates model inference, confidence management, and prediction
    buffering to provide stable, real-time predictions.

    Attributes:
        inference_agent: Low-level model inference
        confidence_manager: Confidence validation
        prediction_manager: Prediction buffering and word accumulation
        labels: Mapping from class index to letter
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.75):
        """
        Initialize sign language agent.

        Args:
            model_path: Path to trained model
            labels_path: Path to labels.json mapping indices to letters
            confidence_threshold: Minimum confidence for accepting predictions

        Raises:
            RuntimeError: If model or labels cannot be loaded

        Example:
            >>> agent = SignLanguageAgent('trained_models/best_model.keras', 'trained_models/labels.json')
            >>> letter, confidence = agent.predict(frame)
        """
        if hasattr(self, '_initialized'):
            return

        self.model_path = model_path
        self.labels_path = labels_path

        try:
            self.inference_agent = InferenceAgent(model_path)
            self.labels = self._load_labels(labels_path)

            self.confidence_manager = ConfidenceManager(confidence_threshold=confidence_threshold)
            self.prediction_manager = PredictionManager()

            logger.info(f"SignLanguageAgent initialized successfully with {len(self.labels)} classes")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize SignLanguageAgent: {str(e)}")
            raise RuntimeError(f"SignLanguageAgent initialization failed: {str(e)}")

    def _load_labels(self, labels_path: str) -> dict:
        """
        Load label mapping from JSON file.

        Args:
            labels_path: Path to labels.json

        Returns:
            dict: Mapping from index to letter

        Raises:
            FileNotFoundError: If labels file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        labels_path = Path(labels_path)

        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)

            # Convert string keys to integers
            labels_dict = {}
            for key, value in labels.items():
                try:
                    labels_dict[int(key)] = value
                except ValueError:
                    labels_dict[key] = value

            logger.info(f"Loaded {len(labels_dict)} labels from {labels_path}")
            return labels_dict

        except Exception as e:
            logger.error(f"Error loading labels from {labels_path}: {str(e)}")
            raise

    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess OpenCV frame for model inference.

        Converts BGR frame to 28x28 grayscale normalized tensor.

        Args:
            frame: OpenCV image frame (BGR)

        Returns:
            np.ndarray: Preprocessed tensor (1, 28, 28, 1), or None if preprocessing fails

        Example:
            >>> tensor = agent.preprocess_frame(frame)
            >>> if tensor is not None:
            ...     letter, conf = agent.predict(frame)
        """
        try:
            tensor = prepare_for_inference(frame)
            logger.debug(f"Frame preprocessed: shape {tensor.shape}, dtype {tensor.dtype}")
            return tensor

        except Exception as e:
            logger.error(f"Frame preprocessing error: {str(e)}")
            return None

    def predict(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predict letter from frame with full confidence management.

        Args:
            frame: OpenCV image frame (BGR)

        Returns:
            Tuple[Optional[str], float]:
                - Predicted letter (A-X) or None if not confident
                - Confidence value (0.0 to 1.0)

        Example:
            >>> letter, confidence = agent.predict(frame)
            >>> if letter:
            ...     print(f"Predicted: {letter} ({confidence:.1%})")
        """
        # Preprocess frame
        tensor = self.preprocess_frame(frame)
        if tensor is None:
            return None, 0.0

        # Get predictions
        probabilities, success = self.inference_agent.predict(tensor)
        if not success or len(probabilities) == 0:
            return None, 0.0

        # Get top prediction
        top_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[top_idx])

        # Check confidence threshold
        self.confidence_manager.add_to_history(confidence)

        if not self.confidence_manager.is_confident(confidence):
            logger.debug(f"Low confidence: {confidence:.4f} < {self.confidence_manager.confidence_threshold}")
            return None, confidence

        # Get letter from label
        letter = self.labels.get(top_idx, f"Class{top_idx}")

        # Add to buffer
        self.prediction_manager.add_prediction(letter, confidence)

        logger.debug(f"Prediction: {letter} ({confidence:.4f})")

        return letter, confidence

    def predict_from_landmarks(self, landmarks) -> Tuple[Optional[str], float]:
        """
        Predict letter from hand landmarks with full confidence management.

        Args:
            landmarks: Hand landmarks - either:
                - numpy array (21, 3) or (63,) with x,y,z coordinates
                - MediaPipe hand landmarks object

        Returns:
            Tuple[Optional[str], float]:
                - Predicted letter (A-Z) or None if not confident
                - Confidence value (0.0 to 1.0)

        Example:
            >>> landmarks = np.random.randn(21, 3)
            >>> letter, confidence = agent.predict_from_landmarks(landmarks)
            >>> if letter:
            ...     print(f"Predicted: {letter} ({confidence:.1%})")
        """
        try:
            # Convert MediaPipe landmarks to numpy array if needed
            from ..realtime.frame_extractor import FrameExtractor
            if not isinstance(landmarks, np.ndarray):
                # It's a MediaPipe landmark object
                landmarks_array = FrameExtractor.extract_landmarks(landmarks)
                if landmarks_array is None:
                    logger.warning("Failed to extract landmarks from MediaPipe object")
                    return None, 0.0
                landmarks = landmarks_array

            # Preprocess landmarks
            tensor = self.preprocess_landmarks(landmarks)
            if tensor is None:
                return None, 0.0

            # Get predictions
            probabilities, success = self.inference_agent.predict(tensor)
            if not success or len(probabilities) == 0:
                return None, 0.0

            # Get top prediction
            top_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[top_idx])

            # Check confidence threshold
            self.confidence_manager.add_to_history(confidence)

            if not self.confidence_manager.is_confident(confidence):
                logger.debug(f"Low confidence: {confidence:.4f} < {self.confidence_manager.confidence_threshold}")
                return None, confidence

            # Get letter from label
            letter = self.labels.get(top_idx, f"Class{top_idx}")

            # Add to buffer
            self.prediction_manager.add_prediction(letter, confidence)

            logger.debug(f"Landmarks prediction: {letter} ({confidence:.4f})")

            return letter, confidence

        except Exception as e:
            logger.error(f"Landmarks prediction error: {str(e)}")
            return None, 0.0

    def preprocess_landmarks(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess hand landmarks for model inference.

        Normalizes landmarks to mean=0, std=1 and adds batch dimension.

        Args:
            landmarks: Hand landmarks (21, 3) or (63,)

        Returns:
            np.ndarray: Preprocessed tensor (1, 63), or None if preprocessing fails

        Example:
            >>> tensor = agent.preprocess_landmarks(landmarks)
        """
        try:
            tensor = landmarks_to_tensor(landmarks, normalize=True, add_batch_dim=True)
            logger.debug(f"Landmarks preprocessed: shape {tensor.shape}, dtype {tensor.dtype}")
            return tensor

        except Exception as e:
            logger.error(f"Landmarks preprocessing error: {str(e)}")
            return None

    def get_stable_prediction(self) -> Tuple[Optional[str], float]:
        """
        Get temporally stable prediction from buffer.

        Args:
            None

        Returns:
            Tuple[Optional[str], float]:
                - Stable letter, or None if unstable
                - Confidence value

        Example:
            >>> letter, confidence = agent.get_stable_prediction()
            >>> if letter:
            ...     print(f"Stable: {letter}")
        """
        result = self.prediction_manager.get_stable_prediction()

        if result is None:
            return None, 0.0

        letter, frequency, confidence = result
        return letter, confidence

    def add_letter_to_word(self, letter: str) -> bool:
        """
        Add letter to word buffer with cooldown.

        Args:
            letter: Letter to add

        Returns:
            bool: True if added, False if cooldown blocked it

        Example:
            >>> if agent.add_letter_to_word('H'):
            ...     print(f"Word: {agent.get_word()}")
        """
        return self.prediction_manager.add_to_word(letter)

    def get_word(self) -> str:
        """
        Get accumulated word.

        Returns:
            str: Current word from buffer

        Example:
            >>> word = agent.get_word()
        """
        return self.prediction_manager.get_word()

    def clear_word(self) -> str:
        """
        Clear word buffer.

        Returns:
            str: The word that was cleared

        Example:
            >>> cleared = agent.clear_word()
            >>> print(f"Cleared: {cleared}")
        """
        return self.prediction_manager.clear_word()

    def undo_letter(self) -> Optional[str]:
        """
        Remove last letter from word (backspace).

        Returns:
            Optional[str]: The removed letter, or None if empty

        Example:
            >>> removed = agent.undo_letter()
        """
        return self.prediction_manager.remove_last_letter()

    def get_prediction_history(self, limit: int = 5) -> List[dict]:
        """
        Get recent prediction history.

        Args:
            limit: Maximum predictions to return

        Returns:
            List[dict]: Recent predictions

        Example:
            >>> history = agent.get_prediction_history(limit=10)
        """
        return self.prediction_manager.get_prediction_history(limit)

    def get_statistics(self) -> dict:
        """
        Get agent statistics.

        Returns:
            dict: Statistics about predictions and buffers

        Example:
            >>> stats = agent.get_statistics()
            >>> print(f"Average confidence: {stats['avg_confidence']:.2%}")
        """
        pm_stats = self.prediction_manager.get_statistics()

        return {
            'word': self.get_word(),
            'word_length': len(self.prediction_manager.word_buffer),
            'buffer_size': len(self.prediction_manager.prediction_buffer),
            'confidence_threshold': self.confidence_manager.confidence_threshold,
            'avg_confidence': self.confidence_manager.get_average_confidence(),
            'max_confidence': self.confidence_manager.get_max_confidence(),
            'min_confidence': self.confidence_manager.get_min_confidence(),
            'confidence_trend': self.confidence_manager.get_confidence_trend(),
            'prediction_distribution': pm_stats['letter_distribution'],
            'total_classes': len(self.labels)
        }

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0)

        Example:
            >>> agent.set_confidence_threshold(0.80)
        """
        self.confidence_manager.set_threshold(threshold)

    def reset(self) -> None:
        """Reset all buffers and predictions."""
        self.prediction_manager.clear_word()
        self.prediction_manager.clear_prediction_buffer()
        self.confidence_manager.clear_history()
        logger.info("Agent reset")

    def __repr__(self) -> str:
        """String representation."""
        return f"SignLanguageAgent(word='{self.get_word()}', classes={len(self.labels)})"
