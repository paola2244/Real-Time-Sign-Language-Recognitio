"""
Confidence manager for prediction validation.

This module handles confidence thresholding, weighted averaging, and
validation of model predictions.
"""

import numpy as np
from collections import deque
from typing import Optional, List
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class ConfidenceManager:
    """
    Manager for handling prediction confidence and thresholding.

    Attributes:
        confidence_threshold: Minimum confidence required to accept prediction
        history_size: Number of recent confidences to track
    """

    def __init__(self, confidence_threshold: float = 0.75, history_size: int = 5):
        """
        Initialize confidence manager.

        Args:
            confidence_threshold: Minimum confidence value (0.0 to 1.0)
            history_size: Number of recent confidences to maintain

        Example:
            >>> manager = ConfidenceManager(confidence_threshold=0.80)
            >>> is_confident = manager.is_confident(0.85)
        """
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.history_size = history_size
        self.confidence_history = deque(maxlen=history_size)

        logger.info(f"ConfidenceManager initialized with threshold: {self.confidence_threshold}")

    def is_confident(self, confidence: float) -> bool:
        """
        Check if confidence value meets threshold.

        Args:
            confidence: Confidence value between 0.0 and 1.0

        Returns:
            bool: True if confidence >= threshold

        Example:
            >>> if manager.is_confident(0.85):
            ...     print("High confidence prediction")
        """
        is_above_threshold = confidence >= self.confidence_threshold
        logger.debug(f"Confidence check: {confidence:.4f} >= {self.confidence_threshold:.4f} = {is_above_threshold}")
        return is_above_threshold

    def add_to_history(self, confidence: float) -> None:
        """
        Add confidence value to history for temporal tracking.

        Args:
            confidence: Confidence value to track

        Example:
            >>> manager.add_to_history(0.92)
        """
        confidence = max(0.0, min(1.0, confidence))
        self.confidence_history.append(confidence)
        logger.debug(f"Added confidence to history: {confidence:.4f}")

    def get_average_confidence(self) -> float:
        """
        Get average confidence from history.

        Returns:
            float: Average confidence, or 0.0 if no history

        Example:
            >>> avg = manager.get_average_confidence()
            >>> print(f"Average confidence: {avg:.4f}")
        """
        if not self.confidence_history:
            return 0.0
        return float(np.mean(list(self.confidence_history)))

    def get_max_confidence(self) -> float:
        """
        Get maximum confidence from history.

        Returns:
            float: Maximum confidence, or 0.0 if no history

        Example:
            >>> max_conf = manager.get_max_confidence()
        """
        if not self.confidence_history:
            return 0.0
        return float(np.max(list(self.confidence_history)))

    def get_min_confidence(self) -> float:
        """
        Get minimum confidence from history.

        Returns:
            float: Minimum confidence, or 0.0 if no history
        """
        if not self.confidence_history:
            return 0.0
        return float(np.min(list(self.confidence_history)))

    def calculate_weighted_confidence(self, confidences: List[float]) -> float:
        """
        Calculate weighted average of confidences.

        More recent confidences are weighted higher. Uses exponential weighting
        where weight = exp(index / total_items).

        Args:
            confidences: List of confidence values

        Returns:
            float: Weighted average confidence

        Example:
            >>> confidences = [0.70, 0.75, 0.80, 0.85, 0.90]
            >>> weighted = manager.calculate_weighted_confidence(confidences)
        """
        if not confidences:
            return 0.0

        confidences = np.array([max(0.0, min(1.0, c)) for c in confidences])
        n = len(confidences)

        # Exponential weights: more recent = higher weight
        weights = np.exp(np.arange(n) / n)
        weights /= weights.sum()

        weighted_avg = float(np.average(confidences, weights=weights))
        logger.debug(f"Calculated weighted confidence: {weighted_avg:.4f}")

        return weighted_avg

    def get_confidence_trend(self) -> str:
        """
        Analyze confidence trend (increasing, decreasing, stable).

        Returns:
            str: "increasing", "decreasing", or "stable"

        Example:
            >>> trend = manager.get_confidence_trend()
            >>> print(f"Confidence trend: {trend}")
        """
        if len(self.confidence_history) < 2:
            return "insufficient_data"

        history_list = list(self.confidence_history)
        first_half = np.mean(history_list[:len(history_list)//2])
        second_half = np.mean(history_list[len(history_list)//2:])

        threshold = 0.02  # 2% difference threshold
        diff = second_half - first_half

        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        else:
            return "stable"

    def clear_history(self) -> None:
        """Clear confidence history."""
        self.confidence_history.clear()
        logger.debug("Confidence history cleared")

    def get_history(self) -> List[float]:
        """
        Get current confidence history.

        Returns:
            List[float]: List of recent confidence values

        Example:
            >>> history = manager.get_history()
            >>> print(f"Last {len(history)} confidences: {history}")
        """
        return list(self.confidence_history)

    def set_threshold(self, new_threshold: float) -> None:
        """
        Update confidence threshold.

        Args:
            new_threshold: New threshold value (0.0 to 1.0)

        Example:
            >>> manager.set_threshold(0.85)
        """
        self.confidence_threshold = max(0.0, min(1.0, new_threshold))
        logger.info(f"Confidence threshold updated to: {self.confidence_threshold}")

    def get_confidence_level(self, confidence: float) -> str:
        """
        Get human-readable confidence level.

        Args:
            confidence: Confidence value

        Returns:
            str: "very_low", "low", "medium", "high", or "very_high"

        Example:
            >>> level = manager.get_confidence_level(0.92)
            >>> print(f"Confidence level: {level}")  # "very_high"
        """
        if confidence < 0.2:
            return "very_low"
        elif confidence < 0.4:
            return "low"
        elif confidence < 0.6:
            return "medium"
        elif confidence < 0.8:
            return "high"
        else:
            return "very_high"

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfidenceManager(threshold={self.confidence_threshold:.2f}, history_size={len(self.confidence_history)})"
