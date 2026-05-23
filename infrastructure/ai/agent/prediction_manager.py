"""
Prediction manager for buffering and stabilizing predictions.

This module maintains a buffer of recent predictions, implements stability
checking, and manages word accumulation with temporal constraints.
"""

import time
from collections import deque
from typing import Optional, List, Dict
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class PredictionManager:
    """
    Manager for handling prediction buffering and word accumulation.

    Implements temporal buffering to reduce noise and cooldown mechanisms
    to prevent rapid prediction changes.

    Attributes:
        buffer_size: Size of prediction buffer (default: 15)
        stability_threshold: Percentage needed for stable prediction (default: 0.40)
        word_cooldown: Minimum seconds between word additions (default: 1.5)
    """

    def __init__(self, buffer_size: int = 15, stability_threshold: float = 0.40, word_cooldown: float = 1.5):
        """
        Initialize prediction manager.

        Args:
            buffer_size: Number of recent predictions to maintain
            stability_threshold: Fraction of buffer needed for stability (0.0 to 1.0)
            word_cooldown: Minimum seconds between letter additions to word

        Example:
            >>> manager = PredictionManager(buffer_size=15, word_cooldown=1.5)
            >>> manager.add_prediction('A', 0.95)
        """
        self.buffer_size = buffer_size
        self.stability_threshold = max(0.0, min(1.0, stability_threshold))
        self.word_cooldown = word_cooldown

        self.prediction_buffer = deque(maxlen=buffer_size)
        self.word_buffer = []
        self.last_word_addition_time = 0
        self.last_added_letter = None

        logger.info(f"PredictionManager initialized: buffer_size={buffer_size}, cooldown={word_cooldown}s")

    def add_prediction(self, letter: str, confidence: float) -> None:
        """
        Add prediction to buffer.

        Args:
            letter: Predicted letter (A-Z)
            confidence: Confidence value (0.0 to 1.0)

        Example:
            >>> manager.add_prediction('A', 0.92)
        """
        confidence = max(0.0, min(1.0, confidence))
        self.prediction_buffer.append({'letter': letter, 'confidence': confidence})
        logger.debug(f"Prediction added: {letter} ({confidence:.4f})")

    def get_stable_prediction(self) -> Optional[tuple]:
        """
        Get stable prediction from buffer.

        A prediction is stable if it appears in more than stability_threshold
        of the recent predictions.

        Returns:
            Optional[tuple]: (letter, frequency, confidence) or None if unstable

        Example:
            >>> result = manager.get_stable_prediction()
            >>> if result:
            ...     letter, frequency, confidence = result
            ...     print(f"{letter} appears {frequency*100:.0f}% of the time")
        """
        if len(self.prediction_buffer) == 0:
            return None

        # Count occurrences of each letter
        letter_counts = {}
        letter_confidences = {}

        for pred in self.prediction_buffer:
            letter = pred['letter']
            confidence = pred['confidence']

            letter_counts[letter] = letter_counts.get(letter, 0) + 1
            if letter not in letter_confidences:
                letter_confidences[letter] = []
            letter_confidences[letter].append(confidence)

        # Find most common letter
        most_common = max(letter_counts.items(), key=lambda x: x[1])
        letter, count = most_common

        # Check if stable (appears in >40% of buffer)
        frequency = count / len(self.prediction_buffer)
        if frequency < self.stability_threshold:
            logger.debug(f"Unstable prediction: {letter} ({frequency*100:.0f}%)")
            return None

        # Calculate average confidence for this letter
        avg_confidence = sum(letter_confidences[letter]) / len(letter_confidences[letter])

        logger.debug(f"Stable prediction: {letter} ({frequency*100:.0f}%, confidence: {avg_confidence:.4f})")
        return (letter, frequency, avg_confidence)

    def add_to_word(self, letter: str) -> bool:
        """
        Add letter to word buffer with cooldown checking.

        Args:
            letter: Letter to add

        Returns:
            bool: True if added, False if cooldown still active

        Example:
            >>> if manager.add_to_word('H'):
            ...     print(f"Word so far: {''.join(manager.word_buffer)}")
        """
        current_time = time.time()
        time_since_last = current_time - self.last_word_addition_time

        # Check cooldown
        if time_since_last < self.word_cooldown:
            logger.debug(f"Cooldown active: {time_since_last:.2f}s < {self.word_cooldown}s")
            return False

        # Check if same letter (avoid duplicates)
        if letter == self.last_added_letter:
            logger.debug(f"Skipping duplicate letter: {letter}")
            return False

        self.word_buffer.append(letter)
        self.last_word_addition_time = current_time
        self.last_added_letter = letter

        logger.info(f"Letter added to word: {letter}. Word: {''.join(self.word_buffer)}")
        return True

    def get_word(self) -> str:
        """
        Get accumulated word from buffer.

        Returns:
            str: Current word as concatenated letters

        Example:
            >>> word = manager.get_word()
            >>> print(f"Current word: {word}")
        """
        return ''.join(self.word_buffer)

    def get_word_buffer(self) -> List[str]:
        """
        Get word buffer as list.

        Returns:
            List[str]: List of letters in current word

        Example:
            >>> letters = manager.get_word_buffer()
        """
        return self.word_buffer.copy()

    def clear_word(self) -> str:
        """
        Clear word buffer and return the cleared word.

        Returns:
            str: The word that was cleared

        Example:
            >>> word = manager.clear_word()
            >>> print(f"Cleared word: {word}")
        """
        cleared_word = ''.join(self.word_buffer)
        self.word_buffer.clear()
        self.last_added_letter = None
        logger.info(f"Word buffer cleared. Word was: {cleared_word}")
        return cleared_word

    def remove_last_letter(self) -> Optional[str]:
        """
        Remove last letter from word buffer (backspace).

        Returns:
            Optional[str]: The removed letter, or None if buffer is empty

        Example:
            >>> removed = manager.remove_last_letter()
            >>> if removed:
            ...     print(f"Removed: {removed}")
        """
        if not self.word_buffer:
            return None

        removed = self.word_buffer.pop()
        logger.info(f"Removed letter: {removed}. Word: {''.join(self.word_buffer)}")
        return removed

    def get_prediction_history(self, limit: int = 5) -> List[Dict]:
        """
        Get recent prediction history.

        Args:
            limit: Maximum number of predictions to return

        Returns:
            List[Dict]: List of recent predictions in order

        Example:
            >>> history = manager.get_prediction_history(limit=10)
            >>> for pred in history:
            ...     print(f"{pred['letter']}: {pred['confidence']:.2%}")
        """
        return list(self.prediction_buffer)[-limit:]

    def clear_prediction_buffer(self) -> None:
        """Clear prediction buffer."""
        self.prediction_buffer.clear()
        logger.debug("Prediction buffer cleared")

    def get_statistics(self) -> Dict:
        """
        Get statistics about current buffers.

        Returns:
            dict: Statistics including buffer sizes and letter frequencies

        Example:
            >>> stats = manager.get_statistics()
            >>> print(f"Most common letter: {stats['most_common_letter']}")
        """
        letter_counts = {}
        for pred in self.prediction_buffer:
            letter = pred['letter']
            letter_counts[letter] = letter_counts.get(letter, 0) + 1

        most_common = max(letter_counts.items(), key=lambda x: x[1]) if letter_counts else (None, 0)

        return {
            'buffer_size': len(self.prediction_buffer),
            'word_length': len(self.word_buffer),
            'word': self.get_word(),
            'most_common_letter': most_common[0],
            'most_common_count': most_common[1],
            'unique_letters': len(letter_counts),
            'letter_distribution': letter_counts
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"PredictionManager(word='{self.get_word()}', buffer={len(self.prediction_buffer)}/{self.buffer_size})"
