"""
Real-time predictor for continuous sign language recognition.

This module combines webcam processing, frame extraction, and model inference
to provide real-time predictions with temporal smoothing.
"""

import time
import numpy as np
from collections import deque
from typing import Tuple, Optional
from .webcam_processor import WebcamProcessor
from .frame_extractor import FrameExtractor
from ..agent.sign_language_agent import SignLanguageAgent
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class RealtimePredictor:
    """
    Real-time predictor combining all components for live sign recognition.

    Integrates:
    - Webcam processing with MediaPipe hand detection
    - Frame extraction and preprocessing
    - CNN model inference
    - Temporal smoothing of predictions

    Attributes:
        webcam_processor: Hand detector
        sign_language_agent: Main AI agent
        prediction_buffer: Buffer for temporal smoothing
        frame_count: Frame counter for FPS calculation
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        confidence_threshold: float = 0.75,
        temporal_buffer_size: int = 10,
        target_fps: int = 30
    ):
        """
        Initialize real-time predictor.

        Args:
            model_path: Path to trained model
            labels_path: Path to labels.json
            confidence_threshold: Minimum confidence threshold
            temporal_buffer_size: Size of temporal smoothing buffer
            target_fps: Target frames per second

        Example:
            >>> predictor = RealtimePredictor('trained_models/best_model.keras', 'trained_models/labels.json')
            >>> while True:
            ...     ret, frame = cap.read()
            ...     result = predictor.process_and_predict(frame)
        """
        self.webcam_processor = WebcamProcessor()
        self.sign_language_agent = SignLanguageAgent(model_path, labels_path, confidence_threshold)

        self.temporal_buffer = deque(maxlen=temporal_buffer_size)
        self.target_fps = target_fps
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        logger.info(f"RealtimePredictor initialized with target FPS: {target_fps}")

    def process_and_predict(self, frame: np.ndarray) -> Tuple[Optional[str], float, np.ndarray, dict]:
        """
        Process frame and predict hand sign using landmarks.

        Args:
            frame: Input frame from webcam (BGR)

        Returns:
            Tuple containing:
                - Predicted letter (or None if no hands/low confidence)
                - Confidence value
                - Annotated frame with landmarks
                - Metadata dict with detection info

        Example:
            >>> while True:
            ...     ret, frame = cap.read()
            ...     letter, conf, annotated, meta = predictor.process_and_predict(frame)
            ...     if letter:
            ...         print(f"Detected: {letter} ({conf:.1%})")
        """
        # Start timing
        start_time = time.time()

        # Process frame for hand detection
        annotated_frame = frame.copy()
        hand_landmarks = None
        detection_confidence = 0.0
        processing_error = None

        try:
            annotated_frame, hand_landmarks, detection_confidence = self.webcam_processor.process_frame(frame)
            logger.debug(f"Frame processed: landmarks={hand_landmarks is not None}, conf={detection_confidence:.2%}")
        except Exception as e:
            error_msg = str(e)[:100]
            logger.debug(f"MediaPipe processing error (graceful fallback): {error_msg}")
            processing_error = error_msg
            # Continue with frame as-is on error

        letter = None
        confidence = 0.0

        # If hand detected with sufficient confidence
        if hand_landmarks is not None and detection_confidence > 0.5:
            try:
                logger.debug(f"Processing landmarks for prediction (detection_conf={detection_confidence:.2%})")
                # Get prediction from agent using landmarks
                letter, confidence = self.sign_language_agent.predict_from_landmarks(hand_landmarks)

                if letter is not None:
                    logger.info(f"Detected letter: {letter} (confidence={confidence:.2%})")
                    self.temporal_buffer.append({'letter': letter, 'confidence': confidence})
                else:
                    logger.debug(f"Low confidence prediction (conf={confidence:.2%})")

            except Exception as e:
                logger.error(f"Prediction error: {str(e)[:100]}")
                # Continue with no prediction

        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.last_fps_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = time.time()

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create metadata
        metadata = {
            'hands_detected': hand_landmarks is not None,
            'detection_confidence': float(detection_confidence),
            'prediction_confidence': float(confidence),
            'processing_time_ms': float(processing_time * 1000),
            'fps': float(self.fps),
            'buffer_size': len(self.temporal_buffer),
            'word': self.sign_language_agent.get_word(),
            'processing_error': processing_error
        }

        return letter, confidence, annotated_frame, metadata

    def batch_process(self, frames: list) -> list:
        """
        Process multiple frames.

        Args:
            frames: List of frames (BGR)

        Returns:
            list: List of (letter, confidence, annotated_frame, metadata) tuples

        Example:
            >>> results = predictor.batch_process(frames)
            >>> for letter, conf, frame, meta in results:
            ...     print(f"{letter}: {conf:.2%}")
        """
        results = []
        for frame in frames:
            result = self.process_and_predict(frame)
            results.append(result)
        return results

    def draw_prediction_on_frame(
        self,
        frame: np.ndarray,
        letter: Optional[str],
        confidence: float,
        position: Tuple[int, int] = (30, 80)
    ) -> np.ndarray:
        """
        Draw prediction text on frame.

        Args:
            frame: Input frame
            letter: Predicted letter
            confidence: Confidence value
            position: (x, y) text position

        Returns:
            np.ndarray: Frame with drawn prediction

        Example:
            >>> annotated = predictor.draw_prediction_on_frame(frame, 'A', 0.92)
        """
        import cv2

        annotated = frame.copy()
        x, y = position

        if letter:
            # Draw letter
            cv2.putText(
                annotated,
                f"Letter: {letter}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 255, 0),
                3
            )

            # Draw confidence
            cv2.putText(
                annotated,
                f"Confidence: {confidence:.1%}",
                (x, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Draw confidence bar
            bar_width = int(confidence * 200)
            cv2.rectangle(annotated, (x, y + 80), (x + 200, y + 100), (200, 200, 200), 2)
            cv2.rectangle(annotated, (x, y + 80), (x + bar_width, y + 100), (0, 255, 0), -1)
        else:
            cv2.putText(
                annotated,
                "No hand detected",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        return annotated

    def add_letter_to_word(self, letter: str) -> bool:
        """
        Add letter to accumulated word.

        Args:
            letter: Letter to add

        Returns:
            bool: True if added (cooldown respected)

        Example:
            >>> if predictor.add_letter_to_word('H'):
            ...     print(f"Word: {predictor.get_word()}")
        """
        return self.sign_language_agent.add_letter_to_word(letter)

    def get_word(self) -> str:
        """Get accumulated word."""
        return self.sign_language_agent.get_word()

    def clear_word(self) -> str:
        """Clear word and return it."""
        return self.sign_language_agent.clear_word()

    def undo_letter(self) -> Optional[str]:
        """Remove last letter from word."""
        return self.sign_language_agent.undo_letter()

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set new confidence threshold."""
        self.sign_language_agent.set_confidence_threshold(threshold)

    def reset(self):
        """Reset predictor and buffers."""
        self.sign_language_agent.reset()
        self.temporal_buffer.clear()
        logger.info("RealtimePredictor reset")

    def close(self):
        """Release resources."""
        self.webcam_processor.close()
        logger.info("RealtimePredictor closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"RealtimePredictor(word='{self.get_word()}', fps={self.fps:.1f})"
