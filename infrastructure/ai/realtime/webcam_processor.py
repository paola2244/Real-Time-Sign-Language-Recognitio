"""
Webcam processing with hand detection using MediaPipe.

This module integrates MediaPipe for real-time hand detection and landmark
extraction from video frames.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "MediaPipe no está instalado. "
        "Instálalo con: pip install mediapipe>=0.10.0"
    ) from e

from .frame_extractor import FrameExtractor
from ..utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class WebcamProcessor:
    """
    Processes webcam frames and detects hands using MediaPipe.
    Supports both old (mp.solutions.hands) and new (mp.tasks) APIs.
    """

    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.5):
        """
        Initialize webcam processor with MediaPipe (compatible with both APIs).
        """
        self.use_tasks_api = False
        self.hands = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.initialized = False

        try:
            # Preferir API antigua (solutions) - más estable
            if hasattr(mp, 'solutions'):
                logger.info("Usando MediaPipe solutions API (recomendado)")
                self._init_solutions_api(min_detection_confidence, min_tracking_confidence)
                self.initialized = True
            else:
                logger.warning("MediaPipe solutions API no disponible, intentando tasks API...")
                # Solo intentar tasks si solutions no está disponible
                if hasattr(mp, 'tasks'):
                    try:
                        self._init_tasks_api(min_detection_confidence)
                        self.use_tasks_api = True
                        self.initialized = True
                    except Exception as e:
                        logger.warning(f"Tasks API falló: {e}. Detección de manos deshabilitada.")
                        self.initialized = False
                else:
                    logger.warning("MediaPipe sin API solutions ni tasks. Detección de manos deshabilitada.")
                    self.initialized = False

            if self.initialized:
                logger.info(f"WebcamProcessor inicializado correctamente (detection_conf={min_detection_confidence})")
            else:
                logger.warning("WebcamProcessor creado sin capacidad de detección. Use imágenes preprocessadas.")

        except Exception as e:
            logger.error(f"Error en inicialización de MediaPipe: {str(e)}")
            self.initialized = False

    def _init_solutions_api(self, min_detection_confidence, min_tracking_confidence):
        """Initialize using old mp.solutions API."""
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp_hands

    def _init_tasks_api(self, min_detection_confidence):
        """Initialize using new mp.tasks API (solo si el modelo existe)."""
        from pathlib import Path

        try:
            from mediapipe.tasks.python import vision

            MODEL_PATH = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "models"
                / "hand_landmarker.task"
            )

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = vision.HandLandmarker
            HandLandmarkerOptions = vision.HandLandmarkerOptions
            RunningMode = vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=RunningMode.IMAGE,
                min_hand_detection_confidence=min_detection_confidence
            )

            self.hands = HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
            logger.info("Tasks API inicializado correctamente")

        except Exception as e:
            logger.warning(f"No se pudo inicializar Tasks API: {e}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Process frame for hand detection and extract landmarks.
        Returns frame as-is if MediaPipe no está inicializado.
        Now returns hand landmarks instead of ROI for landmarks-based prediction.
        """
        if not self.initialized:
            logger.debug("Hand detection no inicializado. Retornando frame sin procesamiento.")
            return frame.copy(), None, 0.0

        if not self.use_tasks_api:
            # Legacy API available - use it
            return self._process_frame_solutions(frame)
        else:
            # Tasks API - return frame without annotations
            return frame.copy(), None, 0.0

    def _process_frame_solutions(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """Process using legacy mp.solutions API. Returns hand landmarks instead of ROI."""
        h, w, c = frame.shape

        # Flip frame for selfie view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        annotated_frame = frame.copy()
        selected_hand_landmarks = None
        max_confidence = 0.0

        try:
            results = self.hands.process(rgb_frame)
        except Exception as e:
            logger.debug(f"MediaPipe processing error: {str(e)[:100]}")
            return annotated_frame, None, 0.0

        if results.multi_hand_landmarks and results.multi_handedness:
            # Seleccionar la mano con mayor confianza
            best_hand_idx = 0
            best_confidence = 0.0

            for idx, hand_info in enumerate(results.multi_handedness):
                confidence = hand_info.classification[0].score
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_hand_idx = idx

            selected_hand_landmarks = results.multi_hand_landmarks[best_hand_idx]
            max_confidence = best_confidence

            # Dibujar landmarks y conexiones
            try:
                # Dibujar con MediaPipe
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    selected_hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            except Exception as e:
                logger.debug(f"MediaPipe drawing error: {str(e)[:50]}")
                # Fallback: Dibujar puntos manualmente
                try:
                    h, w = annotated_frame.shape[:2]
                    for lm in selected_hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)
                except Exception as e2:
                    logger.debug(f"Manual drawing error: {str(e2)[:50]}")

            # Dibujar información adicional
            try:
                result = FrameExtractor.extract_hand_region(annotated_frame, selected_hand_landmarks)
                if result:
                    _, (x_min, y_min, x_max, y_max) = result
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Conf: {max_confidence:.2%}",
                              (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                logger.debug(f"Bounding box error: {str(e)[:50]}")

        logger.debug(f"Frame processed: confidence={max_confidence:.4f}, landmarks={'found' if selected_hand_landmarks is not None else 'not found'}")
        return annotated_frame, selected_hand_landmarks, max_confidence

    def get_hand_landmarks(self, frame: np.ndarray) -> Optional[List]:
        """
        Get raw hand landmarks from frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Optional[List]: List of hand landmark objects, or None if no hands detected

        Example:
            >>> landmarks = processor.get_hand_landmarks(frame)
            >>> if landmarks:
            ...     for hand in landmarks:
            ...         print(f"Hand with {len(hand.landmark)} landmarks")
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks

            return None

        except Exception as e:
            logger.error(f"Error getting hand landmarks: {str(e)}")
            return None

    def draw_landmarks_on_frame(self, frame: np.ndarray, landmarks: Optional[List]) -> np.ndarray:
        """
        Draw hand landmarks on frame.

        Args:
            frame: Input frame (BGR)
            landmarks: Hand landmarks list

        Returns:
            np.ndarray: Frame with drawn landmarks

        Example:
            >>> landmarks = processor.get_hand_landmarks(frame)
            >>> annotated = processor.draw_landmarks_on_frame(frame, landmarks)
        """
        if landmarks is None:
            return frame

        annotated = frame.copy()

        try:
            for hand_landmarks in landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        except Exception as e:
            logger.error(f"Error drawing landmarks: {str(e)}")

        return annotated

    def detect_hands(self, frame: np.ndarray) -> Tuple[bool, int, Optional[List]]:
        """
        Detect hands in frame and return detection info.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple[bool, int, Optional[List]]:
                - Hands detected (bool)
                - Number of hands
                - Hand landmarks, or None

        Example:
            >>> detected, num_hands, landmarks = processor.detect_hands(frame)
            >>> if detected:
            ...     print(f"Found {num_hands} hand(s)")
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                return True, len(results.multi_hand_landmarks), results.multi_hand_landmarks

            return False, 0, None

        except Exception as e:
            logger.error(f"Hand detection error: {str(e)}")
            return False, 0, None

    def close(self):
        """Release resources."""
        self.hands.close()
        logger.info("WebcamProcessor closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return "WebcamProcessor(MediaPipe Hands)"
