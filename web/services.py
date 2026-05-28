"""Shared services used by Flask routes."""

import base64
import uuid
from datetime import datetime
from pathlib import Path

import cv2
from flask import current_app, session

from database.repositories.prediction_repository import PredictionRepository
from infrastructure.ai.agent.sign_language_agent import SignLanguageAgent
from infrastructure.ai.realtime.realtime_predictor import RealtimePredictor
from infrastructure.ai.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

_agent = None
_realtime_predictor = None
_prediction_repository = None


def init_session_defaults():
    """Initialize expected session fields for a new user session."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['start_time'] = datetime.now().isoformat()
        session['current_word'] = ''
        session['confidence_threshold'] = current_app.config['CONFIDENCE_THRESHOLD']
        session['total_predictions'] = 0
        session['predictions_history'] = []
        logger.info(f"New session initialized: {session['session_id']}")


def check_model_files():
    """Return True when model and label files exist."""
    model_path = current_app.config['MODEL_PATH']
    labels_path = current_app.config['LABELS_PATH']
    return model_path.exists() and labels_path.exists()


def allowed_file(filename):
    """Validate uploaded file extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def image_to_base64(image_array):
    """Convert an OpenCV image array to a base64 PNG string."""
    if image_array is None:
        return None
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')


def get_agent():
    """Return the singleton SignLanguageAgent."""
    global _agent
    if _agent is None:
        try:
            _agent = SignLanguageAgent(
                model_path=str(current_app.config['MODEL_PATH']),
                labels_path=str(current_app.config['LABELS_PATH']),
                confidence_threshold=current_app.config['CONFIDENCE_THRESHOLD']
            )
            logger.info("SignLanguageAgent initialized")
        except Exception as e:
            logger.error(f"Error initializing SignLanguageAgent: {e}")
            raise
    return _agent


def get_realtime_predictor():
    """Return the singleton real-time predictor."""
    global _realtime_predictor
    if _realtime_predictor is None:
        try:
            _realtime_predictor = RealtimePredictor(
                model_path=str(current_app.config['MODEL_PATH']),
                labels_path=str(current_app.config['LABELS_PATH']),
                confidence_threshold=current_app.config['CONFIDENCE_THRESHOLD']
            )
            logger.info("RealtimePredictor initialized")
        except Exception as e:
            logger.error(f"Error initializing RealtimePredictor: {e}")
            raise
    return _realtime_predictor


def get_prediction_repository():
    """Return the singleton prediction repository."""
    global _prediction_repository
    if _prediction_repository is None:
        try:
            _prediction_repository = PredictionRepository()
            logger.info("PredictionRepository initialized")
        except Exception as e:
            logger.error(f"Error initializing PredictionRepository: {e}")
            raise
    return _prediction_repository


def clear_loaded_word_buffers():
    """Clear model-side word buffers without forcing model initialization."""
    if _agent is not None:
        _agent.clear_word()
    if _realtime_predictor is not None:
        _realtime_predictor.clear_word()


def collected_data_path(letter=None):
    """Return the collected-data path, optionally for one label."""
    base_path = Path('data/collected_data')
    return base_path / letter if letter else base_path

