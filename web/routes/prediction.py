"""Image prediction endpoints."""

import cv2
import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image

from infrastructure.ai.utils.logger import LoggerFactory
from web.services import allowed_file, get_realtime_predictor, image_to_base64

prediction_bp = Blueprint('prediction', __name__, url_prefix='/api')
logger = LoggerFactory.get_logger(__name__)


def process_image_for_prediction(image_array, confidence_threshold=None):
    """Run the real-time predictor against one image array."""
    try:
        predictor = get_realtime_predictor()
        if confidence_threshold is not None:
            threshold = max(0.0, min(1.0, float(confidence_threshold)))
            predictor.set_confidence_threshold(threshold)

        letter, confidence, annotated_frame, metadata = predictor.process_and_predict(image_array)

        if letter is not None:
            return {
                'success': True,
                'letter': letter,
                'confidence': float(confidence),
                'landmarks_image': None,
                'metadata': metadata
            }

        return {
            'success': True,
            'letter': 'N/A',
            'confidence': float(confidence) if confidence else 0.0,
            'landmarks_image': image_to_base64(annotated_frame) if annotated_frame is not None else None,
            'metadata': metadata
        }

    except Exception as e:
        logger.error(f"Error in process_image_for_prediction: {e}")
        return {
            'success': False,
            'error': str(e),
            'letter': 'N/A',
            'confidence': 0.0
        }


def _request_image_array(file_storage):
    """Read an uploaded image as an OpenCV BGR array."""
    image = Image.open(file_storage.stream).convert('RGB')
    image_array = np.array(image)
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


@prediction_bp.route('/predict', methods=['POST'])
def api_predict():
    """Process one webcam frame."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        result = process_image_for_prediction(
            _request_image_array(file),
            confidence_threshold=request.form.get('confidence_threshold')
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /api/predict: {e}")
        return jsonify({'error': str(e)}), 500


@prediction_bp.route('/upload', methods=['POST'])
def api_upload():
    """Process a user-uploaded image."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        result = process_image_for_prediction(_request_image_array(file))
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /api/upload: {e}")
        return jsonify({'error': str(e)}), 500
