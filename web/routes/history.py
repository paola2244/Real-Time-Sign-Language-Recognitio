"""Prediction history endpoints."""

from datetime import datetime

import numpy as np
from flask import Blueprint, jsonify, make_response, session

from database.models.prediction_model import Prediction
from infrastructure.ai.utils.logger import LoggerFactory
from web.services import get_prediction_repository

history_bp = Blueprint('history', __name__, url_prefix='/api/history')
logger = LoggerFactory.get_logger(__name__)


@history_bp.route('/save-to-db', methods=['POST'])
def api_history_save_to_db():
    """Save session history to the configured database."""
    try:
        session_predictions = session.get('predictions_history', [])

        if not session_predictions:
            return jsonify({
                'success': False,
                'error': 'No hay historial de predicciones para guardar'
            }), 400

        repo = get_prediction_repository()
        saved_count = 0
        skipped_count = 0

        for item in session_predictions:
            if item.get('saved_to_db'):
                skipped_count += 1
                continue

            prediction = Prediction(
                letter=str(item.get('letter', '')),
                confidence=float(item.get('confidence', 0.0)),
                session_id=session.get('session_id', ''),
                word_context=str(item.get('word_context', '')),
                timestamp=item.get('timestamp')
            )
            repo.create(prediction)
            item['saved_to_db'] = True
            saved_count += 1

        session['predictions_history'] = session_predictions
        session.modified = True

        return jsonify({
            'success': True,
            'saved_count': saved_count,
            'skipped_count': skipped_count,
            'message': f'{saved_count} prediccion(es) guardada(s)'
        })

    except Exception as e:
        logger.error(f"Error saving history to database: {e}")
        return jsonify({'error': str(e)}), 500


@history_bp.route('/data', methods=['GET'])
def api_history_data():
    """Return prediction history for the current view."""
    try:
        session_predictions = session.get('predictions_history', [])
        session_data = [{
            'letter': str(p.get('letter', '')),
            'confidence': float(p.get('confidence', 0.0)),
            'session_id': session.get('session_id', ''),
            'word_context': str(p.get('word_context', '')),
            'timestamp': p.get('timestamp', datetime.now().isoformat())
        } for p in session_predictions]

        try:
            predictions = get_prediction_repository().get_all(limit=100)
            data = [{
                'letter': str(p.letter),
                'confidence': float(p.confidence),
                'session_id': str(p.session_id),
                'word_context': str(p.word_context) if p.word_context else '',
                'timestamp': p.timestamp.isoformat() if hasattr(p.timestamp, 'isoformat') else str(p.timestamp)
            } for p in predictions]
        except Exception as db_error:
            logger.warning(f"Database access failed, using session history: {str(db_error)}")
            data = session_data

        if not data and session_data:
            data = session_data

        data = sorted(data, key=lambda p: p.get('timestamp', ''), reverse=True)[:100]
        confidences = [p['confidence'] for p in data] if data else []

        return jsonify({
            'predictions': data,
            'stats': {
                'total': len(data),
                'unique_letters': len(set(p['letter'] for p in data)) if data else 0,
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0
            }
        })

    except Exception as e:
        logger.error(f"Error in /api/history/data: {str(e)}")
        return jsonify({'error': 'Error retrieving history', 'predictions': [], 'stats': {'total': 0}}), 200


@history_bp.route('/csv', methods=['GET'])
def api_history_csv():
    """Download prediction history as CSV."""
    try:
        csv_content = 'Letter,Confidence,Session ID,Word Context,Timestamp\n'
        try:
            predictions = get_prediction_repository().get_all()
            for p in predictions:
                timestamp = p.timestamp.isoformat() if hasattr(p.timestamp, 'isoformat') else str(p.timestamp)
                csv_content += f'{p.letter},{p.confidence},{p.session_id},{p.word_context},{timestamp}\n'
        except Exception as db_error:
            logger.warning(f"Database access failed for CSV, using session history: {str(db_error)}")
            for p in session.get('predictions_history', []):
                csv_content += (
                    f"{p.get('letter', '')},{p.get('confidence', 0.0)},"
                    f"{session.get('session_id', '')},{p.get('word_context', '')},"
                    f"{p.get('timestamp', '')}\n"
                )

        response = make_response(csv_content)
        response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response

    except Exception as e:
        logger.error(f"Error in /api/history/csv: {e}")
        return jsonify({'error': str(e)}), 500


@history_bp.route('/clear', methods=['POST'])
def api_history_clear():
    """Clear prediction history."""
    try:
        session['predictions_history'] = []
        session['total_predictions'] = 0
        session.modified = True

        try:
            get_prediction_repository().clear_all()
        except Exception as db_error:
            logger.warning(f"Database access failed while clearing history: {str(db_error)}")

        return jsonify({
            'success': True,
            'message': 'History cleared'
        })

    except Exception as e:
        logger.error(f"Error in /api/history/clear: {e}")
        return jsonify({'error': str(e)}), 500

