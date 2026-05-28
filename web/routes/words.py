"""Current-word endpoints."""

from datetime import datetime

from flask import Blueprint, jsonify, request, session

from database.models.prediction_model import Prediction
from infrastructure.ai.utils.logger import LoggerFactory
from web.services import clear_loaded_word_buffers, get_agent, get_prediction_repository

words_bp = Blueprint('words', __name__, url_prefix='/api')
logger = LoggerFactory.get_logger(__name__)


@words_bp.route('/save-word', methods=['POST'])
def api_save_word():
    """Persist and clear the current word."""
    try:
        current_word = session.get('current_word', '')

        if not current_word:
            return jsonify({'error': 'No word to save'}), 400

        prediction = Prediction(
            letter=current_word,
            confidence=1.0,
            session_id=session.get('session_id'),
            word_context=current_word
        )
        get_prediction_repository().create(prediction)

        session['current_word'] = ''
        session.modified = True

        return jsonify({
            'success': True,
            'saved_word': current_word
        })

    except Exception as e:
        logger.error(f"Error saving word: {e}")
        return jsonify({'error': str(e)}), 500


@words_bp.route('/add-letter', methods=['POST'])
def api_add_letter():
    """Append one letter to the current word."""
    try:
        data = request.get_json() or {}
        letter = data.get('letter', '').upper()
        confidence = float(data.get('confidence') or 0.0)

        if not letter or len(letter) != 1:
            return jsonify({'error': 'Invalid letter'}), 400

        get_agent().add_letter_to_word(letter)

        current_word = session.get('current_word', '') + letter
        session['current_word'] = current_word
        session['total_predictions'] = session.get('total_predictions', 0) + 1
        session.setdefault('predictions_history', []).append({
            'letter': 'ESPACIO' if letter == ' ' else letter,
            'confidence': confidence,
            'word_context': current_word,
            'timestamp': datetime.now().isoformat()
        })
        session['predictions_history'] = session['predictions_history'][-100:]
        session.modified = True

        return jsonify({
            'success': True,
            'current_word': current_word
        })

    except Exception as e:
        logger.error(f"Error in /api/add-letter: {e}")
        return jsonify({'error': str(e)}), 500


@words_bp.route('/remove-letter', methods=['POST'])
def api_remove_letter():
    """Remove the last letter from the current word."""
    try:
        get_agent().undo_letter()

        current_word = session.get('current_word', '')
        if current_word:
            current_word = current_word[:-1]
        session['current_word'] = current_word
        session.modified = True

        return jsonify({
            'success': True,
            'current_word': current_word
        })

    except Exception as e:
        logger.error(f"Error in /api/remove-letter: {e}")
        return jsonify({'error': str(e)}), 500


@words_bp.route('/reset-word', methods=['POST'])
def api_reset_word():
    """Clear the current word."""
    try:
        cleared_word = session.get('current_word', '')
        session['current_word'] = ''
        session.modified = True
        clear_loaded_word_buffers()

        return jsonify({
            'success': True,
            'cleared_word': cleared_word,
            'current_word': ''
        })

    except Exception as e:
        logger.error(f"Error in /api/reset-word: {e}")
        return jsonify({'error': str(e)}), 500

