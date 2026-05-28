"""Text-to-speech endpoint."""

from flask import Blueprint, jsonify, request, session

from infrastructure.ai.utils.logger import LoggerFactory

speech_bp = Blueprint('speech', __name__, url_prefix='/api')
logger = LoggerFactory.get_logger(__name__)


@speech_bp.route('/speak', methods=['POST'])
def api_speak():
    """Return text to be spoken by the browser."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip() if data else ''

        if not text:
            text = session.get('current_word', '').strip()

        if not text:
            return jsonify({'error': 'No text to speak'}), 400

        session['current_word'] = ''
        session.modified = True

        return jsonify({
            'success': True,
            'spoken_text': text,
            'speaker': 'browser'
        })

    except Exception as e:
        logger.error(f"Error in /api/speak: {e}")
        return jsonify({'error': str(e)}), 500
