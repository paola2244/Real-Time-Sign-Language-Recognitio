"""Status and diagnostics endpoints."""

from datetime import datetime

from flask import Blueprint, jsonify, session

from database.connection import get_db
from infrastructure.ai.utils.logger import LoggerFactory
from web.services import check_model_files

status_bp = Blueprint('status', __name__, url_prefix='/api')
logger = LoggerFactory.get_logger(__name__)


@status_bp.route('/status', methods=['GET'])
def api_status():
    """Return application status."""
    return jsonify({
        'model_exists': check_model_files(),
        'session_id': session.get('session_id'),
        'timestamp': datetime.now().isoformat()
    })


@status_bp.route('/database-status', methods=['GET'])
def api_database_status():
    """Return database connection status."""
    try:
        db = get_db()
        return jsonify({
            'success': True,
            'backend': 'mongodb' if db.use_mongodb else 'json',
            'database': db.db_name,
            'configured_host': db.configured_host,
            'env_path': db.env_path,
            'message': db.connection_message
        })
    except Exception as e:
        logger.error(f"Error in /api/database-status: {e}")
        return jsonify({
            'success': False,
            'backend': 'unknown',
            'error': str(e)
        }), 500


@status_bp.route('/session-state', methods=['GET'])
def api_session_state():
    """Return current browser session state."""
    return jsonify({
        'session_id': session.get('session_id'),
        'current_word': session.get('current_word', ''),
        'confidence_threshold': session.get('confidence_threshold', 0.75),
        'total_predictions': session.get('total_predictions', 0),
        'start_time': session.get('start_time')
    })

