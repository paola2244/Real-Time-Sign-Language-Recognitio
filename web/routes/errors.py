"""Flask error handlers."""

from flask import jsonify

from infrastructure.ai.utils.logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def register_error_handlers(app):
    """Register application-level error handlers."""

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Page not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({'error': 'File too large'}), 413

