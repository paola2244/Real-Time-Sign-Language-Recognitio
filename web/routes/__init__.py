"""Route registration for the Flask web layer."""

from .dataset import dataset_bp
from .errors import register_error_handlers
from .history import history_bp
from .main import main_bp
from .prediction import prediction_bp
from .speech import speech_bp
from .status import status_bp
from .words import words_bp


def register_blueprints(app):
    """Register all web/API blueprints."""
    app.register_blueprint(main_bp)
    app.register_blueprint(status_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(words_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(speech_bp)
    register_error_handlers(app)

