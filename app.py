"""Flask entrypoint for the sign language recognition app."""

import os

from flask import Flask
from flask_session import Session

from config import config
from web.routes import register_blueprints
from web.services import init_session_defaults


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    env = os.getenv('FLASK_ENV', 'development')
    app.config.from_object(config[env])

    Session(app)

    @app.before_request
    def init_session():
        init_session_defaults()

    register_blueprints(app)
    return app


app = create_app()


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=app.config['DEBUG'],
        use_reloader=True
    )
