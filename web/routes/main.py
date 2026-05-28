"""Page routes."""

from flask import Blueprint, render_template

from web.services import check_model_files

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Main single-page application."""
    return render_template('index.html', model_exists=check_model_files())


@main_bp.route('/about')
def about():
    """Project information page."""
    return render_template('about.html')

