import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración base de Flask"""

    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'

    # Session
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = 7 * 24 * 60 * 60  # 7 días

    # Upload
    UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

    # Modelos
    # Landmarks-based model (new - preferred)
    # NOTA: Ruta sin caracteres especiales para evitar problemas de encoding en Windows
    MODEL_PATH = Path(__file__).parent / 'trained_models' / 'best_model_hybrid.h5'
    LABELS_PATH = Path(__file__).parent / 'trained_models' / 'labels_hybrid.json'

    # Dataset type: 'landmarks' or 'mnist'
    DATASET_TYPE = 'landmarks'

    # Inferencia
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.75'))
    PREDICTION_BUFFER_SIZE = int(os.getenv('PREDICTION_BUFFER_SIZE', '15'))
    PREDICTION_STABILITY_THRESHOLD = float(os.getenv('PREDICTION_STABILITY_THRESHOLD', '0.4'))
    PREDICTION_COOLDOWN = float(os.getenv('PREDICTION_COOLDOWN', '1.5'))

    # Database
    MONGODB_URI = os.getenv('MONGO_URI', os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
    DATABASE_NAME = os.getenv('MONGO_DB_NAME', os.getenv('DATABASE_NAME', 'sign_language'))

    # Crear carpeta de uploads
    UPLOAD_FOLDER.mkdir(exist_ok=True)


class DevelopmentConfig(Config):
    """Configuración para desarrollo"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Configuración para producción"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Configuración para testing"""
    DEBUG = True
    TESTING = True
    MONGODB_URI = 'mongodb://localhost:27017/sign_language_test'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
