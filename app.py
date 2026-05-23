import os
import json
import base64
import uuid
from datetime import datetime
from pathlib import Path
from io import BytesIO
from functools import lru_cache

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from PIL import Image
import cv2
import numpy as np

from config import config, Config
from infrastructure.ai.agent.sign_language_agent import SignLanguageAgent
from infrastructure.ai.realtime.realtime_predictor import RealtimePredictor
from infrastructure.ai.realtime.webcam_processor import WebcamProcessor
from infrastructure.ai.utils.image_utils import prepare_for_inference
from infrastructure.ai.utils.logger import LoggerFactory
from database.repositories.prediction_repository import PredictionRepository
from database.models.prediction_model import Prediction

# Configuración
app = Flask(__name__, template_folder='templates', static_folder='static')
env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env])
Session(app)

logger = LoggerFactory.get_logger(__name__)

# Singletons
_agent = None
_realtime_predictor = None
_prediction_repository = None


def get_agent():
    """Obtiene instancia singleton de SignLanguageAgent"""
    global _agent
    if _agent is None:
        try:
            _agent = SignLanguageAgent(
                model_path=str(app.config['MODEL_PATH']),
                labels_path=str(app.config['LABELS_PATH']),
                confidence_threshold=app.config['CONFIDENCE_THRESHOLD']
            )
            logger.info("SignLanguageAgent initialized")
        except Exception as e:
            logger.error(f"Error initializing SignLanguageAgent: {e}")
            raise
    return _agent


def get_realtime_predictor():
    """Obtiene instancia singleton de RealtimePredictor"""
    global _realtime_predictor
    if _realtime_predictor is None:
        try:
            _realtime_predictor = RealtimePredictor(
                model_path=str(app.config['MODEL_PATH']),
                labels_path=str(app.config['LABELS_PATH']),
                confidence_threshold=app.config['CONFIDENCE_THRESHOLD']
            )
            logger.info("RealtimePredictor initialized")
        except Exception as e:
            logger.error(f"Error initializing RealtimePredictor: {e}")
            raise
    return _realtime_predictor


def get_prediction_repository():
    """Obtiene instancia singleton de PredictionRepository"""
    global _prediction_repository
    if _prediction_repository is None:
        try:
            _prediction_repository = PredictionRepository()
            logger.info("PredictionRepository initialized")
        except Exception as e:
            logger.error(f"Error initializing PredictionRepository: {e}")
            raise
    return _prediction_repository


@app.before_request
def init_session():
    """Inicializa la sesión del usuario"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['start_time'] = datetime.now().isoformat()
        session['current_word'] = ''
        session['confidence_threshold'] = app.config['CONFIDENCE_THRESHOLD']
        session['total_predictions'] = 0
        session['predictions_history'] = []
        logger.info(f"New session initialized: {session['session_id']}")


def check_model_files():
    """Verifica que los archivos del modelo existan"""
    model_path = app.config['MODEL_PATH']
    labels_path = app.config['LABELS_PATH']
    return model_path.exists() and labels_path.exists()


def allowed_file(filename):
    """Verifica si el archivo tiene extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def image_to_base64(image_array):
    """Convierte array de imagen OpenCV a string base64"""
    if image_array is None:
        return None
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')


def process_image_for_prediction(image_array, save_to_db=True):
    """Procesa imagen y realiza predicción"""
    try:
        predictor = get_realtime_predictor()
        agent = get_agent()

        # Procesar frame y obtener predicción
        # process_and_predict retorna: (letter, confidence, annotated_frame, metadata)
        letter, confidence, annotated_frame, metadata = predictor.process_and_predict(image_array)

        if letter is not None:
            # Guardar predicción en sesión
            session['total_predictions'] = session.get('total_predictions', 0) + 1
            session['predictions_history'].append({
                'letter': letter,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            })


            # Dibujar landmarks si existen
            landmarks_image = image_to_base64(annotated_frame) if annotated_frame is not None else None

            return {
                'success': True,
                'letter': letter,
                'confidence': float(confidence),
                'landmarks_image': landmarks_image,
                'metadata': metadata
            }
        else:
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


# ==================== RUTAS ====================

@app.route('/')
def index():
    """Página principal con SPA"""
    return render_template('index.html', model_exists=check_model_files())


@app.route('/about')
def about():
    """Página de información del proyecto"""
    return render_template('about.html')


# ==================== API ENDPOINTS ====================

@app.route('/api/status', methods=['GET'])
def api_status():
    """Obtiene estado de la aplicación"""
    return jsonify({
        'model_exists': check_model_files(),
        'session_id': session.get('session_id'),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/session-state', methods=['GET'])
def api_session_state():
    """Obtiene el estado actual de la sesión"""
    return jsonify({
        'session_id': session.get('session_id'),
        'current_word': session.get('current_word', ''),
        'confidence_threshold': session.get('confidence_threshold', 0.75),
        'total_predictions': session.get('total_predictions', 0),
        'start_time': session.get('start_time')
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Procesa un frame de webcam y realiza predicción"""
    try:
        # Obtener imagen del request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Leer imagen
        image = Image.open(file.stream).convert('RGB')
        image_array = np.array(image)

        # Convertir BGR para OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Procesar y predecir
        result = process_image_for_prediction(image_array)

        return jsonify(result) 

    except Exception as e:
        logger.error(f"Error in /api/predict: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Procesa imagen subida por el usuario"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        # Leer y procesar imagen
        image = Image.open(file.stream).convert('RGB')
        image_array = np.array(image)

        # Convertir a BGR para OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Procesar y predecir
        result = process_image_for_prediction(image_array)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /api/upload: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/save-word', methods=['POST'])
def api_save_word():
    try:
        current_word = session.get('current_word', '')

        if not current_word:
            return jsonify({'error': 'No word to save'}), 400

        repo = get_prediction_repository()

        prediction = Prediction(
            letter=current_word,
            confidence=1.0,
            session_id=session.get('session_id'),
            word_context=current_word
        )

        repo.create(prediction)

        saved_word = current_word

        # limpiar palabra actual
        session['current_word'] = ''
        session.modified = True

        return jsonify({
            'success': True,
            'saved_word': saved_word
        })

    except Exception as e:
        logger.error(f"Error saving word: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/add-letter', methods=['POST'])
def api_add_letter():
    """Agrega una letra a la palabra actual"""
    try:
        data = request.get_json()
        letter = data.get('letter', '').upper()

        if not letter or len(letter) != 1:
            return jsonify({'error': 'Invalid letter'}), 400

        agent = get_agent()
        agent.add_letter_to_word(letter)

        current_word = session.get('current_word', '')
        current_word += letter
        session['current_word'] = current_word
        session.modified = True

        return jsonify({
            'success': True,
            'current_word': current_word
        })

    except Exception as e:
        logger.error(f"Error in /api/add-letter: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/remove-letter', methods=['POST'])
def api_remove_letter():
    """Remueve la última letra de la palabra actual"""
    try:
        agent = get_agent()
        agent.remove_last_letter()

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


@app.route('/api/reset-word', methods=['POST'])
def api_reset_word():
    """Reinicia la palabra actual"""
    try:
        agent = get_agent()
        cleared_word = agent.clear_word()

        session['current_word'] = ''
        session.modified = True

        return jsonify({
            'success': True,
            'cleared_word': cleared_word,
            'current_word': ''
        })

    except Exception as e:
        logger.error(f"Error in /api/reset-word: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/data', methods=['GET'])
def api_history_data():
    """Obtiene historial de predicciones"""
    try:
        try:
            repo = get_prediction_repository()
            predictions = repo.get_all(limit=100)

            data = [{
                'letter': str(p.letter),
                'confidence': float(p.confidence),
                'session_id': str(p.session_id),
                'word_context': str(p.word_context) if p.word_context else '',
                'timestamp': p.timestamp.isoformat() if hasattr(p.timestamp, 'isoformat') else str(p.timestamp)
            } for p in predictions]
        except Exception as db_error:
            logger.warning(f"Database access failed, returning empty history: {str(db_error)}")
            data = []

        confidences = [p['confidence'] for p in data] if data else []
        stats = {
            'total': len(data),
            'unique_letters': len(set(p['letter'] for p in data)) if data else 0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0
        }

        result = {
            'predictions': data,
            'stats': stats
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /api/history/data: {str(e)}")
        return jsonify({'error': 'Error retrieving history', 'predictions': [], 'stats': {'total': 0}}), 200


@app.route('/api/history/csv', methods=['GET'])
def api_history_csv():
    """Descarga historial como CSV"""
    try:
        repo = get_prediction_repository()
        predictions = repo.get_all()

        csv_content = 'Letter,Confidence,Session ID,Word Context,Timestamp\n'
        for p in predictions:
            timestamp = p.timestamp.isoformat() if hasattr(p.timestamp, 'isoformat') else str(p.timestamp)
            csv_content += f'{p.letter},{p.confidence},{p.session_id},{p.word_context},{timestamp}\n'

        from flask import make_response
        response = make_response(csv_content)
        response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response

    except Exception as e:
        logger.error(f"Error in /api/history/csv: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    """Limpia el historial de predicciones"""
    try:
        repo = get_prediction_repository()
        repo.clear_all()

        return jsonify({
            'success': True,
            'message': 'History cleared'
        })

    except Exception as e:
        logger.error(f"Error in /api/history/clear: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/collect-data', methods=['POST'])
def api_collect_data():
    """Recibe fotos para recopilar dataset de entrenamiento"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        letter = request.form.get('letter', '').upper()
        if not letter or len(letter) != 1 or not letter.isalpha():
            return jsonify({'error': 'Invalid letter'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Crear estructura de carpetas: data/collected_data/A/
        collected_data_dir = Path('data/collected_data') / letter
        collected_data_dir.mkdir(parents=True, exist_ok=True)

        # Generar nombre único basado en timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{letter}_{timestamp}.jpg"
        filepath = collected_data_dir / filename

        # Leer y guardar imagen
        image = Image.open(file.stream).convert('RGB')
        image.save(str(filepath), quality=95)

        # Contar imágenes en esta carpeta
        image_count = len(list(collected_data_dir.glob('*.jpg')))

        logger.info(f"Data collected: {letter} ({image_count} total images)")

        return jsonify({
            'success': True,
            'letter': letter,
            'images_count': image_count,
            'message': f'Foto guardada para la letra "{letter}" ({image_count} total)'
        })

    except Exception as e:
        logger.error(f"Error in /api/collect-data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/dataset-stats', methods=['GET'])
def api_dataset_stats():
    """Obtiene estadísticas del dataset recopilado"""
    try:
        collected_data_dir = Path('data/collected_data')
        stats = {}
        total_images = 0

        if collected_data_dir.exists():
            for letter_dir in sorted(collected_data_dir.iterdir()):
                if letter_dir.is_dir():
                    count = len(list(letter_dir.glob('*.jpg')))
                    if count > 0:
                        stats[letter_dir.name] = count
                        total_images += count

        return jsonify({
            'total_images': total_images,
            'letters_collected': len(stats),
            'by_letter': stats
        })

    except Exception as e:
        logger.error(f"Error in /api/dataset-stats: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Maneja errores 404"""
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Maneja errores 500"""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Maneja archivos demasiado grandes"""
    return jsonify({'error': 'File too large'}), 413


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        use_reloader=True
    )
