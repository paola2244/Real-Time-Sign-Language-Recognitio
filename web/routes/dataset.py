"""Dataset collection endpoints."""

from datetime import datetime

from flask import Blueprint, jsonify, request
from PIL import Image

from infrastructure.ai.utils.logger import LoggerFactory
from web.services import collected_data_path

dataset_bp = Blueprint('dataset', __name__, url_prefix='/api')
logger = LoggerFactory.get_logger(__name__)

VALID_LABELS = {'BORRAR', 'ESPACIO', 'ESCUCHAR'}


@dataset_bp.route('/collect-data', methods=['POST'])
def api_collect_data():
    """Receive images for the training dataset."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        letter = request.form.get('letter', '').upper()
        if not letter:
            return jsonify({'error': 'Invalid letter'}), 400
        if not ((len(letter) == 1 and letter.isalpha()) or letter in VALID_LABELS):
            return jsonify({'error': 'Invalid label'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        label_dir = collected_data_path(letter)
        label_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{letter}_{timestamp}.jpg"
        filepath = label_dir / filename

        image = Image.open(file.stream).convert('RGB')
        image.save(str(filepath), quality=95)

        image_count = len(list(label_dir.glob('*.jpg')))
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


@dataset_bp.route('/dataset-stats', methods=['GET'])
def api_dataset_stats():
    """Return collected dataset statistics."""
    try:
        collected_data_dir = collected_data_path()
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

