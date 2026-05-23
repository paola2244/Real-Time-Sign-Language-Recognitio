#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Sign Language Model with Collected Data.

Trains the model combining:
1. Original dataset (landmarks-based)
2. Newly collected data from the app

Usage:
    python scripts/train_model_with_collected_data.py
"""

import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.ai.models.cnn_model import create_hybrid_model
from infrastructure.ai.utils.logger import LoggerFactory
from infrastructure.ai.utils.image_utils import normalize_landmarks

logger = LoggerFactory.get_logger(__name__)


class CollectedDataProcessor:
    """Procesa las imágenes recopiladas para extraer landmarks."""

    def __init__(self):
        """Inicializa MediaPipe Hands."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def extract_landmarks_from_image(self, image_path: str) -> np.ndarray:
        """
        Extrae landmarks de una imagen.

        Args:
            image_path: Ruta a la imagen JPG

        Returns:
            np.ndarray: Array de shape (63,) con landmarks normalizados, o None si falla
        """
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"No se pudo leer imagen: {image_path}")
                return None

            # Convertir a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detectar landmarks
            results = self.hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                logger.debug(f"No landmarks detectados en: {image_path}")
                return None

            # Tomar la primera mano
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extraer coordenadas (x, y, z)
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.append([lm.x, lm.y, lm.z])

            landmarks_array = np.array(landmarks_list, dtype=np.float32).flatten()

            # Normalizar
            landmarks_normalized = normalize_landmarks(landmarks_array, standardize=True)

            return landmarks_normalized

        except Exception as e:
            logger.error(f"Error extrayendo landmarks de {image_path}: {e}")
            return None

    def load_collected_data(self, collected_data_dir: str = 'data/collected_data') -> tuple:
        """
        Carga todas las imágenes recopiladas.

        Args:
            collected_data_dir: Directorio con datos recopilados

        Returns:
            (X, y): Arrays de landmarks y etiquetas
        """
        collected_path = Path(collected_data_dir)
        if not collected_path.exists():
            logger.warning(f"No existe directorio: {collected_data_dir}")
            return np.array([]), np.array([])

        X = []
        y = []
        label_count = {}

        # Iterar por cada carpeta de letra
        for letter_dir in sorted(collected_path.iterdir()):
            if not letter_dir.is_dir():
                continue

            letter = letter_dir.name
            image_files = list(letter_dir.glob('*.jpg'))

            print(f'[*] Procesando letra "{letter}": {len(image_files)} imágenes...')

            for image_file in image_files:
                landmarks = self.extract_landmarks_from_image(str(image_file))

                if landmarks is not None:
                    X.append(landmarks)
                    y.append(letter)
                    label_count[letter] = label_count.get(letter, 0) + 1

        if len(X) == 0:
            logger.warning("No se pudieron cargar imágenes")
            return np.array([]), np.array([])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=str)

        print(f'[+] Datos recopilados cargados:')
        print(f'    Total imágenes: {len(X)}')
        print(f'    Distribución:')
        for letter in sorted(label_count.keys()):
            print(f'      {letter}: {label_count[letter]} imágenes')

        return X, y

    def __del__(self):
        """Libera recursos de MediaPipe."""
        if hasattr(self, 'hands'):
            self.hands.close()


class HybridDataTrainer:
    """Entrena el modelo con datos combinados."""

    def __init__(self, model_path: str = 'C:/models_prod/best_model_hybrid.h5',
                 output_dir: str = 'trained_models'):
        """
        Inicializa el entrenador.

        Args:
            model_path: Ruta del modelo actual
            output_dir: Directorio para guardar modelo mejorado
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_existing_model(self) -> tf.keras.Model:
        """Carga el modelo actual."""
        if not self.model_path.exists():
            logger.error(f"Modelo no existe: {self.model_path}")
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        model = tf.keras.models.load_model(str(self.model_path))
        logger.info(f"Modelo cargado: {self.model_path}")
        return model

    def prepare_training_data(self, X: np.ndarray, y: np.ndarray,
                             validation_split: float = 0.2) -> tuple:
        """
        Prepara datos para entrenamiento usando las 28 clases originales.

        Args:
            X: Features (landmarks)
            y: Labels (letras)
            validation_split: Proporción para validación

        Returns:
            (X_train, y_train, X_val, y_val, label_to_idx)
        """
        # Usar las 28 clases originales
        original_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']
        label_to_idx = {label: idx for idx, label in enumerate(original_labels)}
        num_classes = len(original_labels)

        y_numeric = np.array([label_to_idx.get(label, 0) for label in y], dtype=np.int32)

        # One-hot encoding
        y_encoded = to_categorical(y_numeric, num_classes=num_classes)

        # Split train/validation
        num_samples = len(X)
        indices = np.random.permutation(num_samples)
        num_val = int(num_samples * validation_split)

        val_indices = indices[:num_val]
        train_indices = indices[num_val:]

        X_train = X[train_indices]
        y_train = y_encoded[train_indices]
        X_val = X[val_indices]
        y_val = y_encoded[val_indices]

        print(f'[+] Datos preparados:')
        print(f'    Train: {X_train.shape}')
        print(f'    Validation: {X_val.shape}')
        print(f'    Clases: {num_classes}')

        return X_train, y_train, X_val, y_val, label_to_idx

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50):
        """
        Entrena el modelo.

        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs: Número de épocas
        """
        print('[*] Entrenando modelo...')

        # Cargar modelo
        model = self.load_existing_model()

        # Callbacks
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = self.output_dir / f'best_model_hybrid_improved_{timestamp}.h5'

        callbacks = [
            ModelCheckpoint(
                str(model_file),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Entrenar
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        print(f'[+] Modelo mejorado guardado: {model_file}')

        return model, model_file, history

    def update_labels(self, label_to_idx: dict):
        """Actualiza el archivo de labels."""
        labels_file = self.output_dir / f'labels_hybrid_improved_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        # Invertir mapeo
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        with open(labels_file, 'w') as f:
            json.dump(idx_to_label, f, indent=2)

        print(f'[+] Labels guardados: {labels_file}')

        return labels_file


def main():
    """Función principal."""
    print("=" * 60)
    print("Entrenamiento de Modelo con Datos Recopilados")
    print("=" * 60)

    # Procesar datos recopilados
    print("\n[PASO 1] Procesando imágenes recopiladas...")
    processor = CollectedDataProcessor()
    X_collected, y_collected = processor.load_collected_data()

    if len(X_collected) == 0:
        print("[!] No hay imágenes recopiladas para entrenar")
        print("[*] Por favor, captura algunas imágenes primero")
        return

    # Preparar datos
    print("\n[PASO 2] Preparando datos para entrenamiento...")
    trainer = HybridDataTrainer()
    X_train, y_train, X_val, y_val, label_to_idx = trainer.prepare_training_data(
        X_collected, y_collected
    )

    # Entrenar
    print("\n[PASO 3] Entrenando modelo mejorado...")
    model, model_file, history = trainer.train(X_train, y_train, X_val, y_val, epochs=50)

    # Guardar labels
    print("\n[PASO 4] Guardando configuración...")
    labels_file = trainer.update_labels(label_to_idx)

    print("\n" + "=" * 60)
    print("[EXITO] Entrenamiento completado!")
    print("=" * 60)
    print(f"\nArchivos generados:")
    print(f"  - Modelo: {model_file}")
    print(f"  - Labels: {labels_file}")
    print(f"\nPróximos pasos:")
    print(f"  1. Copiar el modelo a: C:/models_prod/best_model_hybrid.h5")
    print(f"  2. Copiar labels a: C:/models_prod/labels_hybrid.json")
    print(f"  3. Reiniciar la aplicación Flask")
    print(f"  4. Probar la detección mejorada")
    print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
