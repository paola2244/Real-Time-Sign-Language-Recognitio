#!/usr/bin/env python3
"""
Quick runtime check for the sign-language project.

It verifies the dependency versions that usually break on Windows, checks the
MediaPipe hand model asset, loads the trained model/labels, and initializes the
real-time predictor once.
"""

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def ok(message):
    print(f"[OK] {message}")


def fail(message):
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def main():
    print("=" * 60)
    print("Verificacion del entorno")
    print("=" * 60)

    try:
        import numpy as np
        import tensorflow as tf
        import mediapipe as mp
    except Exception as exc:
        fail(f"No se pudieron importar dependencias principales: {exc}")

    ok(f"Python: {sys.version.split()[0]}")
    ok(f"NumPy: {np.__version__}")
    ok(f"TensorFlow: {tf.__version__}")
    ok(f"MediaPipe: {mp.__version__}")

    numpy_version = tuple(int(part) for part in np.__version__.split(".")[:3])
    if numpy_version > (1, 24, 3):
        fail("NumPy debe ser <= 1.24.3 para TensorFlow 2.13. Reinstala requirements.txt.")

    mediapipe_root = Path(mp.__file__).resolve().parent
    hand_graph = mediapipe_root / "modules" / "hand_landmark" / "hand_landmark_tracking_cpu.binarypb"
    if not hand_graph.exists():
        fail(f"MediaPipe esta incompleto. Falta: {hand_graph}")
    ok("Archivo interno de MediaPipe encontrado")

    model_path = PROJECT_ROOT / "trained_models" / "best_model_hybrid.h5"
    labels_path = PROJECT_ROOT / "trained_models" / "labels_hybrid.json"

    if not model_path.exists():
        fail(f"No existe el modelo: {model_path}")
    if not labels_path.exists():
        fail(f"No existe el archivo de labels: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)
    ok(f"Labels cargados: {len(labels)} clases")

    from infrastructure.ai.realtime.realtime_predictor import RealtimePredictor

    predictor = RealtimePredictor(
        model_path=str(model_path),
        labels_path=str(labels_path),
        confidence_threshold=0.75,
    )
    try:
        model_output = int(predictor.sign_language_agent.inference_agent.model.output_shape[-1])
        if model_output != len(labels):
            fail(
                f"El modelo tiene {model_output} salidas, pero labels_hybrid.json tiene {len(labels)} labels."
            )
        ok("Modelo y labels coinciden")
        ok("Predictor inicializado correctamente")
    finally:
        predictor.close()

    print("=" * 60)
    print("[EXITO] El entorno esta listo para ejecutar la app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
