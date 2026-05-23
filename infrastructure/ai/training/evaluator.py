"""
Model evaluation and reporting utilities.

This module provides comprehensive evaluation, metrics calculation,
and report generation.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
from ..utils.logger import LoggerFactory
from ..utils.metrics import (
    calculate_metrics, plot_confusion_matrix,
    plot_training_history, generate_classification_report
)

logger = LoggerFactory.get_logger(__name__)


class ModelEvaluator:
    """Evaluates model performance."""

    @staticmethod
    def evaluate_full(
        model: tf.keras.Sequential,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list] = None,
        save_dir: str = 'data/metrics'
    ) -> Dict:
        """
        Complete model evaluation with all metrics and visualizations.

        Args:
            model: Trained model
            X_test: Test images
            y_test: Test labels (one-hot)
            class_names: List of class names
            save_dir: Directory to save reports

        Returns:
            dict: Comprehensive evaluation results

        Example:
            >>> results = ModelEvaluator.evaluate_full(model, X_test, y_test, class_names=letters)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting full model evaluation")

        # Get predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_indices = np.argmax(y_test, axis=1)

        # Calculate metrics
        metrics = calculate_metrics(y_test_indices, y_pred)

        logger.info(f"Metrics: {metrics}")

        # Generate classification report
        report = generate_classification_report(
            y_test_indices, y_pred,
            class_names=class_names,
            save_path=str(save_dir / 'classification_report.txt')
        )

        logger.info("Classification report generated")

        # Plot confusion matrix
        plot_confusion_matrix(
            y_test_indices, y_pred,
            classes=class_names,
            save_path=str(save_dir / 'confusion_matrix.png')
        )

        logger.info("Confusion matrix plotted")

        # Prepare results
        results = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'test_samples': len(X_test),
            'confusion_matrix_path': str(save_dir / 'confusion_matrix.png'),
            'report_path': str(save_dir / 'classification_report.txt')
        }

        logger.info(f"Full evaluation completed: Accuracy={results['accuracy']:.4f}")

        return results

    @staticmethod
    def evaluate_per_class(
        model: tf.keras.Sequential,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get per-class evaluation metrics.

        Args:
            model: Trained model
            X_test: Test images
            y_test: Test labels
            class_names: List of class names

        Returns:
            dict: Per-class metrics

        Example:
            >>> per_class = ModelEvaluator.evaluate_per_class(model, X_test, y_test)
        """
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_indices = np.argmax(y_test, axis=1)

        num_classes = y_test.shape[1]
        per_class_metrics = {}

        for class_idx in range(num_classes):
            class_mask = y_test_indices == class_idx

            if class_mask.sum() == 0:
                continue

            class_accuracy = (y_pred[class_mask] == class_idx).mean()

            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"

            per_class_metrics[class_name] = {
                'samples': int(class_mask.sum()),
                'accuracy': float(class_accuracy)
            }

        logger.info(f"Per-class evaluation: {len(per_class_metrics)} classes")

        return per_class_metrics

    @staticmethod
    def plot_training_history(
        history: tf.keras.callbacks.History,
        save_dir: str = 'data/metrics'
    ) -> None:
        """
        Plot and save training history.

        Args:
            history: Training history object
            save_dir: Directory to save plots

        Example:
            >>> ModelEvaluator.plot_training_history(history, 'data/metrics')
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history_dict = history.history

        plot_training_history(history_dict, save_dir=str(save_dir))

        logger.info(f"Training history plots saved to {save_dir}")

    @staticmethod
    def get_worst_predictions(
        model: tf.keras.Sequential,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_examples: int = 10
    ) -> list:
        """
        Get worst predictions (lowest confidence).

        Args:
            model: Trained model
            X_test: Test images
            y_test: Test labels
            num_examples: Number of examples to return

        Returns:
            list: List of (true_label, predicted_label, confidence) tuples

        Example:
            >>> worst = ModelEvaluator.get_worst_predictions(model, X_test, y_test)
        """
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        confidences = np.max(y_pred_probs, axis=1)
        y_test_indices = np.argmax(y_test, axis=1)

        # Get incorrect predictions
        incorrect = y_pred != y_test_indices
        incorrect_indices = np.where(incorrect)[0]

        if len(incorrect_indices) == 0:
            logger.info("No incorrect predictions found")
            return []

        # Sort by confidence (ascending)
        sorted_indices = incorrect_indices[np.argsort(confidences[incorrect_indices])][:num_examples]

        worst = [
            (y_test_indices[i], y_pred[i], float(confidences[i]))
            for i in sorted_indices
        ]

        logger.info(f"Found {len(worst)} worst predictions")

        return worst

    @staticmethod
    def save_evaluation_summary(
        results: Dict,
        save_path: str = 'data/metrics/evaluation_summary.json'
    ) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            save_path: Path to save JSON file

        Example:
            >>> ModelEvaluator.save_evaluation_summary(results)
        """
        import json

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation summary saved to {save_path}")
