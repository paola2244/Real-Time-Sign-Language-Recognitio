"""
Metrics calculation and evaluation utilities.

This module provides functions for calculating model performance metrics
and visualization utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from .logger import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels (can be one-hot or class indices)
        y_pred: Predicted labels or probabilities

    Returns:
        dict: Dictionary with accuracy, precision, recall, f1

    Example:
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    # Convert one-hot to indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true.flatten()

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred.flatten()

    return {
        'accuracy': float(accuracy_score(y_true_indices, y_pred_indices)),
        'precision': float(precision_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0)),
        'f1': float(f1_score(y_true_indices, y_pred_indices, average='weighted', zero_division=0))
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot and optionally save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        save_path: Path to save figure
        figsize: Figure size

    Example:
        >>> plot_confusion_matrix(y_test, predictions, classes=letters)
    """
    # Convert one-hot to indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_training_history(
    history: dict,
    save_dir: str = 'data/metrics',
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training history curves.

    Args:
        history: Training history dictionary with 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        save_dir: Directory to save plots
        figsize: Figure size

    Example:
        >>> plot_training_history(model_history, save_dir='data/metrics')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Train Accuracy')
    axes[0].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(history['loss'], label='Train Loss')
    axes[1].plot(history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    accuracy_path = save_dir / 'accuracy_curve.png'
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history saved to {save_dir}")

    plt.close()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    save_path: str = None
) -> str:
    """
    Generate classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save report

    Returns:
        str: Classification report as string

    Example:
        >>> report = generate_classification_report(y_test, predictions, letters)
        >>> print(report)
    """
    # Convert one-hot to indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Crear lista de etiquetas que incluya todas las clases posibles
    all_labels = list(range(len(class_names))) if class_names else None

    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {save_path}")

    return report
