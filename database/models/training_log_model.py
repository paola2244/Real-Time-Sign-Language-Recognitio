"""Training log data model."""

from datetime import datetime
from typing import Optional


class TrainingLog:
    """Data model for training logs."""

    def __init__(
        self,
        epoch: int,
        accuracy: float,
        val_accuracy: float,
        loss: float,
        val_loss: float,
        _id: Optional[str] = None
    ):
        """Initialize training log."""
        self._id = _id
        self.epoch = epoch
        self.accuracy = accuracy
        self.val_accuracy = val_accuracy
        self.loss = loss
        self.val_loss = val_loss
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            'epoch': self.epoch,
            'accuracy': self.accuracy,
            'val_accuracy': self.val_accuracy,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'timestamp': self.timestamp
        }
        if self._id:
            data['_id'] = self._id
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            epoch=data['epoch'],
            accuracy=data['accuracy'],
            val_accuracy=data['val_accuracy'],
            loss=data['loss'],
            val_loss=data['val_loss'],
            _id=data.get('_id')
        )
