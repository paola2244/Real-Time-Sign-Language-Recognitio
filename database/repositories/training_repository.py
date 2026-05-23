"""Repository for training log data access."""

from typing import List, Optional
from ..connection import get_db
from ..models.training_log_model import TrainingLog


class TrainingRepository:
    """Data access layer for training logs."""

    COLLECTION_NAME = 'training_logs'

    @staticmethod
    def create(log: TrainingLog) -> str:
        """Create training log record."""
        db = get_db()
        return db.insert_one(TrainingRepository.COLLECTION_NAME, log.to_dict())

    @staticmethod
    def create_many(logs: List[TrainingLog]) -> List[str]:
        """Create multiple training logs."""
        db = get_db()
        return db.insert_many(
            TrainingRepository.COLLECTION_NAME,
            [log.to_dict() for log in logs]
        )

    @staticmethod
    def get_all() -> List[TrainingLog]:
        """Get all training logs."""
        db = get_db()
        data = db.find(TrainingRepository.COLLECTION_NAME)
        return [TrainingLog.from_dict(d) for d in data]

    @staticmethod
    def get_by_epoch(epoch: int) -> Optional[TrainingLog]:
        """Get training log by epoch."""
        db = get_db()
        data = db.find_one(
            TrainingRepository.COLLECTION_NAME,
            {'epoch': epoch}
        )
        return TrainingLog.from_dict(data) if data else None

    @staticmethod
    def get_last_epoch() -> Optional[TrainingLog]:
        """Get most recent training log."""
        db = get_db()
        logs = db.find(TrainingRepository.COLLECTION_NAME)
        return TrainingLog.from_dict(logs[-1]) if logs else None

    @staticmethod
    def delete_all() -> int:
        """Delete all training logs."""
        db = get_db()
        return db.delete_many(TrainingRepository.COLLECTION_NAME, {})

    @staticmethod
    def get_best_accuracy() -> Optional[float]:
        """Get best validation accuracy from logs."""
        logs = TrainingRepository.get_all()
        if not logs:
            return None
        return max(log.val_accuracy for log in logs)
