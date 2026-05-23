"""Repository for prediction data access."""

from typing import List, Optional, Dict
from ..connection import get_db
from ..models.prediction_model import Prediction


class PredictionRepository:
    """Data access layer for predictions."""

    COLLECTION_NAME = 'predictions'

    @staticmethod
    def create(prediction: Prediction) -> str:
        """Create prediction record."""
        db = get_db()
        return db.insert_one(PredictionRepository.COLLECTION_NAME, prediction.to_dict())

    @staticmethod
    def get_all(limit: int = 100) -> List[Prediction]:
        """Get all predictions."""
        db = get_db()
        data = db.find(PredictionRepository.COLLECTION_NAME, limit=limit)
        return [Prediction.from_dict(d) for d in data]

    @staticmethod
    def get_by_session(session_id: str, limit: int = 50) -> List[Prediction]:
        """Get predictions by session."""
        db = get_db()
        data = db.find(
            PredictionRepository.COLLECTION_NAME,
            {'session_id': session_id},
            limit=limit
        )
        return [Prediction.from_dict(d) for d in data]

    @staticmethod
    def get_recent(limit: int = 5) -> List[Prediction]:
        """Get recent predictions."""
        db = get_db()
        data = db.find(PredictionRepository.COLLECTION_NAME, limit=limit)
        return [Prediction.from_dict(d) for d in reversed(data)][-limit:]

    @staticmethod
    def delete_by_session(session_id: str) -> int:
        """Delete predictions by session."""
        db = get_db()
        return db.delete_many(
            PredictionRepository.COLLECTION_NAME,
            {'session_id': session_id}
        )

    @staticmethod
    def clear_all() -> int:
        """Delete all predictions."""
        db = get_db()
        return db.delete_many(PredictionRepository.COLLECTION_NAME, {})

    @staticmethod
    def count(session_id: Optional[str] = None) -> int:
        """Count predictions."""
        db = get_db()
        query = {'session_id': session_id} if session_id else {}
        data = db.find(PredictionRepository.COLLECTION_NAME, query)
        return len(data)
