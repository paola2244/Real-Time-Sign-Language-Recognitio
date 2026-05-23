"""User data model."""

from datetime import datetime
from typing import Optional


class User:
    """Data model for application users."""

    def __init__(
        self,
        session_id: str,
        total_predictions: int = 0,
        _id: Optional[str] = None
    ):
        """Initialize user."""
        self._id = _id
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.total_predictions = total_predictions

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'total_predictions': self.total_predictions
        }
        if self._id:
            data['_id'] = self._id
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            session_id=data['session_id'],
            total_predictions=data.get('total_predictions', 0),
            _id=data.get('_id')
        )
