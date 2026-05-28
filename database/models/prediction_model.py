"""Prediction data model."""

from datetime import datetime
from typing import Optional


class Prediction:
    """Data model for sign language predictions."""

    def __init__(
        self,
        letter: str,
        confidence: float,
        word_context: str,
        session_id: str,
        timestamp: Optional[str] = None,
        _id: Optional[str] = None
    ):
        """Initialize prediction."""
        self._id = _id
        self.letter = letter
        self.confidence = confidence
        self.word_context = word_context
        self.timestamp = timestamp or datetime.now().isoformat()
        self.session_id = session_id

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            'letter': self.letter,
            'confidence': self.confidence,
            'word_context': self.word_context,
            'timestamp': self.timestamp,
            'session_id': self.session_id
        }
        if self._id:
            data['_id'] = self._id
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            letter=data['letter'],
            confidence=data['confidence'],
            word_context=data.get('word_context', ''),
            session_id=data['session_id'],
            timestamp=data.get('timestamp'),
            _id=data.get('_id')
        )
