"""
Database connection management with MongoDB and JSON fallback.

This module provides a singleton database connection that automatically
falls back to JSON storage if MongoDB is unavailable.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class Database:
    """
    Database connection handler with MongoDB support and JSON fallback.

    Implements singleton pattern to ensure single database instance.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize database connection."""
        if self._initialized:
            return

        self.mongodb_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = os.getenv('MONGO_DB_NAME', 'sign_language_db')
        self.use_mongodb = False
        self.client = None
        self.db = None

        # Try to connect to MongoDB
        if MONGODB_AVAILABLE:
            self._try_mongodb_connection()
        else:
            print("MongoDB client not available. Using JSON storage.")

        # Setup JSON fallback
        self.json_dir = Path('data/json_db')
        self.json_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True

    def _try_mongodb_connection(self) -> bool:
        """
        Attempt connection to MongoDB.

        Returns:
            bool: True if connection successful
        """
        try:
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=2000)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.use_mongodb = True
            print(f"[OK] Connected to MongoDB: {self.db_name}")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError):
            print("[WARN] MongoDB connection failed. Using JSON storage.")
            return False
        except Exception as e:
            print(f"[WARN] MongoDB error: {e}. Using JSON storage.")
            return False

    def get_collection(self, collection_name: str):
        """
        Get collection from database.

        Args:
            collection_name: Name of collection

        Returns:
            MongoDB collection or JSONCollection wrapper
        """
        if self.use_mongodb:
            return self.db[collection_name]
        else:
            return JSONCollection(self.json_dir, collection_name)

    def insert_one(self, collection_name: str, document: Dict) -> str:
        """
        Insert single document.

        Args:
            collection_name: Collection name
            document: Document to insert

        Returns:
            str: Document ID
        """
        collection = self.get_collection(collection_name)

        if self.use_mongodb:
            result = collection.insert_one(document)
            return str(result.inserted_id)
        else:
            return collection.insert_one(document)

    def insert_many(self, collection_name: str, documents: List[Dict]) -> List[str]:
        """
        Insert multiple documents.

        Args:
            collection_name: Collection name
            documents: Documents to insert

        Returns:
            List[str]: Document IDs
        """
        collection = self.get_collection(collection_name)

        if self.use_mongodb:
            result = collection.insert_many(documents)
            return [str(id) for id in result.inserted_ids]
        else:
            return collection.insert_many(documents)

    def find(self, collection_name: str, query: Dict = None, limit: int = None):
        """
        Find documents.

        Args:
            collection_name: Collection name
            query: Query filter
            limit: Result limit

        Returns:
            Cursor or list of documents
        """
        if query is None:
            query = {}

        collection = self.get_collection(collection_name)

        if self.use_mongodb:
            cursor = collection.find(query)
            if limit:
                cursor = cursor.limit(limit)
            return list(cursor)
        else:
            return collection.find(query, limit)

    def find_one(self, collection_name: str, query: Dict = None):
        """
        Find single document.

        Args:
            collection_name: Collection name
            query: Query filter

        Returns:
            Document or None
        """
        if query is None:
            query = {}

        collection = self.get_collection(collection_name)

        if self.use_mongodb:
            return collection.find_one(query)
        else:
            return collection.find_one(query)

    def update_one(self, collection_name: str, query: Dict, update: Dict) -> int:
        """
        Update single document.

        Args:
            collection_name: Collection name
            query: Query filter
            update: Update operations

        Returns:
            int: Number of documents modified
        """
        collection = self.get_collection(collection_name)

        if self.use_mongodb:
            result = collection.update_one(query, {'$set': update})
            return result.modified_count
        else:
            return collection.update_one(query, update)

    def delete_one(self, collection_name: str, query: Dict) -> int:
        """
        Delete single document.

        Args:
            collection_name: Collection name
            query: Query filter

        Returns:
            int: Number of documents deleted
        """
        collection = self.get_collection(collection_name)

        if self.use_mongodb:
            result = collection.delete_one(query)
            return result.deleted_count
        else:
            return collection.delete_one(query)

    def delete_many(self, collection_name: str, query: Dict = None) -> int:
        """
        Delete multiple documents.

        Args:
            collection_name: Collection name
            query: Query filter

        Returns:
            int: Number of documents deleted
        """
        if query is None:
            query = {}

        collection = self.get_collection(collection_name)

        if self.use_mongodb:
            result = collection.delete_many(query)
            return result.deleted_count
        else:
            return collection.delete_many(query)

    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            print("Database connection closed")


class JSONCollection:
    """Fallback JSON storage implementation."""

    def __init__(self, base_dir: Path, collection_name: str):
        """
        Initialize JSON collection.

        Args:
            base_dir: Base directory for JSON files
            collection_name: Collection name
        """
        self.base_dir = base_dir
        self.collection_name = collection_name
        self.file_path = base_dir / f"{collection_name}.json"
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create JSON file if it doesn't exist."""
        if not self.file_path.exists():
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    def _load_data(self) -> List[Dict]:
        """Load data from JSON file."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_data(self, data: List[Dict]):
        """Save data to JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def insert_one(self, document: Dict) -> str:
        """Insert single document."""
        data = self._load_data()
        doc_id = str(len(data) + 1)
        document['_id'] = doc_id
        document['timestamp'] = datetime.now().isoformat()
        data.append(document)
        self._save_data(data)
        return doc_id

    def insert_many(self, documents: List[Dict]) -> List[str]:
        """Insert multiple documents."""
        ids = []
        for doc in documents:
            ids.append(self.insert_one(doc))
        return ids

    def find(self, query: Dict = None, limit: int = None) -> List[Dict]:
        """Find documents."""
        if query is None:
            query = {}

        data = self._load_data()
        results = []

        for doc in data:
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                results.append(doc)
                if limit and len(results) >= limit:
                    break

        return results

    def find_one(self, query: Dict = None) -> Optional[Dict]:
        """Find single document."""
        results = self.find(query, limit=1)
        return results[0] if results else None

    def update_one(self, query: Dict, update: Dict) -> int:
        """Update single document."""
        data = self._load_data()
        for doc in data:
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                doc.update(update)
                self._save_data(data)
                return 1
        return 0

    def delete_one(self, query: Dict) -> int:
        """Delete single document."""
        data = self._load_data()
        for i, doc in enumerate(data):
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if match:
                data.pop(i)
                self._save_data(data)
                return 1
        return 0

    def delete_many(self, query: Dict = None) -> int:
        """Delete multiple documents."""
        if query is None:
            query = {}

        data = self._load_data()
        new_data = []
        deleted = 0

        for doc in data:
            match = True
            for key, value in query.items():
                if key not in doc or doc[key] != value:
                    match = False
                    break
            if not match:
                new_data.append(doc)
            else:
                deleted += 1

        self._save_data(new_data)
        return deleted


def get_db() -> Database:
    """Get singleton database instance."""
    return Database()
