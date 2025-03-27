from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from models.db_schemas import RetrievedFace

class VectorDBInterface(ABC):
    @abstractmethod
    def connect(self):
        """Establish a connection to the vector database."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close the connection to the vector database."""
        pass

    @abstractmethod
    def _validate_collection(self, collection_name: str) -> bool:
        """Check if a collection exists in the database."""
        
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        """Create a new collection with a specified embedding size."""
        pass

    @abstractmethod
    def insert_record(self, collection_name: str, person_id: str, vector: List[float]) -> bool:
        """Insert a single record into the database."""
        pass

    @abstractmethod
    def insert_many(self, collection_name: str, persons: List[Dict[str, List[float]]], batch_size: int = 50) -> bool:
        """Insert multiple records into the database in batches."""
        pass

    @abstractmethod
    def search(self, collection_name: str, vector: List[float], limit: int = 5) -> Optional[List[RetrievedFace]]:
        """Search for similar vectors in the database."""
        pass

    @abstractmethod
    def delete_record(self, collection_name: str, person_id: str) -> bool:
        """Delete a record from the database."""
        pass
