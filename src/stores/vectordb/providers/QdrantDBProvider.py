from typing import List, Dict, Optional
import logging

from qdrant_client import QdrantClient, models
from qdrant_client.http import exceptions as qdrant_exceptions

from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
from models.db_schemas import RetrievedFace

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):
        self.db_path = db_path
        self.client: Optional[QdrantClient] = None
        self.distance_method = self._get_distance_method(distance_method)
        self.logger = logging.getLogger(self.__class__.__name__)



    def _get_distance_method(self, distance_method: str) -> models.Distance:

        distance_map = {
            DistanceMethodEnums.COSINE.value: models.Distance.COSINE,
            DistanceMethodEnums.DOT.value: models.Distance.DOT
        }
        return distance_map.get(distance_method, models.Distance.COSINE)

    def connect(self) -> None:
        try:
            self.client = QdrantClient(path=self.db_path)
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant database: {e}")
            raise

    def disconnect(self) -> None:
        self.client = None

    def create_collection_name(self, project_id: str) -> str:

        return f"collection_{project_id}".strip()
    

    def is_collection_existed(self, collection_name: str) -> bool:

        if not self.client:
            raise ValueError("Database client not initialized. Call connect() first.")
        
        return self.client.collection_exists(collection_name=collection_name)

    def _validate_collection(self, collection_name: str) -> bool:

        if not self.client:
            raise ValueError("Database client not initialized. Call connect() first.")
        
        return self.client.collection_exists(collection_name=collection_name)

    def create_collection(
        self, 
        collection_name: str, 
        embedding_size: int, 
        do_reset: bool = False
    ) -> bool:

        if not self.client:
            raise ValueError("Database client not initialized. Call connect() first.")

        try:
            if do_reset and self._validate_collection(collection_name):
                self.client.delete_collection(collection_name=collection_name)

            if not self._validate_collection(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_method
                    )
                )
                return True
            return False

        except qdrant_exceptions.QdrantException as e:
            self.logger.error(f"Error creating collection {collection_name}: {e}")
            return False

    def insert_record(
        self, 
        collection_name: str, 
        person_id: str, 
        vector: List[float]
    ) -> bool:
        """
        Insert a single record into the collection.

        :param collection_name: Name of the collection
        :param person_id: Unique identifier for the person
        :param vector: Embedding vector
        :return: True if insertion successful, False otherwise
        """
        if not self._validate_collection(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist")
            return False

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=person_id,
                        vector=vector,
                        payload={"person_id": person_id}
                    )
                ]
            )
            return True
        except qdrant_exceptions.QdrantException as e:
            self.logger.error(f"Error inserting record: {e}")
            return False

    def insert_many(
        self, 
        collection_name: str, 
        persons: List[Dict[str, List[float]]], 
        batch_size: int = 50
    ) -> bool:

        if not self._validate_collection(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist")
            return False

        try:
            for i in range(0, len(persons), batch_size):
                batch = persons[i:i + batch_size]
                batch_points = [
                    models.PointStruct(
                        id=person["person_id"],
                        vector=person["vector"],
                        payload={"person_id": person["person_id"]}
                    )
                    for person in batch
                ]
                self.client.upsert(collection_name=collection_name, points=batch_points)
            return True
        except qdrant_exceptions.QdrantException as e:
            self.logger.error(f"Error inserting batch: {e}")
            return False

    def search(
        self, 
        collection_name: str, 
        vector: List[float], 
        limit: int = 1
    ) -> Optional[List[RetrievedFace]]:
        """
        Search for similar vectors in the collection.

        :param collection_name: Name of the collection
        :param vector: Query vector
        :param limit: Maximum number of results to return
        :return: List of retrieved faces or None if no results
        """
        if not self._validate_collection(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist")
            return None

        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit
            )

            return [
                RetrievedFace(
                    person_id=result.payload.get("person_id"),
                    similarity=result.score
                )
                for result in results
            ] or None

        except qdrant_exceptions.QdrantException as e:
            self.logger.error(f"Search error: {e}")
            return None

    def delete_record(self, collection_name: str, person_id: str) -> bool:

        if not self._validate_collection(collection_name):
            self.logger.error(f"Collection {collection_name} does not exist")
            return False

        try:
            self.client.delete(
                collection_name=collection_name,
                points=[person_id]
            )
            return True
        except qdrant_exceptions.QdrantException as e:
            self.logger.error(f"Error deleting record: {e}")
            return False