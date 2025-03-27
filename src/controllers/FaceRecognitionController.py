from .BaseController import BaseController
from models.db_schemas import PersonSchema
from typing import List, Dict, Optional
import numpy as np
import os
import json
from fastapi import Request, UploadFile, File, HTTPException




class FaceEmbeddingService(BaseController):
    def __init__(self, face_detector, feature_extractor):

        super().__init__()  
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor

    def detect_faces(self, image_path: str) -> List[np.ndarray]:
        pass

    def get_embedding(self, image_path: str) -> Optional[List[float]]:
        pass    


    def compare_embeddings(self, embedding1: List[float], embedding2: List[float]) -> float:
        pass




class FaceRecognitionController(BaseController):
    def __init__(self, embedding_service, vector_db):

        self.embedding_service = embedding_service
        self.vector_db = vector_db

    def _get_average_embedding(self, image_paths: List[str]) -> Optional[List[float]]:

        embeddings = []
        for image_path in image_paths:
            try:
                embedding = self.embedding_service.get_embedding(image_path)
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        if not embeddings:
            return None
        
        return list(np.mean(embeddings, axis=0))




    async def process_person(self, project_id: str, person_id: str, image_paths: List[str]) -> bool:


        # Create collection if not exists
        collection_name = f"collection_{project_id}"
        
        
        # Ensure collection exists
        if not self.vector_db.is_collection_existed(collection_name):
            self.vector_db.create_collection(
                collection_name=collection_name, 
                embedding_size=512  
            )


        # Get average embedding

        embedding = self._get_average_embedding(image_paths)

        
        if embedding is None:
            print(f"No valid embeddings found for person {person_id}")
            return False
        
        # Insert embedding into vector DB
        return self.vector_db.insert_record(
            collection_name=collection_name, 
            person_id=person_id, 
            vector=embedding
        )

    async def search_embedding(self, project_id: str, image_file: UploadFile, limit: int = 1) -> List[Dict[str, float]]:

        # Save uploaded file temporarily
        temp_path = f"/tmp/{image_file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await image_file.read())
        
        # Get embedding for search image
        embedding = self.embedding_service.get_embedding(temp_path)
        
        # Remove temporary file
        os.unlink(temp_path)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not generate embedding from image")
        
        # Perform search in vector database
        collection_name = f"collection_{project_id}"
        self.vector_db.connect()
        
        results = self.vector_db.search(
            collection_name=collection_name, 
            vector=embedding, 
            limit=limit
        )
        
        if results is None:
            return []
        
        return [
            {
                "person_id": result.person_id, 
                "similarity": result.similarity
            } 
            for result in results
        ]

    async def batch_process(self, project_id: str, persons: List[Dict[str, List[str]]]) -> Dict[str, bool]:

        results = {}
        for person_data in persons:
            for person_id, image_paths in person_data.items():
                try:
                    result = await self.process_person(project_id, person_id, image_paths)
                    results[person_id] = result
                except Exception as e:
                    print(f"Error processing {person_id}: {e}")
                    results[person_id] = False
        
        return results

    async def delete_person(self, project_id: str, person_id: str) -> bool:


        collection_name = f"collection_{project_id}"
        self.vector_db.connect()
        
        return self.vector_db.delete_record(
            collection_name=collection_name, 
            person_id=person_id
        )