from .BaseController import BaseController
from models.db_schemas import PersonSchema
from typing import List, Dict, Optional
import numpy as np
import os
import json
from fastapi import Request, UploadFile, File, HTTPException
from models.db_schemas import ProjectSchema 
from deepface import DeepFace


class FaceEmbeddingService(BaseController):
    def __init__(self, face_detector, feature_extractor, config):

        super().__init__()  
        self.face_detector = config.FACE_DETECTION_BACKEND
        self.feature_extractor = config.FACE_EMBEDDING_BACKEND
        self.embedding_size = config.EMBEDDING_SIZE 


    def get_embedding(self, image_path: str) -> Optional[List[float]]:

        embedding = DeepFace.represent(
            img_path=image_path,
            model_name=self.feature_extractor,
            detector_backend=self.face_detector,
            align=True,
            enforce_detection=False
        )
        return embedding




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

    def create_collection_name(self , project_id):
        return f"collection_{project_id}".strip()

    def get_vector_db_collection_info(self, project_id: str):
        collection_name = self.create_collection_name(project_id=project_id)
        collection_info = self.vector_db.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )

    async def process_person(self, project_id: str, person : PersonSchema) -> bool:


        person_id = person.person_id
        person_images = person.images


        # Process using controller
        person_dir = BaseController().get_image_path(str(project_id), person_id)
        person_images = [os.path.join(person_dir, image) for image in person_images]


        # Create collection if not exists
        collection_name = self.create_collection_name(project_id)
        
        # Ensure collection exists
        if not self.vector_db.is_collection_existed(collection_name):
            self.vector_db.create_collection(
                collection_name=collection_name, 
                embedding_size=self.embedding_service.embedding_size,
                do_reset=False  
            )


        # Get average embedding
        

        embedding = self._get_average_embedding(person_images)
        
        if embedding is None:
            print(f"No valid embeddings found for person {person_id}")
            return False
        
        # Insert embedding into vector DB


        if person.has_embedding:
            print(f"Updating embedding for person {person_id}")
            return self.vector_db.update_record(
                collection_name=collection_name, 
                person_id=person_id, 
                vector=embedding
            )
        

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

    async def batch_process(self, project_id: str, persons: List[PersonSchema]) -> Dict[str, bool]:

        results = {}
        for person in persons:
            results[person.person_id] = await self.process_person(project_id=project_id, person=person)
        
        return results


