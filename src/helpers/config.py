from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List



class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str
 


    MONGODB_URL: str
    MONGODB_DATABASE: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int


    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None

    FACE_EMBEDDING_BACKEND: str
    EMBEDDING_SIZE: int
    EMBEDDING_BATCH_SIZE: str
    FACENET_MODEL_PATH: str







    FACE_DETECTION_BACKEND: str
    DLIB_DETECTOR_MODEL_PATH: str

    MTCNN_MIN_FACE_SIZE: int








    class Config:
        env_file = ".env"

def get_settings():
    return Settings()