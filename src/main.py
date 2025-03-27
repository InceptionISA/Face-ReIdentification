from fastapi import FastAPI 
from routes import base_route , data_route , face_route
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.vectordb import VectorDBProviderFactory
from stores.facedetection import FaceDetectorFactory
from stores.featureextraction import FeatureExtractorFactory

# Initialize FastAPI app
app = FastAPI()


async def initialize_resources():
    config = get_settings()

    app.mongo_client = AsyncIOMotorClient(config.MONGODB_URL)
    app.database = app.mongo_client[config.MONGODB_DATABASE]


    face_detector_factory = FaceDetectorFactory(config=config)
    face_feature_extractor_factory = FeatureExtractorFactory(config=config)

    app.face_detector = face_detector_factory.create(provider=config.FACE_DETECTION_BACKEND)
    app.feature_extractor = face_feature_extractor_factory.create(provider=config.FACE_EMBEDDING_BACKEND)



    vectordb_provider_factory = VectorDBProviderFactory(config=config)
    # Initialize VectorDB providers
    app.vectordb_client = vectordb_provider_factory.create(provider=config.VECTOR_DB_BACKEND)
    app.vectordb_client.connect()





async def close_resources():
    app.mongo_client.close()
    app.vectordb_client.disconnect()



app.add_event_handler("startup", initialize_resources)
app.add_event_handler("shutdown", close_resources)

# Include routers for API endpoints

app.include_router(base_route.base_router)
app.include_router(data_route.data_router)
app.include_router(face_route.face_router)
