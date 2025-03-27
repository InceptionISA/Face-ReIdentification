from fastapi import APIRouter, Depends, File, UploadFile, status, Request
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from .schemas.data_schema import ProcessRequest
from models.ProjectModel import ProjectModel 
from models.PersonModel import PersonModel
from controllers.FaceRecognitionController import FaceRecognitionController, FaceEmbeddingService
import logging
from controllers.BaseController import BaseController
import os 

logger = logging.getLogger('uvicorn.error')

face_router = APIRouter(
    prefix="/api/v1/faces",
    tags=["faces", 'api_v1'],
    responses={404: {"description": "Not found"}}
)




@face_router.post("/embeddings/{project_id}/{person_id}")
async def generate_person_embeddings(
    request: Request,
    project_id: str,
    person_id: str,
    app_settings: Settings = Depends(get_settings)

):
    try:
        db = request.app.database

        project = await ProjectModel.create_instance(db)
        project = await project.get_project(project_id)

        if not project:
            return JSONResponse(
                {"error": "Project not found"},
                status_code=404
            )
        

        person_model = await PersonModel.create_instance(db)
        person = await person_model.get_person(project.id, person_id)
        
        if not person:
            return JSONResponse(
                {"error": "Person not found"},
                status_code=404
            )    
        


        # controller
        embedding_service = FaceEmbeddingService(request.app.face_detector, request.app.feature_extractor)
        face_controller = FaceRecognitionController(embedding_service, request.app.vectordb_client)
                

                
        # Process using controller
        person_dir = BaseController().get_image_path(project_id, person_id)
        person_images = [os.path.join(person_dir, image) for image in os.listdir(person_dir)]


        result = await face_controller.process_person(project_id, person_id , person_images)

        

        if not result:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "message": result.get("error", "Processing failed"),
                    "person_id": person_id
                }
            )

        # Update person status in DB
        await person_model.update_embedding_status(
            project_id=project_id,
            person_id=person_id,
            has_embeddings=True
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "person_id": person_id,
                "project_id": project_id,
                "embedding_size": len(result["embedding"]) if result["embedding"] else 0,
                "processed_images": result["processed_images"],
                "collection": face_controller.db.create_collection_name(project_id),
                "message": "Face processing completed"
            }
        )

    except Exception as e:
        logger.error(f"Face processing failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Internal processing error",
                "error": str(e),
                "person_id": person_id
            }
        )




# @face_router.post("/embeddings/{project_id}/{person_id}")
# async def generate_person_embeddings(request: Request, project_id: str, process_request: ProcessRequest):

    # get the person's images paths 
    # for each image get the embedding using the face_net_512_model
    # avg the embeddings of all the images to get the final embedding
    # save the embedding in the qdrant db with the person_id as the id
    # return the person_id and message that the process is done successfully

    # pass



# Batch Processing
@face_router.post("/embeddings/batch/{project_id}")
async def generate_project_embeddings(request: Request, project_id: str, process_request: ProcessRequest):
    
    # get all persons in the project
    # for each person get the images paths
    # for each image get the embedding using the face_net_512_model
    # avg the embeddings of all the images to get the final embedding
    # save the embedding in the qdrant db with the person_id as the id
    # return the message that the process is done successfully

    pass



@face_router.post("/search/{project_id}")
async def search_similar_faces(
    request: Request, 
    project_id: str,  
    file: UploadFile = File(...), 
    limit: int = 1
):
    # get the embedding of the image
    # search the embedding in the qdrant db collection named project_id
    # return the person_id and message that the process is done successfully

    pass




