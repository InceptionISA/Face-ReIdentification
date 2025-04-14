from fastapi import APIRouter, File, UploadFile, status, Request 
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from models import ResponseMessage 

from models.ProjectModel import ProjectModel 
from models.PersonModel import PersonModel
from controllers.FaceRecognitionController import FaceRecognitionController, FaceEmbeddingService
import logging



logger = logging.getLogger('uvicorn.error')

face_router = APIRouter(
    prefix="/api/v1/faces",
    tags=["faces", 'api_v1'],
    responses={404: {"description": "Not found"}}
)




@face_router.post("/{project_id}/persons/{person_id}/generate-embeddings")
async def generate_person_embeddings(
    request: Request,
    project_id: str,
    person_id: str,
    
):
    try:
        db = request.app.database

        project_model = await ProjectModel.create_instance(db)
        project = await project_model.get_project(project_id)

        if not project:
            return JSONResponse(
                {"message": ResponseMessage.PROJECTNOTFOUND.value},
                status_code=404
            )
        

        person_model = await PersonModel.create_instance(db)
        person = await person_model.get_person(project.id, person_id)
        
        if not person:
            return JSONResponse(
                {"message": ResponseMessage.PERSONNOTEXIST.value},
                status_code=404
            )


        # controller
        embedding_service = FaceEmbeddingService(request.app.face_detector, request.app.feature_extractor)
        face_controller = FaceRecognitionController(embedding_service, request.app.vectordb_client)
                

                


        result = await face_controller.process_person(project_id, person)

        if not result:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "message": ResponseMessage.FACEPROCESSINGFAILED.value,
                    "person_id": person_id
                }
            )

        # Update person status in DB
        await person_model.update_embedding_status(
            project_id=project.id,
            person_id=person_id,
            has_embeddings=True
        )


        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "person_id": person_id,
                "project_id": project_id,
                "collection": face_controller.vector_db.create_collection_name(project_id),
                "message": "Face processing completed"
            }
        )
    
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": ResponseMessage.INTERNALSERVERERROR.value}
        )









# Batch Processing
@face_router.post("/{project_id}/generate-embeddings-batch")
async def generate_project_embeddings(request: Request, project_id: str):

    try:
        print(project_id)
        db = request.app.database

        project = await ProjectModel.create_instance(db)
        project = await project.get_project(project_id)

        if not project:
            return JSONResponse(
                {"message": ResponseMessage.PROJECTNOTFOUND.value},
                status_code=404
            )
        
        # controller
        embedding_service = FaceEmbeddingService(request.app.face_detector, request.app.feature_extractor)
        face_controller = FaceRecognitionController(embedding_service, request.app.vectordb_client)


        # Process using controller
        person_model = await PersonModel.create_instance(db)
        persons, _ = await person_model.get_all_persons(project.id)




        processing_results = await face_controller.batch_process(project_id=project_id,persons=persons)


        print(processing_results)

        
        for person in persons:
            await person_model.update_embedding_status(
            project_id=project.id,
            person_id=person.person_id,
            has_embeddings=True
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Face processing completed for all persons",
                "project_id": project_id,
                "collection": face_controller.vector_db.create_collection_name(project_id)
            }
        )
    
    except Exception as e:
        logger.error(f"Face processing failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": ResponseMessage.INTERNALSERVERERROR.value,
                "error": str(e)
            }
        )

    
    # get all persons in the project
    # for each person get the images paths
    # for each image get the embedding using the face_net_512_model
    # avg the embeddings of all the images to get the final embedding
    # save the embedding in the qdrant db with the person_id as the id
    # return the message that the process is done successfully

    # pass



@face_router.post("/{project_id}/search-similar")
async def search_similar_faces(
    request: Request, 
    project_id: str,  
    file: UploadFile = File(...), 
    limit: int = 1
):

    db = request.app.database
    project_model = await ProjectModel.create_instance(db)
    project = await project_model.get_project(project_id=project_id)

    if not project:
        return JSONResponse(
            {"message": ResponseMessage.PROJECTNOTFOUND.value},
            status_code=404
        )

    # controller
    embedding_service = FaceEmbeddingService(request.app.face_detector, request.app.feature_extractor)
    face_controller = FaceRecognitionController(embedding_service, request.app.vectordb_client)

    # search
    results = await face_controller.search_embedding(project_id, file, limit=limit)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Face search completed",
            "results": results
        }
    )







# get the vectordb info
@face_router.get("/{project_id}/vector-db-info")
async def get_vectordb_info(request: Request, project_id:str):
    
    project_model =await ProjectModel.create_instance(db_client=request.app.database)

    project = await project_model.get_project(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "message": ResponseMessage.PROJECTNOTFOUND.value
            }
        )


    face_controller = FaceRecognitionController(embedding_service=None, vector_db=request.app.vectordb_client)

    collection_info = face_controller.get_vector_db_collection_info(project_id=project_id)

    if not collection_info:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "message": ResponseMessage.VECTORDBCOLLECTIONNOTEXIST.value,
                "collection_info": None
            }
        )

    return JSONResponse(status_code=status.HTTP_200_OK, 
                        content={
                                "message": ResponseMessage.VECTORDB_COLLECTION_RETRIEVED.value ,
                                "collection_info": collection_info
                                })
    

