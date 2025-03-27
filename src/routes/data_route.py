from fastapi import APIRouter, Depends, File, UploadFile, status, Request
from fastapi.responses import JSONResponse
import aiofiles
import os
from models import ResponseMessage
from helpers.config import get_settings, Settings
from .schemas.data_schema import ProcessRequest
from models.ProjectModel import ProjectModel 
from models.PersonModel import PersonModel
from controllers.FaceRecognitionController import FaceRecognitionController , FaceEmbeddingService 
from controllers import DataController
# from controllers import ProcessController
import logging


logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["data", 'api_v1'],
    responses={404: {"description": "Not found"}}
)


@data_router.post("/upload/{project_id}/{person_id}")
async def upload_image(
    request: Request,
    project_id: str,
    person_id: str,
    file: UploadFile = File(...),
    app_settings: Settings = Depends(get_settings)
):

    data_base = request.app.database


    project_model = await ProjectModel.create_instance(db_client=data_base)
    project = await project_model.get_or_create_project(project_id=project_id)

    person_model = await PersonModel.create_instance(db_client=data_base)
    person = await person_model.get_or_create_person(project_id=project.id, person_id=person_id)


    if not person:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseMessage.PERSONNOTEXIST.value}
        )
    

    data_controller = DataController()
    is_valid, response_message = data_controller.validate_uploaded_file(file)

    if not is_valid:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": response_message})

    file_path, file_id = data_controller.generate_unique_file_path(
        file_name=file.filename, project_id=project_id, person_id=person_id
    )

    try:
        async with aiofiles.open(file_path, 'wb') as buffer:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await buffer.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseMessage.FILEUPLOADFAILED.value}
        )




    update_result = await person_model.add_image(project_id=project.id, person_id=person_id, image_path=file_id)


    if not update_result:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": ResponseMessage.FILEUPLOADFAILED.value}
        )
    


    return JSONResponse(
        content={
            "message": ResponseMessage.FILEUPLOADSUCCESS.value,
            "file_id": str(file_id)
        }
    )




@data_router.delete("/{project_id}/{person_id}")
async def remove_person(
    request: Request, 
    project_id: str, 
    person_id: str
):
    
    # delete the person from the database
    # delete the person's images from the disk
    # delete the person's embedding from the qdrant db
    # return the message that the process is done successfully

    pass
