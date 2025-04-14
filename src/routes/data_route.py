from fastapi import APIRouter, Depends, File, UploadFile, status, Request 
from fastapi.responses import JSONResponse
import aiofiles
import os
from models import ResponseMessage
from helpers.config import get_settings, Settings
from .schemas.data_schema import PersonRequest
from models.ProjectModel import ProjectModel 
from models.PersonModel import PersonModel
from controllers import DataController , BaseController
import logging
import shutil

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["data", 'api_v1'],
    responses={404: {"description": "Not found"}}
)


def get_database(request: Request):
    return request.app.database

@data_router.post("/{project_id}/persons/{person_id}/upload-image")
async def upload_image(
    project_id: str,
    person_id: str,
    person_request: PersonRequest = Depends(PersonRequest.as_form),  
    file: UploadFile = File(...),
    app_settings: Settings = Depends(get_settings),
    db = Depends(get_database),
):

    project_model = await ProjectModel.create_instance(db_client=db)
    project = await project_model.get_or_create_project(project_id=project_id)

    person_model = await PersonModel.create_instance(db_client=db)
    

    person = await person_model.get_or_create_person(project_id=project.id,
                                                      person_id=person_id ,
                                                        name=person_request.name,
                                                          age=person_request.age)


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




    update_result = await person_model.add_image(project_id=project.id, person_id=person_id, image_path=file_id )


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




@data_router.delete("/{project_id}")
async def remove_project(request: Request, project_id: str , db = Depends(get_database)):
    

    project_model = await ProjectModel.create_instance(db)
    project = await project_model.get_project(project_id=project_id)

    if not project:
        return JSONResponse(
            {"message": ResponseMessage.PROJECTNOTFOUND.value},
            status_code=404
        )


    # delete persons 
    person_model = await PersonModel.create_instance(db)
    persons , _ = await person_model.get_all_persons(project_id=project.id)


    for person in persons:
        await person_model.delete_person(project_id=project.id, person_id=person.person_id)



    await project_model.delete_project(project_id=project_id)




    request.app.vectordb_client.delete_collection(f'collection_{project_id}')

    project_dir = BaseController().get_project_path(str(project_id))

    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)



    return JSONResponse(
        {"message": ResponseMessage.PROJECTDELETED.value},
        status_code=200
    )



@data_router.delete("/{project_id}/persons/{person_id}")
async def remove_person(
    request: Request, 
    project_id: str, 
    person_id: str
):
    
    db = request.app.database

    project_model = await ProjectModel.create_instance(db)
    project = await project_model.get_project(project_id=project_id)

    if not project:
        return JSONResponse(
            {"message": ResponseMessage.PROJECTNOTFOUND.value},
            status_code=404
        )

    person_model = await PersonModel.create_instance(db)
    person = await person_model.get_person(project_id=project.id, person_id=person_id)

    if not person:
        return JSONResponse(
            {"message": ResponseMessage.PERSONNOTEXIST.value},
            status_code=404
        )
    await person_model.delete_person(project_id=project.id, person_id=person_id)

    person_dir = BaseController().get_image_path(str(project_id), person_id)

    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)


    request.app.vectordb_client.delete_record( collection_name=f'collection_{project_id}', person_id=person_id)


    return JSONResponse(
        {"message": ResponseMessage.PERSONDELETED.value},
        status_code=200
    )