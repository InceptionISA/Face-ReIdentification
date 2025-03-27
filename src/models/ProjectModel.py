
from .BaseDataModel import BaseDataModel
from .db_schemas import ProjectSchema
from models import DataBaseEnums

class ProjectModel(BaseDataModel):
    def __init__(self,db_client: object):
        super().__init__(db_client= db_client)

        self.collection = self.db_client[DataBaseEnums.COLLECTION_RPOJECT_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client=db_client)
        await instance.init_collection()
        return instance
    


    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnums.COLLECTION_RPOJECT_NAME.value not in all_collections:
            self.collection =  self.db_client[DataBaseEnums.COLLECTION_RPOJECT_NAME.value]
            indexes = ProjectSchema.get_indexing()
            for index in indexes:
                await self.collection.create_index(index["key"], name=index["name"], unique=index["unique"])
           

    async def get_project(self, project_id:str):
        project = await self.collection.find_one({"project_id": project_id})
        if project:
            return ProjectSchema(**project)
        return None

    async def create_project(self, project :ProjectSchema):
        result = await self.collection.insert_one(project.dict(by_alias=True, exclude_unset=True))
        project.id = result.inserted_id
        return project

    # get or create or prject 
    async def get_or_create_project(self, project_id:str):
        project = await self.collection.find_one({"project_id": project_id})
        if project:
            return ProjectSchema(**project)
        else:
            project = ProjectSchema(project_id=project_id)
            return await self.create_project(project=project)
        

    async def get_all_project(self , page:int =1, page_size:int=10):
        total_documents = await self.collection.count_documents({})

        total_pages = total_documents // page_size 
        if total_documents % page_size != 0:
            total_pages += 1

        cursur = await self.collection.find({}).skip((page-1)*page_size).limit(page_size)
        projects =  []

        async for document in await cursur: 
            projects.append(ProjectSchema(**document))

        return projects , total_pages
    

         

 
 

