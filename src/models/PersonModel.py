from .BaseDataModel import BaseDataModel
from .db_schemas import PersonSchema
from models import DataBaseEnums
from bson import ObjectId

class PersonModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnums.COLLECTION_PERSON_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client=db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        """Initialize collection with required indexes."""
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnums.COLLECTION_PERSON_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnums.COLLECTION_PERSON_NAME.value]
            indexes = PersonSchema.get_indexing()
            for index in indexes:
                await self.collection.create_index(index["key"], name=index["name"], unique=index["unique"])

    

    async def create_person(self, person: PersonSchema):

        result = await self.collection.insert_one(person.dict(by_alias=True, exclude_unset=True))
        person.id = result.inserted_id
        return person
    

    async def get_person(self, project_id: str, person_id: str):
        person = await self.collection.find_one({"project_id": str(project_id), "person_id": person_id})
        if person:
            person['project_id'] = project_id
            return PersonSchema(**person)
        return None

    async def get_or_create_person(self, project_id: str, person_id: str):

        person = await self.collection.find_one({"project_id": str(project_id), "person_id": person_id })
        
        if person:
            person['project_id'] = project_id
            return PersonSchema(**person)
        else:
            person = PersonSchema(project_id=project_id, person_id=person_id , has_embedding=False)
            return await self.create_person(person=person)


    async def get_all_people(self, project_id: str, page: int = 1, page_size: int = 10):
        """Retrieve paginated list of people in a project."""
        total_documents = await self.collection.count_documents({"project_id": project_id})

        total_pages = total_documents // page_size
        if total_documents % page_size != 0:
            total_pages += 1

        cursor = self.collection.find({"project_id": project_id}).skip((page - 1) * page_size).limit(page_size)
        people = [PersonSchema(**doc) async for doc in cursor]

        return people, total_pages

    async def add_image(self, project_id: str, person_id: str, image_path: str):


        """Add an image path to a person's record."""
        return await self.collection.update_one(
            {"project_id": str(project_id), "person_id": person_id},
            {"$push": {"images": image_path}}
        )

    async def update_embedding_status(self, project_id: str, person_id: str, has_embeddings: bool):
        """Update person's embedding status."""


        return await self.collection.update_one(
            {"project_id": str(project_id), "person_id": person_id},
            {"$set": {"has_embedding": has_embeddings}}
        )
    
    async def get_images(self, project_id: str, person_id: str):
        person = await self.collection.find_one({"project_id": str(project_id), "person_id": person_id})
        if person:
            return person["images"]
        return None