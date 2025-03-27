from pydantic import BaseModel ,Field,  field_validator , ConfigDict
from typing import List, Optional
from bson.objectid import ObjectId

class ProjectSchema(BaseModel):
    id :  Optional[ObjectId] = Field(None, alias='_id')

    project_id : str = Field(...,  min_length=1)



    @field_validator('project_id')
    def project_id_validator(cls, v):
        if not v.isalnum():
            raise ValueError('project_id must be alphanumeric')
        return v
    
    @classmethod
    def get_indexing(cls):

        return [
            {"key": [("project_id", 1)], "name": "project_id_index", "unique": True}
        ]
         

    model_config = ConfigDict(arbitrary_types_allowed=True)


