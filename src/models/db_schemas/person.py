from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional
from bson import ObjectId


# Helper function to handle MongoDB ObjectId
class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)


class PersonSchema(BaseModel):
    id: Optional[ObjectId] = Field(default=None, alias="_id")  
    project_id: ObjectId = Field(..., example="65d2a4f6c8e9f5a3b4d1e2f7")  
    person_id: str = Field(..., min_length=1, example="person_001")
    # qdrant_vector_id: str = Field(..., example="abcde-xyz")
    images: List[str] = Field(default=[], example=["image1.jpg", "image2.jpg"])
    has_embedding: bool = Field(default=False, example=False)

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v):
        """Ensures `project_id` is a valid ObjectId."""
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid project_id ObjectId")
        return str(v)  

    @field_validator("images")
    @classmethod
    def check_images(cls, v):
        if not v:
            raise ValueError("At least one image must be provided")
        return v

    @classmethod
    def get_indexing(cls):
        return [
            {"key": [("project_id", 1)], "name": "project_id_index", "unique": False},
            {"key": [("project_id", 1), ("person_id", 1)], "name": "project_person_index", "unique": True},
        ]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RetrievedFace(BaseModel):
    person_id: str
    similarity: float