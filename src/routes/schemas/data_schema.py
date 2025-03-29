from pydantic import BaseModel
from fastapi import Form

class PersonRequest(BaseModel):
    name: str
    age: int

    @classmethod
    def as_form(cls, name: str = Form(...), age: int = Form(...)):
        return cls(name=name, age=age)
