from pydantic import BaseModel

class User(BaseModel):
    name: str
    password: str
    email:str
    authority:bool
    
class Project(BaseModel):
    name: str
    password: str
    description:str
    owner_id:int
    