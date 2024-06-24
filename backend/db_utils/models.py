from pydantic import BaseModel

class User(BaseModel):
    name: str
    password: str
    email:str
    authority:int
    
class Project(BaseModel):
    name: str
    password: str
    description:str
    