from pydantic import BaseModel

class User(BaseModel):
    name: str
    email:str
    password: str
    authority:bool
    
class Project(BaseModel):
    name: str
    password: str
    description:str
    owner_id:int
    
class LoginUser(BaseModel):
    name: str
    email:str
    password: str