from pydantic import BaseModel

class NewUser(BaseModel):
    name: str
    email:str
    password: str
    authority:bool
    
class NewProject(BaseModel):
    name: str
    password: str
    description:str
    owner_id:int
class NewProjectMembership(BaseModel):
    project_id:int
    user_id:int
    
class LoginUser(BaseModel):
    name: str
    email:str
    password: str
    
class JoinUser(BaseModel):
    user_id:int
    project_id:int
    project_password:str
    