from typing import Union
from fastapi import File, UploadFile
from pydantic import BaseModel

#レスポンスのタイプ
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

class UpdateProjectMembershipState(BaseModel):
    user_id: int
    project_id: int
    init_clustering_state: int = None  # 0: 未実行, 1: 実行中, 2: 完了, 3: 失敗
    continuous_clustering_state: int = None  # 0: 実行不可能, 1: 実行中, 2: 実行可能

class NewImage(BaseModel):
    name:str
    project_id:int
    image_file:UploadFile
    
class LoginUser(BaseModel):
    name: str
    email:str
    password: str
    
class JoinUser(BaseModel):
    user_id:int
    project_id:int
    project_password:str

class UserImageClusteringState(BaseModel):
    user_id: int
    image_id: int
    project_id: int
    is_clustered: bool
    
#swaggerにレスポンスを出力する
class CustomResponseModel(BaseModel):
    message: str
    data: Union[dict, None]
    