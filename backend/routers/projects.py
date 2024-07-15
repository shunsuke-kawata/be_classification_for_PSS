import json
from fastapi import APIRouter, status
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewProject

#分割したエンドポイントの作成
projects_endpoint = APIRouter()

#プロジェクト一覧の取得
@projects_endpoint.get('/projects',tags=["projects"],description="プロジェクト一覧の取得",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_projects(user_id=None):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})

    #SQLの実行
    if(user_id is None):
        query_text =f"SELECT id, name, description,images_folder_path,object_images_folder_path,owner_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM projects;"
    else:
        query_text = f"SELECT projects.id, projects.name, projects.description, projects.images_folder_path,projects.object_images_folder_path,projects.owner_id,DATE_FORMAT(projects.created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(projects.updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at,CASE WHEN project_memberships.user_id IS NOT NULL THEN true ELSE false END as joined FROM projects LEFT JOIN project_memberships ON projects.id = project_memberships.project_id AND project_memberships.user_id = {user_id};"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        project_list = [dict(row) for row in rows]
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeed to read projects", "data":project_list})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to read projects", "data":None})

#単一のプロジェクトを取得
@projects_endpoint.get('/projects/{project_id}',tags=["projects"],description="単一プロジェクトの情報を取得",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_project(project_id:str):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    try:
        id = int(project_id)
    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})

    #SQLの実行
    query_text =f"SELECT id, name, password, description, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM projects WHERE id='{id}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        project_info = [dict(row) for row in rows][0]
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to read project", "data":project_info})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to read projects", "data":None})
    
#プロジェクトの作成
@projects_endpoint.post('/projects',tags=["projects"],description="新規プロジェクトの作成",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def create_project(project:NewProject):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #バリデーションの実行
    if not(validate_data(project, 'project')):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})
    
    images_folder_path = ""
    object_images_folder_path = ""
    
    #SQLの実行
    query_text =f"INSERT INTO projects(name, password, description,images_folder_path,object_images_folder_path,owner_id) VALUES ('{project.name}', '{project.password}','{project.description}','{images_folder_path}','{object_images_folder_path}','{project.owner_id}');"
    result,new_project_id = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return JSONResponse(status_code=status.HTTP_201_CREATED,content={"message": "succeeded to create project", "data":{'project_id':new_project_id}})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to create project", "data":None})
    
#プロジェクトの削除
@projects_endpoint.delete('/projects/{project_id}',tags=["projects"],description="プロジェクトの削除",responses={
    204: {"description": "No Content", "model": None},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def delete_project(project_id:str):
    connect_session = create_connect_session()
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    try:
        id = int(project_id)
    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "invalid project_id", "data":None})
    
    #SQLの実行
    query_text =f"DELETE FROM projects WHERE id='{id}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to delete project", "data":None})