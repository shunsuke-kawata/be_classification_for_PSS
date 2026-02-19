from datetime import datetime
import json
from fastapi import APIRouter, status
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session
from db_utils.projects_queries import (
    get_projects,
    get_projects_for_user,
    get_project,
    get_project_for_user,
    insert_project,
    delete_project,
)
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewProject
from pathlib import Path
from config import DEFAULT_IMAGE_PATH
from clustering.utils import Utils

projects_endpoint = APIRouter()

# プロジェクト一覧の取得
@projects_endpoint.get('/projects', tags=["projects"], description="プロジェクト一覧の取得", responses={
    200: {"description": "OK", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_projects(user_id=None):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})

    if user_id is None:
        result, _ = get_projects(session=connect_session)
    else:
        result, _ = get_projects_for_user(session=connect_session, user_id=user_id)
    if result is not None:
        rows = result.mappings().all()
        res_data = []
        for row in rows:
            row_dict = dict(row)
            # すべての datetime を文字列に変換
            for k, v in row_dict.items():
                if isinstance(v, datetime):
                    row_dict[k] = v.isoformat()
            res_data.append(row_dict)
        print(res_data)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "succeeded to read projects", "data": res_data})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to read projects", "data": None})

# 単一のプロジェクトを取得
@projects_endpoint.get('/projects/{project_id}', tags=["projects"], description="単一プロジェクトの情報を取得", responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_project(project_id: str,user_id=None):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})
    
    try:
        id = int(project_id)
    except Exception:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "invalid project_id", "data": None})

    if user_id is None:
        result, _ = get_project(session=connect_session, project_id=id)
    else:
        result, _ = get_project_for_user(session=connect_session, project_id=id, user_id=user_id)
    if result is not None:
        rows = result.mappings().first()  # 最初の行だけ取得
        
        if rows is None:
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": "project not found", "data": None})
                
        row_dict = dict(rows)
        # すべての datetime を文字列に変換
        for k, v in row_dict.items():
            if isinstance(v, datetime):
                row_dict[k] = v.isoformat()

        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to read projects", "data": row_dict})
    else:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,content={"message": "failedß to read projects", "data": None})

# プロジェクトの作成
@projects_endpoint.post('/projects', tags=["projects"], description="新規プロジェクトの作成", responses={
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def create_project(project: NewProject):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})

    if not validate_data(project, 'project'):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "failed to validate", "data": None})

    original_images_folder_path = Utils.generate_uuid()
    
    os.makedirs(Path(DEFAULT_IMAGE_PATH) / original_images_folder_path, exist_ok=True)
    if not os.path.exists(Path(DEFAULT_IMAGE_PATH) / original_images_folder_path):
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to create original images folder", "data": None})
    
    result, new_project_id = insert_project(session=connect_session, name=project.name, password=project.password, description=project.description, original_images_folder_path=original_images_folder_path, owner_id=project.owner_id)
    if result:
        return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message": "succeeded to create project", "data": {"project_id": new_project_id}})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to create project", "data": None})

# プロジェクトの削除
@projects_endpoint.delete('/projects/{project_id}', tags=["projects"], description="プロジェクトの削除", responses={
    204: {"description": "No Content"},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def delete_project(project_id: str):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})

    try:
        id = int(project_id)
    except Exception:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "invalid project_id", "data": None})

    result, _ = delete_project(session=connect_session, project_id=id)
    if result:
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to delete project", "data": None})