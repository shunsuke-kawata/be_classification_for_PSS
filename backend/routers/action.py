import json
from fastapi import APIRouter, HTTPException, status,Response
import sys
import os

from fastapi.responses import JSONResponse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, LoginUser,JoinUser

#分割したエンドポイントの作成
#ログイン操作
action_endpoint = APIRouter()

@action_endpoint.get("/action/clustering/init/{project_id}",tags=["auth"],description="ログイン処理を行う")
def execute_init_clustering(project_id:int = None):
    if project_id is None:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "project_id is required", "data": None})

    try:
        project_id = int(project_id)
    except:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "project_id is invalid", "data": None})
    
    connect_session = create_connect_session()
    query_text = f"""
        SELECT init_clustering_state FROM projects WHERE id = {project_id};
    """
    result,_ = execute_query(session=connect_session,query_text=query_text)
    
    print(result.mappings().first()["init_clustering_state"])
    

    return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "init clustering started", "data": project_id})