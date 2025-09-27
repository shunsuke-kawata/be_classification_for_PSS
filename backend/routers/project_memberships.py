import json
from fastapi import APIRouter, status,Response
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewProjectMembership
from clustering.utils import Utils

#分割したエンドポイントの作成
project_memberships_endpoint = APIRouter()

#ユーザとプロジェクトの紐付けの取得
@project_memberships_endpoint.get('/project_memberships',tags=["project_memberships"],description="ユーザとプロジェクト間の関係一覧を取得",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_project_memberships(user_id=None, project_id=None):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #どちらかでしか検索を受け付けない
    if (project_id is not None and user_id is not None):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"use at least user_id or project_id", "data":None})

    if(project_id is not None):
        try:
            id = int(project_id)
        except Exception as e:
            print(e)
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"invalid project_id", "data":None})
        # SQLの実行
        query_text =f"SELECT user_id, project_id, mongo_result_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships WHERE project_id='{id}';"
    elif(user_id is not None):
        try:
            id = int(user_id)
        except Exception as e:
            print(e)
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"invalid project_id", "data":None})
        # SQLの実行
        query_text =f"SELECT user_id, project_id, mongo_result_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships WHERE user_id='{id}';"
    else:
        # SQLの実行
        query_text =f"SELECT user_id, project_id,  mongo_result_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships;"
        
    result,_ = execute_query(session=connect_session, query_text=query_text)
    
    if result is not None:
        rows = result.mappings().all()
        project_membership_list = [dict(row) for row in rows]
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to read project", "data":project_membership_list})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to read project_memberships", "data":None})

#ユーザとプロジェクトの紐付けを作成
@project_memberships_endpoint.post('/project_memberships',tags=["project_memberships"],description="ユーザとプロジェクト間の紐付けを行う",responses={
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def create_project_membership(project_membership:NewProjectMembership):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #バリデーションの実行
    if not(validate_data(project_membership, 'project_membership')):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})

    mongo_result_id = Utils.generate_uuid()
    print(mongo_result_id)
    
    #SQLの実行
    query_text =f"INSERT INTO project_memberships(user_id, project_id,mongo_result_id) VALUES ('{project_membership.user_id}','{project_membership.project_id}','{mongo_result_id}');"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return JSONResponse(status_code=status.HTTP_201_CREATED,content={"message": "succeeded to create project_membership", "data":None})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to create project_membership", "data":None})