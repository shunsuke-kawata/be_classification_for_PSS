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
auth_endpoint = APIRouter()

#ログイン処理
@auth_endpoint.post('/auth/login',tags=["auth"],description="ログイン処理を行う",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def login(login_user:LoginUser):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #バリデーションの実行
    if not(validate_data(login_user, 'login_user')):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})
    
    #SQLの実行
    query_text =f"SELECT id, name, password, email, authority FROM users WHERE (name='{login_user.name}' OR email='{login_user.email}') AND password='{login_user.password}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if result is  None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to login", "data":None})
    
    rows = result.mappings().all()
    login_info_list = [dict(row) for row in rows]
    if(len(login_info_list)!=1):
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to login", "data":None})
    login_user_info = login_info_list[0]
    return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to login", "data":login_user_info})
    
#プロジェクトログイン処理
@auth_endpoint.post('/auth/join/{project_id}',tags=["auth"],description="プロジェクト追加処理を行う",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def join(project_id:str,join_user:JoinUser):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #バリデーションの実行
    if not(validate_data(join_user, 'join_user')):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})

    try:
        id = int(project_id)
    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})
    
    if(id!=join_user.project_id):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "invalid project_id", "data":None})
    
    #SQLの実行(プロジェクトのパスワードがあっているかの確認)
    query_text =f"SELECT id,password FROM projects WHERE id='{id}' AND password='{join_user.project_password}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    
    if (result is None):
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to join", "data":None})

    rows = result.mappings().all()
    project_pass_info_list = [dict(row) for row in rows]
    
    if(len(project_pass_info_list)==0):
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to join", "data":None})
    
    project_pass_info = project_pass_info_list[0]
    
    #SQLの実行
    query_text =f"INSERT INTO project_memberships(user_id, project_id) VALUES ('{join_user.user_id}','{project_pass_info['id']}');"
    
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to join", "data":None})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to join", "data":None})