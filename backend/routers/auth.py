import json
from fastapi import APIRouter, HTTPException, status,Response
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import LoginUser,JoinUser

#分割したエンドポイントの作成
#ログイン操作
auth_endpoint = APIRouter()

#ログイン処理
@auth_endpoint.post('/auth/login')
def login(login_user:LoginUser):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #バリデーションの実行
    if not(validate_data(login_user, 'login_user')):
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"failed to validate"}))
    
    #SQLの実行
    query_text =f"SELECT id, name, password, email, authority, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM users WHERE (name='{login_user.name}' OR email='{login_user.email}') AND password='{login_user.password}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        login_info = [dict(row) for row in rows][0]
        return Response(status_code=status.HTTP_200_OK,content=json.dumps(login_info))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to login"}))
    
#ログイン処理
@auth_endpoint.post('/auth/join/{project_id}')
def join(project_id:str,join_user:JoinUser):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #バリデーションの実行
    if not(validate_data(join_user, 'join_user')):
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"failed to validate"}))

    try:
        id = int(project_id)
    except Exception as e:
        print(e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"invalid user_id"}))
    
    if(id!=join_user.project_id):
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"invalid project_id"}))
    
    #SQLの実行(プロジェクトのパスワードがあっているかの確認)
    query_text =f"SELECT id,password FROM projects WHERE id='{id}' AND password='{join_user.project_password}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    
    if (result is None):
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to join"}))

    rows = result.mappings().all()
    project_pass_info_list = [dict(row) for row in rows]
    
    if(len(project_pass_info_list)==0):
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"incorrect password"}))
    
    project_pass_info = project_pass_info_list[0]
    print(project_pass_info)
    
    #SQLの実行
    query_text =f"INSERT INTO project_memberships(user_id, project_id) VALUES ('{join_user.user_id}','{project_pass_info['id']}');"
    
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_200_OK,content=json.dumps({'project_membership':f'{join_user.user_id}-{project_pass_info['id']}'}))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to create"}))