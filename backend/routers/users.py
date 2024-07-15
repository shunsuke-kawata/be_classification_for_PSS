import json
from fastapi import APIRouter, status
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewUser

#分割したエンドポイントの作成
users_endpoint = APIRouter()

#ユーザ一覧の取得
@users_endpoint.get('/users',tags=["users"],description="ユーザ一覧の取得",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_users():
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #SQLの実行
    query_text =f"SELECT id, name, email, authority, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM users;"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        user_list = [dict(row) for row in rows]
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to read users", "data":user_list})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to read users", "data":None})
    
#ユーザの作成
@users_endpoint.post('/users',tags=["users"],description="新規ユーザの作成",responses={
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def create_user(user:NewUser):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #バリデーションの実行
    if not(validate_data(user, 'user')):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})
    
    #SQLの実行
    query_text =f"INSERT INTO users(name, password, email, authority) VALUES ('{user.name}', '{user.password}', '{user.email}', {user.authority});"
    result,signup_user_id = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return JSONResponse(status_code=status.HTTP_201_CREATED,content={"message": "succeeded to create user", "data":{'user_id':signup_user_id}})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to create users", "data":None})

#ユーザの更新(後から定義・変更)
@users_endpoint.put('/users/{user_id}',tags=["users"],description="ユーザ情報の変更")
def update_user(user_id:str):
    print(user_id)
    return {'update-user':'put'}

#ユーザの削除
@users_endpoint.delete('/users/{user_id}',tags=["users"],description="ユーザの削除",responses={
    204: {"description": "No Content", "model": None},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def delete_user(user_id:str):
    connect_session = create_connect_session()
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    try:
        id = int(user_id)
    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "invalid user_id", "data":None})
    
    #SQLの実行
    query_text =f"DELETE FROM users WHERE id='{id}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to delete user", "data":None})