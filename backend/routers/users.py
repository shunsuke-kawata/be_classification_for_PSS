import json
from fastapi import APIRouter, status,Response
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import NewUser

#分割したエンドポイントの作成
users_endpoint = APIRouter()

#ユーザ一覧の取得
@users_endpoint.get('/users')
def read_users():
    connect_session = create_connect_session()
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #SQLの実行
    query_text =f"SELECT id, name, email, authority, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM users;"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        user_list = [dict(row) for row in rows]
        return Response(status_code=status.HTTP_200_OK,content=json.dumps(user_list))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to read database"}))
    
#ユーザの作成
@users_endpoint.post('/users')
def create_user(user:NewUser):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #バリデーションの実行
    if not(validate_data(user, 'user')):
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"failed to validate"}))
    
    #SQLの実行
    query_text =f"INSERT INTO users(name, password, email, authority) VALUES ('{user.name}', '{user.password}', '{user.email}', {user.authority});"
    result,signup_user_id = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_201_CREATED,content=json.dumps({'user_id':signup_user_id}))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to create"}))

#ユーザの更新(後から定義・変更)
@users_endpoint.put('/users/{user_id}')
def update_user(user_id:str):
    print(user_id)
    return {'update-user':'put'}

#ユーザの削除
@users_endpoint.delete('/users/{user_id}')
def delete_user(user_id:str):
    connect_session = create_connect_session()
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to connect database"}))
    
    try:
        id = int(user_id)
    except Exception as e:
        print(e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"invalid user_id"}))
    
    #SQLの実行
    query_text =f"DELETE FROM users WHERE id='{id}';"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to delete"}))