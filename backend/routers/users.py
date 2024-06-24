import json
from fastapi import APIRouter, HTTPException, status,Response
import sys
sys.path.append('../')
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import User

#複数ユーザに関するCRUD
#分割したエンドポイントの作成
users_endpoint = APIRouter()

#ユーザ一覧の取得
@users_endpoint.get('/users')
def get_users():
    connect_session = create_connect_session()
    if connect_session is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to connect to database")
    
    #SQLの実行
    query_text =f"SELECT id, name, pass, email, authority, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM users;"
    result = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        user_list = [dict(row) for row in rows]
        return Response(status_code=status.HTTP_200_OK,content=json.dumps(user_list))
    else:
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="failed to read database")
#ユーザの作成
@users_endpoint.post('/users')
def create_user(user:User):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to connect to database")
    
    #バリデーションの実行
    if not(validate_data(user, 'user')):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="failed to validate")
    
    #SQLの実行
    query_text =f"INSERT INTO users(name, pass, email, authority) VALUES ('{user.name}', '{user.password}', '{user.email}', {user.authority});"
    result = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="failed to create")

#ユーザの更新
@users_endpoint.put('/users/{id}')
def update_user(id:str):
    print(id)
    return {'update-user':'put'}

#ユーザの削除
@users_endpoint.delete('/users/{id}')
def delete_user(id:str):
    connect_session = create_connect_session()
    #データベース接続確認
    if connect_session is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to connect to database")
    
    try:
        user_id = int(id)
    except Exception as e:
        print(e)
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="invalid user_id")
    
    #SQLの実行
    query_text =f"DELETE FROM users WHERE id='{user_id}';"
    result = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail="failed to delete")