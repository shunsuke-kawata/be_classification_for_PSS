import json
from fastapi import APIRouter, HTTPException, status,Response
import sys
sys.path.append('../')
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import LoginUser

#分割したエンドポイントの作成
#ログイン操作
auth_endpoint = APIRouter()

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
    result = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        login_info = [dict(row) for row in rows][0]
        return Response(status_code=status.HTTP_200_OK,content=json.dumps(login_info))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to login"}))
    