import json
from fastapi import APIRouter, status,Response
import sys
sys.path.append('../')
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import Project

#分割したエンドポイントの作成
projects_endpoint = APIRouter()

#ユーザ一覧の取得
@projects_endpoint.get('/projects')
def read_projects():
    connect_session = create_connect_session()
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))

    #SQLの実行
    query_text =f"SELECT id, name, password, description, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM projects;"
    result = execute_query(session=connect_session,query_text=query_text)
    if result is not None:
        rows = result.mappings().all()
        project_list = [dict(row) for row in rows]
        return Response(status_code=status.HTTP_200_OK,content=json.dumps(project_list, ensure_ascii=False))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to read database"}))
    
#プロジェクトの作成
@projects_endpoint.post('/projects')
def create_project(project:Project):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #バリデーションの実行
    if not(validate_data(project, 'project')):
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"failed to validate"}))
    
    #SQLの実行
    query_text =f"INSERT INTO projects(name, password, description,owner_id) VALUES ('{project.name}', '{project.password}','{project.description}','{project.owner_id}');"
    result = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to create"}))
    
#プロジェクトの削除
@projects_endpoint.delete('/projects/{id}')
def delete_project(id:str):
    connect_session = create_connect_session()
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to connect database"}))
    
    try:
        project_id = int(id)
    except Exception as e:
        print(e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"invalid project_id"}))
    
    #SQLの実行
    query_text =f"DELETE FROM projects WHERE id='{project_id}';"
    result = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to delete"}))