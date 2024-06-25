import json
from fastapi import APIRouter, status,Response
import sys
sys.path.append('../')
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import ProjectMembership

#分割したエンドポイントの作成
project_memberships_endpoint = APIRouter()

#ユーザとプロジェクトの紐付けの作成
@project_memberships_endpoint.get('/project_memberships')
def read_project_memberships(user_id, project_id):
    connect_session = create_connect_session()
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    if (project_id is not None and user_id is not None):
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=json.dumps({"message":"use user_id or project_id"}))
    
    if(project_id is not None):
        try:
            id = int(project_id)
        except Exception as e:
            print(e)
            return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"invalid project_id"}))
        # SQLの実行
        query_text =f"SELECT user_id, project_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships WHERE project_id='{id}';"
    elif(user_id is not None):
        try:
            id = int(user_id)
        except Exception as e:
            print(e)
            return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"invalid project_id"}))
        # SQLの実行
        query_text =f"SELECT user_id, project_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships WHERE user_id='{id}';"
    else:
        # SQLの実行
        query_text =f"SELECT user_id, project_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships;"
        
    result = execute_query(session=connect_session, query_text=query_text)
    
    if result is not None:
        rows = result.mappings().all()
        project_membership_list = [dict(row) for row in rows]
        return Response(status_code=status.HTTP_200_OK, content=json.dumps(project_membership_list, ensure_ascii=False))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to read database"}))

@project_memberships_endpoint.post('/project_memberships')
def create_project(project_membership:ProjectMembership):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #バリデーションの実行
    if not(validate_data(project_membership, 'project_membership')):
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"failed to validate"}))
    
    #SQLの実行
    query_text =f"INSERT INTO project_memberships(user_id, project_id) VALUES ('{project_membership.user_id}','{project_membership.project_id}');"
    result = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to create"}))