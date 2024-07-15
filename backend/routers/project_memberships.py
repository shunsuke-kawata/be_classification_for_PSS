import json
from fastapi import APIRouter, status,Response
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import NewProjectMembership

#分割したエンドポイントの作成
project_memberships_endpoint = APIRouter()

#ユーザとプロジェクトの紐付けの作成
@project_memberships_endpoint.get('/project_memberships',tags=["project_memberships"],description="ユーザとプロジェクト間の関係一覧を取得")
def read_project_memberships(user_id=None, project_id=None):
    connect_session = create_connect_session()
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #どちらかでしか検索を受け付けない
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
        
    result,_ = execute_query(session=connect_session, query_text=query_text)
    
    if result is not None:
        rows = result.mappings().all()
        project_membership_list = [dict(row) for row in rows]
        print(project_membership_list)
        return Response(status_code=status.HTTP_200_OK, content=json.dumps(project_membership_list, ensure_ascii=False))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to read database"}))

@project_memberships_endpoint.post('/project_memberships',tags=["project_memberships"],description="ユーザとプロジェクト間の紐付けを行う")
def create_project_membership(project_membership:NewProjectMembership):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({"message":"failed to connect database"}))
    
    #バリデーションの実行
    if not(validate_data(project_membership, 'project_membership')):
        return Response(status_code=status.HTTP_400_BAD_REQUEST,content=json.dumps({"message":"failed to validate"}))
    
    #SQLの実行
    query_text =f"INSERT INTO project_memberships(user_id, project_id) VALUES ('{project_membership.user_id}','{project_membership.project_id}');"
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if not(result is None):
        return Response(status_code=status.HTTP_201_CREATED,content=json.dumps({'project_membership':f'{project_membership.user_id}-{project_membership.project_id}'}))
    else:
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to create"}))