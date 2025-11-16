import json
from clustering.utils import Utils
from fastapi import APIRouter, HTTPException, status,Response
import sys
import os

from fastapi.responses import JSONResponse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session
from db_utils.auth_queries import (
    query_user_login,
    verify_project_password,
    insert_project_membership,
    select_images_by_project,
    insert_user_image_state,
)
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
    result,_ = query_user_login(session=connect_session, name=login_user.name, email=login_user.email, password=login_user.password)
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
    result,_ = verify_project_password(session=connect_session, project_id=id, password=join_user.project_password)
    
    if (result is None):
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to join", "data":None})

    rows = result.mappings().all()
    project_pass_info_list = [dict(row) for row in rows]
    
    if(len(project_pass_info_list)==0):
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to join", "data":None})
    
    project_pass_info = project_pass_info_list[0]
    print(project_pass_info)
    
    mongo_result_id = Utils.generate_uuid()
    
    #SQLの実行
    result,_ = insert_project_membership(session=connect_session, user_id=join_user.user_id, project_id=project_pass_info['id'], mongo_result_id=mongo_result_id)
    
    if not(result is None):
        # プロジェクト参加成功時、既存の画像に対してuser_image_clustering_statesレコードを作成
        try:
            # プロジェクト内の既存画像を取得
            images_result, _ = select_images_by_project(session=connect_session, project_id=join_user.project_id)
            
            if images_result:
                images = images_result.mappings().all()
                created_count = 0
                
                for image in images:
                    image_id = image["id"]
                    # user_image_clustering_statesレコードを作成
                    state_result, _ = insert_user_image_state(session=connect_session, user_id=join_user.user_id, image_id=image_id, project_id=join_user.project_id, is_clustered=0)
                    if state_result:
                        created_count += 1
                
                print(f"✅ ユーザ{join_user.user_id}に対して{created_count}個の画像クラスタリング状態レコードを作成しました")
        except Exception as e:
            print(f"⚠️ user_image_clustering_states作成エラー: {e}")
            # エラーが発生してもプロジェクト参加自体は成功しているので処理は続行
        
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to join", "data":None})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to join", "data":None})