import json
from fastapi import APIRouter, status,Response
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session
from db_utils import project_memberships_queries as pm_queries
from db_utils.auth_queries import select_images_by_project, insert_user_image_state
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewProjectMembership, UpdateProjectMembershipState
from clustering.utils import Utils

#分割したエンドポイントの作成
project_memberships_endpoint = APIRouter()

#ユーザとプロジェクトの紐付けの取得
@project_memberships_endpoint.get('/project_memberships',tags=["project_memberships"],description="ユーザとプロジェクト間の関係一覧を取得",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_project_memberships(user_id=None, project_id=None):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #どちらかでしか検索を受け付けない
    if (project_id is not None and user_id is not None):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"use at least user_id or project_id", "data":None})

    if(project_id is not None):
        try:
            id = int(project_id)
        except Exception as e:
            print(e)
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"invalid project_id", "data":None})
        # SQLの実行
        result,_ = pm_queries.get_memberships_by_project(connect_session, id)
    elif(user_id is not None):
        try:
            id = int(user_id)
        except Exception as e:
            print(e)
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"invalid project_id", "data":None})
        # SQLの実行
        result,_ = pm_queries.get_memberships_by_user(connect_session, id)
    else:
        # SQLの実行
        result,_ = pm_queries.get_all_memberships(connect_session)
    
    if result is not None:
        rows = result.mappings().all()
        project_membership_list = [dict(row) for row in rows]
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to read project", "data":project_membership_list})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to read project_memberships", "data":None})

#ユーザとプロジェクトの紐付けを作成
@project_memberships_endpoint.post('/project_memberships',tags=["project_memberships"],description="ユーザとプロジェクト間の紐付けを行う",responses={
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def create_project_membership(project_membership:NewProjectMembership):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    #バリデーションの実行
    if not(validate_data(project_membership, 'project_membership')):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})

    mongo_result_id = Utils.generate_uuid()
    print(mongo_result_id)
    
    #SQLの実行
    result,_ = pm_queries.insert_project_membership(session=connect_session, user_id=project_membership.user_id, project_id=project_membership.project_id, mongo_result_id=mongo_result_id)
    
    if not(result is None):
        # プロジェクトメンバーシップ作成成功時、既存の画像に対してuser_image_clustering_statesレコードを作成
        try:
            # プロジェクト内の既存画像を取得
            images_result, _ = select_images_by_project(session=connect_session, project_id=project_membership.project_id)

            if images_result:
                images = images_result.mappings().all()
                created_count = 0

                for image in images:
                    image_id = image["id"]
                    # user_image_clustering_statesレコードを作成
                    state_result, _ = insert_user_image_state(session=connect_session, user_id=project_membership.user_id, image_id=image_id, project_id=project_membership.project_id, is_clustered=0)
                    if state_result:
                        created_count += 1

                print(f"✅ ユーザ{project_membership.user_id}に対して{created_count}個の画像クラスタリング状態レコードを作成しました")
        except Exception as e:
            print(f"⚠️ user_image_clustering_states作成エラー: {e}")
            # エラーが発生してもproject_membership作成自体は成功しているので処理は続行
        
        return JSONResponse(status_code=status.HTTP_201_CREATED,content={"message": "succeeded to create project_membership", "data":None})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to create project_membership", "data":None})

#プロジェクトメンバーシップのクラスタリング状態を更新
@project_memberships_endpoint.put('/project_memberships/state',tags=["project_memberships"],description="プロジェクトメンバーシップのクラスタリング状態を更新",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def update_project_membership_state(state_update: UpdateProjectMembershipState):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    # バリデーション
    if state_update.init_clustering_state is not None and state_update.init_clustering_state not in [0, 1, 2, 3]:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "init_clustering_state must be 0, 1, 2, or 3", "data":None})
    
    if state_update.continuous_clustering_state is not None and state_update.continuous_clustering_state not in [0, 1, 2]:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "continuous_clustering_state must be 0, 1, or 2", "data":None})
    
    # 少なくとも1つの状態が指定されているか確認
    if state_update.init_clustering_state is None and state_update.continuous_clustering_state is None:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "at least one state must be specified", "data":None})
    
    # 更新するフィールドを動的に構築
    update_fields = []
    if state_update.init_clustering_state is not None:
        update_fields.append(f"init_clustering_state={state_update.init_clustering_state}")
    if state_update.continuous_clustering_state is not None:
        update_fields.append(f"continuous_clustering_state={state_update.continuous_clustering_state}")
    
    update_clause = ", ".join(update_fields)
    
    #SQLの実行
    result, _ = pm_queries.update_project_membership_state(session=connect_session, user_id=state_update.user_id, project_id=state_update.project_id, init_clustering_state=state_update.init_clustering_state, continuous_clustering_state=state_update.continuous_clustering_state)
    
    if result is not None:
        # 更新されたレコードを取得（プロジェクト単位で取得して該当ユーザを抽出）
        result, _ = pm_queries.get_memberships_by_project_after_update(session=connect_session, project_id=state_update.project_id)

        if result is not None:
            rows = result.mappings().all()
            # 該当ユーザのレコードを抽出
            for r in rows:
                if r.get('user_id') == state_update.user_id:
                    membership_data = dict(r)
                    return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to update project_membership state", "data": membership_data})

        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to update project_membership state", "data":None})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to update project_membership state", "data":None})

#プロジェクト内の全ユーザーのcontinuous_clustering_stateを更新（画像アップロード時）
@project_memberships_endpoint.put('/project_memberships/state/{project_id}',tags=["project_memberships"],description="プロジェクト内の全ユーザーのcontinuous_clustering_stateを更新（画像アップロード時）",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def update_all_members_continuous_state(project_id: int):
    connect_session = create_connect_session()
    
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    # project_idの検証
    try:
        project_id = int(project_id)
    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"invalid project_id", "data":None})
    
    # プロジェクトが存在するか確認
    result, _ = pm_queries.project_exists(session=connect_session, project_id=project_id)
    if result is None or len(result.mappings().all()) == 0:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message":"project not found", "data":None})
    
    # init_clustering_stateが1または2のユーザーのcontinuous_clustering_stateを2に更新
    # init_clustering_stateが0または3のユーザーはそのまま
    result, _ = pm_queries.update_all_members_continuous_state(session=connect_session, project_id=project_id)
    
    if result is not None:
        # 更新後のプロジェクトメンバーシップ一覧を取得
        result, _ = pm_queries.get_memberships_by_project_after_update(session=connect_session, project_id=project_id)

        if result is not None:
            rows = result.mappings().all()
            membership_list = [dict(row) for row in rows]
            return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to update continuous_clustering_state for all members", "data": membership_list})

        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to update continuous_clustering_state for all members", "data":None})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to update continuous_clustering_state", "data":None})


#初期クラスタリング完了ユーザー一覧を取得（データコピー用）
@project_memberships_endpoint.get('/project_memberships/completed_users/{project_id}',tags=["project_memberships"],description="初期クラスタリングが完了したユーザー一覧を取得",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def get_completed_clustering_users(project_id: int):
    """
    プロジェクト内で初期クラスタリングが完了したユーザーの一覧を取得
    
    Args:
        project_id: プロジェクトID
    """
    connect_session = create_connect_session()
    
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    try:
        project_id = int(project_id)
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "invalid project_id", "data": None}
        )
    
    # init_clustering_state = 2（完了）のユーザーを取得
    query_text = f"""
        SELECT 
            pm.user_id,
            pm.project_id,
            pm.init_clustering_state,
            pm.continuous_clustering_state,
            pm.mongo_result_id,
            u.name as user_name,
            u.email as user_email,
            DATE_FORMAT(pm.created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at,
            DATE_FORMAT(pm.updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at
        FROM project_memberships pm
        JOIN users u ON pm.user_id = u.id
        WHERE pm.project_id = {project_id} AND pm.init_clustering_state = 2
        ORDER BY pm.updated_at DESC;
    """
    
    result, _ = pm_queries.get_completed_clustering_users(session=connect_session, project_id=project_id)
    
    if result is not None:
        rows = result.mappings().all()
        completed_users = [dict(row) for row in rows]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "succeeded to get completed clustering users",
                "data": completed_users
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to get completed clustering users", "data": None}
        )
