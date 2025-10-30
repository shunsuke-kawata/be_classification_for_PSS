from fastapi import APIRouter, status, Query
from fastapi.responses import JSONResponse
import sys
import os
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session, execute_query
from db_utils.models import CustomResponseModel

# エンドポイントの作成
user_image_clustering_states_endpoint = APIRouter()

@user_image_clustering_states_endpoint.get(
    '/user_image_clustering_states',
    tags=["user_image_clustering_states"],
    description="ユーザの画像クラスタリング状態一覧を取得",
    responses={
        200: {"description": "OK", "model": CustomResponseModel},
        400: {"description": "Bad Request", "model": CustomResponseModel},
        500: {"description": "Internal Server Error", "model": CustomResponseModel}
    }
)
def get_user_image_clustering_states(
    user_id: int = None,
    project_id: int = None,
    is_clustered: int = None  # 0: 未クラスタリング, 1: クラスタリング済み
):
    """
    ユーザの画像クラスタリング状態を取得
    
    Args:
        user_id: ユーザID（オプション）
        project_id: プロジェクトID（オプション）
        is_clustered: クラスタリング状態（オプション、0 or 1）
    """
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    # クエリ条件を構築
    conditions = []
    if user_id is not None:
        conditions.append(f"user_id = {user_id}")
    if project_id is not None:
        conditions.append(f"project_id = {project_id}")
    if is_clustered is not None:
        if is_clustered not in [0, 1]:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "is_clustered must be 0 or 1", "data": None}
            )
        conditions.append(f"is_clustered = {is_clustered}")
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    
    query_text = f"""
        SELECT 
            uics.user_id,
            uics.image_id,
            uics.project_id,
            uics.is_clustered,
            uics.clustered_at,
            uics.created_at,
            uics.updated_at,
            i.name as image_name,
            i.clustering_id
        FROM user_image_clustering_states uics
        JOIN images i ON uics.image_id = i.id
        WHERE {where_clause}
        ORDER BY uics.created_at DESC;
    """
    
    result, _ = execute_query(connect_session, query_text)
    
    if result:
        rows = result.mappings().all()
        states_list = [dict(row) for row in rows]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "succeeded to read user image clustering states", "data": states_list}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to read user image clustering states", "data": None}
        )


@user_image_clustering_states_endpoint.get(
    '/user_image_clustering_states/unclustered_count',
    tags=["user_image_clustering_states"],
    description="ユーザの未クラスタリング画像数を取得",
    responses={
        200: {"description": "OK", "model": CustomResponseModel},
        400: {"description": "Bad Request", "model": CustomResponseModel},
        500: {"description": "Internal Server Error", "model": CustomResponseModel}
    }
)
def get_unclustered_count(user_id: int, project_id: int):
    """
    ユーザの未クラスタリング画像数を取得
    
    Args:
        user_id: ユーザID
        project_id: プロジェクトID
    """
    if user_id is None or project_id is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "user_id and project_id are required", "data": None}
        )
    
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    query_text = f"""
        SELECT COUNT(*) as unclustered_count
        FROM user_image_clustering_states
        WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 0;
    """
    
    result, _ = execute_query(connect_session, query_text)
    
    if result:
        count = result.mappings().first()["unclustered_count"]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "succeeded to get unclustered count",
                "data": {
                    "user_id": user_id,
                    "project_id": project_id,
                    "unclustered_count": count
                }
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to get unclustered count", "data": None}
        )


@user_image_clustering_states_endpoint.put(
    '/user_image_clustering_states/mark_clustered',
    tags=["user_image_clustering_states"],
    description="指定された画像をクラスタリング済みとしてマーク",
    responses={
        200: {"description": "OK", "model": CustomResponseModel},
        400: {"description": "Bad Request", "model": CustomResponseModel},
        500: {"description": "Internal Server Error", "model": CustomResponseModel}
    }
)
def mark_images_as_clustered(
    user_id: int = Query(...),
    project_id: int = Query(...),
    image_ids: List[int] = Query(...)
):
    """
    指定された画像をクラスタリング済みとしてマーク
    
    Args:
        user_id: ユーザID
        project_id: プロジェクトID
        image_ids: 画像IDのリスト
    """
    if not image_ids or len(image_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "image_ids must not be empty", "data": None}
        )
    
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    # 複数の画像IDを一度に更新
    image_ids_str = ",".join(map(str, image_ids))
    
    query_text = f"""
        UPDATE user_image_clustering_states
        SET is_clustered = 1, clustered_at = CURRENT_TIMESTAMP(6)
        WHERE user_id = {user_id} AND project_id = {project_id} AND image_id IN ({image_ids_str});
    """
    
    result, _ = execute_query(connect_session, query_text)
    
    if result is not None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "succeeded to mark images as clustered",
                "data": {
                    "user_id": user_id,
                    "project_id": project_id,
                    "updated_image_count": len(image_ids)
                }
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to mark images as clustered", "data": None}
        )


@user_image_clustering_states_endpoint.put(
    '/user_image_clustering_states/mark_all_clustered',
    tags=["user_image_clustering_states"],
    description="ユーザのプロジェクト内の全画像をクラスタリング済みとしてマーク（初期クラスタリング実行時用）",
    responses={
        200: {"description": "OK", "model": CustomResponseModel},
        400: {"description": "Bad Request", "model": CustomResponseModel},
        500: {"description": "Internal Server Error", "model": CustomResponseModel}
    }
)
def mark_all_images_as_clustered(user_id: int, project_id: int):
    """
    ユーザのプロジェクト内の全画像をクラスタリング済みとしてマーク
    初期クラスタリング実行時に呼び出される
    
    Args:
        user_id: ユーザID
        project_id: プロジェクトID
    """
    if user_id is None or project_id is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "user_id and project_id are required", "data": None}
        )
    
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    query_text = f"""
        UPDATE user_image_clustering_states
        SET is_clustered = 1, clustered_at = CURRENT_TIMESTAMP(6)
        WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 0;
    """
    
    result, _ = execute_query(connect_session, query_text)
    
    if result is not None:
        # 更新された件数を取得
        count_query = f"""
            SELECT COUNT(*) as updated_count
            FROM user_image_clustering_states
            WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 1;
        """
        count_result, _ = execute_query(connect_session, count_query)
        updated_count = count_result.mappings().first()["updated_count"] if count_result else 0
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "succeeded to mark all images as clustered",
                "data": {
                    "user_id": user_id,
                    "project_id": project_id,
                    "total_clustered_count": updated_count
                }
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to mark all images as clustered", "data": None}
        )
