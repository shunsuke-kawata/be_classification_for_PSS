from typing import Tuple, Any
from db_utils.commons import execute_query


def query_user_login(session, name: str, email: str, password: str) -> Tuple[Any, Any]:
    """指定された name/email と password で users テーブルを検索します。

    Returns (result, last_insert_id) same as execute_query.
    """
    query_text = f"SELECT id, name, password, email, authority FROM users WHERE (name='{name}' OR email='{email}') AND password='{password}';"
    return execute_query(session=session, query_text=query_text)


def verify_project_password(session, project_id: int, password: str) -> Tuple[Any, Any]:
    """projects テーブルで id と password を確認します。"""
    query_text = f"SELECT id,password FROM projects WHERE id='{project_id}' AND password='{password}';"
    return execute_query(session=session, query_text=query_text)


def insert_project_membership(session, user_id: int, project_id: int, mongo_result_id: str) -> Tuple[Any, Any]:
    """project_memberships に参加レコードを挿入します。"""
    query_text = f"INSERT INTO project_memberships(user_id, project_id,mongo_result_id) VALUES ('{user_id}','{project_id}','{mongo_result_id}');"
    return execute_query(session=session, query_text=query_text)


def select_images_by_project(session, project_id: int) -> Tuple[Any, Any]:
    """指定プロジェクトの images.id を取得します。"""
    query_text = f"SELECT id FROM images WHERE project_id = {project_id};"
    return execute_query(session=session, query_text=query_text)


def insert_user_image_state(session, user_id: int, image_id: int, project_id: int, is_clustered: int = 0) -> Tuple[Any, Any]:
    """user_image_clustering_states にレコードを挿入します。"""
    query_text = f"""
        INSERT INTO user_image_clustering_states(user_id, image_id, project_id, is_clustered)
        VALUES ({user_id}, {image_id}, {project_id}, {is_clustered});
    """
    return execute_query(session=session, query_text=query_text)


def bulk_insert_user_image_states(session, user_ids: list, image_id: int, project_id: int, is_clustered: int = 0) -> Tuple[Any, Any]:
    """user_image_clustering_states に複数レコードを一括挿入します。"""
    if not user_ids:
        return None, None
    
    values = ", ".join([f"({user_id}, {image_id}, {project_id}, {is_clustered})" for user_id in user_ids])
    query_text = f"""
        INSERT INTO user_image_clustering_states(user_id, image_id, project_id, is_clustered)
        VALUES {values};
    """
    return execute_query(session=session, query_text=query_text)
