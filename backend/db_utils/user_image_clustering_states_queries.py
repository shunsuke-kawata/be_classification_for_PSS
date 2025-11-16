from typing import Tuple, Any
from db_utils.commons import execute_query


def get_user_image_clustering_states(session, where_clause: str) -> Tuple[Any, Any]:
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
    return execute_query(session=session, query_text=query_text)


def get_unclustered_count(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT COUNT(*) as unclustered_count
        FROM user_image_clustering_states
        WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 0;
    """
    return execute_query(session=session, query_text=query_text)


def mark_images_as_clustered(session, user_id: int, project_id: int, image_ids_str: str) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE user_image_clustering_states
        SET is_clustered = 1, clustered_at = CURRENT_TIMESTAMP(6)
        WHERE user_id = {user_id} AND project_id = {project_id} AND image_id IN ({image_ids_str});
    """
    return execute_query(session=session, query_text=query_text)


def mark_all_images_as_clustered(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE user_image_clustering_states
        SET is_clustered = 1, clustered_at = CURRENT_TIMESTAMP(6)
        WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 0;
    """
    return execute_query(session=session, query_text=query_text)


def get_clustered_count_after_mark_all(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT COUNT(*) as updated_count
        FROM user_image_clustering_states
        WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 1;
    """
    return execute_query(session=session, query_text=query_text)
