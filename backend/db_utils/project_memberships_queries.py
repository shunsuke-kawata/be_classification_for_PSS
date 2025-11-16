from typing import Tuple, Any, Optional
from db_utils.commons import execute_query


def get_memberships_by_project(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT user_id, project_id, init_clustering_state, continuous_clustering_state, mongo_result_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships WHERE project_id='{project_id}';"
    return execute_query(session=session, query_text=query_text)


def get_memberships_by_user(session, user_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT user_id, project_id, init_clustering_state, continuous_clustering_state, mongo_result_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships WHERE user_id='{user_id}';"
    return execute_query(session=session, query_text=query_text)


def get_all_memberships(session) -> Tuple[Any, Any]:
    query_text = f"SELECT user_id, project_id, mongo_result_id, init_clustering_state, continuous_clustering_state, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships;"
    return execute_query(session=session, query_text=query_text)


def insert_project_membership(session, user_id: int, project_id: int, mongo_result_id: str) -> Tuple[Any, Any]:
    query_text = f"INSERT INTO project_memberships(user_id, project_id,mongo_result_id) VALUES ('{user_id}','{project_id}','{mongo_result_id}');"
    return execute_query(session=session, query_text=query_text)


def update_project_membership_state(session, user_id: int, project_id: int, init_clustering_state: Optional[int] = None, continuous_clustering_state: Optional[int] = None) -> Tuple[Any, Any]:
    update_fields = []
    if init_clustering_state is not None:
        update_fields.append(f"init_clustering_state={init_clustering_state}")
    if continuous_clustering_state is not None:
        update_fields.append(f"continuous_clustering_state={continuous_clustering_state}")

    if not update_fields:
        return None, None

    update_clause = ", ".join(update_fields)
    query_text = f"UPDATE project_memberships SET {update_clause} WHERE user_id={user_id} AND project_id={project_id};"
    return execute_query(session=session, query_text=query_text)


def project_exists(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT id FROM projects WHERE id={project_id};"
    return execute_query(session=session, query_text=query_text)


def update_all_members_continuous_state(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE project_memberships 
        SET continuous_clustering_state = 2 
        WHERE project_id = {project_id} 
        AND init_clustering_state IN (1, 2);
    """
    return execute_query(session=session, query_text=query_text)


def get_memberships_by_project_after_update(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT user_id, project_id, init_clustering_state, continuous_clustering_state, mongo_result_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%sZ') as created_at, DATE_FORMAT(updated_at, '%Y-%m-%dT%H:%i:%sZ') as updated_at FROM project_memberships WHERE project_id={project_id};"
    return execute_query(session=session, query_text=query_text)


def get_completed_clustering_users(session, project_id: int) -> Tuple[Any, Any]:
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
    return execute_query(session=session, query_text=query_text)
