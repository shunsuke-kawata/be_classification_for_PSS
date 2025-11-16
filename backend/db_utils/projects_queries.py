from typing import Tuple, Any, Optional
from db_utils.commons import execute_query


def get_projects(session) -> Tuple[Any, Any]:
    query_text = """
        SELECT id, name, description,original_images_folder_path, owner_id
        FROM projects;
    """
    return execute_query(session=session, query_text=query_text)


def get_projects_for_user(session, user_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT projects.id, projects.name, projects.description, 
               projects.original_images_folder_path, projects.owner_id,
               projects.created_at,
               projects.updated_at,
               project_memberships.init_clustering_state,
               project_memberships.continuous_clustering_state,
               project_memberships.mongo_result_id,
               CASE WHEN project_memberships.user_id IS NOT NULL THEN true ELSE false END as joined
        FROM projects
        LEFT JOIN project_memberships
        ON projects.id = project_memberships.project_id AND project_memberships.user_id = {user_id};
    """
    return execute_query(session=session, query_text=query_text)


def get_project(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT id, name, description,original_images_folder_path, owner_id, created_at, updated_at
        FROM projects WHERE id = {project_id};
    """
    return execute_query(session=session, query_text=query_text)


def get_project_for_user(session, project_id: int, user_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT projects.id, projects.name, projects.description, 
           projects.original_images_folder_path, projects.owner_id,
           projects.created_at,
           projects.updated_at,
           project_memberships.init_clustering_state,
           project_memberships.continuous_clustering_state,
           project_memberships.mongo_result_id,
           CASE WHEN project_memberships.user_id IS NOT NULL THEN true ELSE false END as joined
        FROM projects
        LEFT JOIN project_memberships
        ON projects.id = project_memberships.project_id
        WHERE projects.id = {project_id} AND project_memberships.user_id = {user_id};
    """
    return execute_query(session=session, query_text=query_text)


def insert_project(session, name: str, password: str, description: str, original_images_folder_path: str, owner_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        INSERT INTO projects(name, password, description,original_images_folder_path, owner_id)
        VALUES ('{name}', '{password}', '{description}','{original_images_folder_path}', {owner_id});
    """
    return execute_query(session=session, query_text=query_text)


def delete_project(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"DELETE FROM projects WHERE id = {project_id};"
    return execute_query(session=session, query_text=query_text)
