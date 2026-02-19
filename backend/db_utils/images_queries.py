from typing import Tuple, Any
from db_utils.commons import execute_query


def get_project_original_images_folder_path(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT original_images_folder_path FROM projects WHERE id = {project_id};"
    return execute_query(session, query_text)


def check_image_exists(session, name: str, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT id FROM images WHERE name = '{name}' AND project_id = {project_id};"
    return execute_query(session, query_text)


def insert_image(session, name: str, is_created_for_sql: str, caption_sql_value: str, project_id: int,
                 clustering_id: str, sentence_id: str, image_id: str, uploaded_user_id: int, folder_name: str = None) -> Tuple[Any, Any]:
    folder_name_sql = 'NULL' if folder_name is None else f"'{folder_name}'"
    
    query_text = f"""
        INSERT INTO images(name, folder_name, is_created_caption, caption, project_id, clustering_id, chromadb_sentence_id, chromadb_image_id, uploaded_user_id)
        VALUES ('{name}', {folder_name_sql}, {is_created_for_sql}, {caption_sql_value}, {project_id}, '{clustering_id}', '{sentence_id}', '{image_id}', '{uploaded_user_id}');
    """
    return execute_query(session, query_text)


def select_image_id_by_clustering_id(session, clustering_id: str) -> Tuple[Any, Any]:
    query_text = f"SELECT id FROM images WHERE clustering_id = '{clustering_id}';"
    return execute_query(session, query_text)


def select_caption_by_clustering_id(session, clustering_id: str) -> Tuple[Any, Any]:
    """Return caption for the image with the given clustering_id."""
    query_text = f"SELECT caption FROM images WHERE clustering_id = '{clustering_id}';"
    return execute_query(session, query_text)


def select_project_members(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT user_id FROM project_memberships WHERE project_id = {project_id};"
    return execute_query(session, query_text)


def update_project_members_continuous_state(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE project_memberships
        SET continuous_clustering_state = 2
        WHERE project_id = {project_id}
        AND init_clustering_state = 2
        AND continuous_clustering_state != 1;
    """
    return execute_query(session, query_text)


def get_images_by_project(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT id, name, folder_name, is_created_caption, caption ,project_id, uploaded_user_id, created_at FROM images WHERE project_id = {project_id};"
    return execute_query(session, query_text)


def get_folder_names_by_clustering_ids(session, clustering_ids: list) -> dict:
    """
    clustering_idのリストから、各clustering_idに対応するfolder_nameを取得する
    
    Returns:
        dict: {clustering_id: folder_name} の辞書
    """
    if not clustering_ids:
        return {}
    
    # clustering_idをSQLのIN句用に整形
    ids_str = "','".join(clustering_ids)
    query_text = f"SELECT clustering_id, folder_name FROM images WHERE clustering_id IN ('{ids_str}');"
    result, _ = execute_query(session, query_text)
    
    folder_name_dict = {}
    if result and result.rowcount > 0:
        for row in result.mappings():
            folder_name_dict[row['clustering_id']] = row['folder_name']
    
    return folder_name_dict


def delete_image(session, image_id: str) -> Tuple[Any, Any]:
    query_text = f"DELETE FROM images WHERE id = '{image_id}';"
    return execute_query(session, query_text)
