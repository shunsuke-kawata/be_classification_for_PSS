from typing import Tuple, Any
from db_utils.commons import execute_query


def get_membership_init_and_mongo(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT init_clustering_state, mongo_result_id
        FROM project_memberships
        WHERE user_id = {user_id} AND project_id = {project_id};
    """
    return execute_query(session=session, query_text=query_text)


def get_membership_mongo_and_init(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    # same as get_membership_init_and_mongo but kept for clarity
    return get_membership_init_and_mongo(session, user_id, project_id)


def get_membership_and_project_info(session, project_id: int, user_id: int) -> Tuple[Any, Any]:
    # include continuous_clustering_state because callers (e.g. execute_continuous_clustering)
    # expect this column to be present in the result mapping
    query_text = f"""
        SELECT project_memberships.init_clustering_state,
               project_memberships.continuous_clustering_state,
               project_memberships.mongo_result_id,
               projects.original_images_folder_path
        FROM project_memberships
        JOIN projects ON project_memberships.project_id = projects.id
        WHERE project_memberships.project_id = {project_id} AND project_memberships.user_id = {user_id};
    """
    return execute_query(session=session, query_text=query_text)


def select_images_for_init(session, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT clustering_id, chromadb_sentence_id, chromadb_image_id
        FROM images
        WHERE project_id = {project_id} AND is_created_caption = TRUE;
    """
    return execute_query(session=session, query_text=query_text)


def update_init_state(session, user_id: int, project_id: int, state) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE project_memberships
        SET init_clustering_state = '{state}'
        WHERE project_id = {project_id} AND user_id = {user_id};
    """
    return execute_query(session=session, query_text=query_text)


def mark_user_images_clustered(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE user_image_clustering_states
        SET is_clustered = 1, clustered_at = CURRENT_TIMESTAMP(6)
        WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 0;
    """
    return execute_query(session=session, query_text=query_text)


def mark_user_images_clustered_with_executed_count(session, user_id: int, project_id: int, executed_count: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE user_image_clustering_states
        SET is_clustered = 1, executed_clustering_count = {executed_count}, clustered_at = CURRENT_TIMESTAMP(6)
        WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 0;
    """
    return execute_query(session=session, query_text=query_text)


def get_unclustered_images(session, project_id: int, user_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT 
            i.id as image_id,
            i.name as image_name,
            i.clustering_id, 
            i.chromadb_sentence_id, 
            i.chromadb_image_id,
            i.caption,
            i.created_at
        FROM images i
        LEFT JOIN user_image_clustering_states uics 
            ON i.id = uics.image_id AND uics.user_id = {user_id}
        WHERE i.project_id = {project_id} 
            AND i.is_created_caption = TRUE
            AND (uics.is_clustered = 0 OR uics.is_clustered IS NULL);
    """
    return execute_query(session=session, query_text=query_text)


def get_user_info(session, user_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT id, name, email FROM users WHERE id = {user_id};"
    return execute_query(session=session, query_text=query_text)


def get_executed_clustering_count(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT executed_clustering_count FROM project_memberships WHERE user_id = {user_id} AND project_id = {project_id};"
    return execute_query(session=session, query_text=query_text)


def get_chromadb_image_id_by_clustering_id(session, clustering_id: str, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT chromadb_image_id FROM images
        WHERE clustering_id = '{clustering_id}' AND project_id = {project_id};
    """
    return execute_query(session=session, query_text=query_text)


def get_chromadb_sentence_id_by_clustering_id(session, clustering_id: str, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT chromadb_sentence_id FROM images
        WHERE clustering_id = '{clustering_id}' AND project_id = {project_id};
    """
    return execute_query(session=session, query_text=query_text)


def update_user_image_state_for_image(session, user_id: int, image_id: int, new_count: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE user_image_clustering_states
        SET is_clustered = 1, 
            executed_clustering_count = {new_count}, 
            clustered_at = CURRENT_TIMESTAMP(6)
        WHERE user_id = {user_id} AND image_id = {image_id};
    """
    return execute_query(session=session, query_text=query_text)


def update_project_executed_clustering_count(session, user_id: int, project_id: int, new_count: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE project_memberships
        SET executed_clustering_count = {new_count}
        WHERE user_id = {user_id} AND project_id = {project_id};
    """
    return execute_query(session=session, query_text=query_text)


def get_unclustered_count_for_project(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT COUNT(*) as unclustered_count
        FROM images i
        LEFT JOIN user_image_clustering_states uics 
            ON i.id = uics.image_id AND uics.user_id = {user_id}
        WHERE i.project_id = {project_id} 
            AND i.is_created_caption = TRUE
            AND (uics.is_clustered = 0 OR uics.is_clustered IS NULL);
    """
    return execute_query(session=session, query_text=query_text)


def update_continuous_state(session, user_id: int, project_id: int, new_state: int) -> Tuple[Any, Any]:
    query_text = f"""
        UPDATE project_memberships
        SET continuous_clustering_state = {new_state}
        WHERE user_id = {user_id} AND project_id = {project_id};
    """
    return execute_query(session=session, query_text=query_text)


def get_project_info_and_mongo(session, project_id: int, user_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT 
            p.name as project_name,
            p.original_images_folder_path,
            pm.mongo_result_id,
            pm.init_clustering_state
        FROM projects p
        JOIN project_memberships pm ON p.id = pm.project_id
        WHERE p.id = {project_id} AND pm.user_id = {user_id};
    """
    return execute_query(session=session, query_text=query_text)


def get_project_name(session, project_id: int) -> Tuple[Any, Any]:
    """Return project name for given project id."""
    query_text = f"""
        SELECT name FROM projects WHERE id = {project_id}
    """
    return execute_query(session=session, query_text=query_text)


def get_image_name_by_id(session, image_id: int) -> Tuple[Any, Any]:
    """Return image name (path) from images table by id."""
    query_text = f"""
        SELECT name FROM images WHERE id = {image_id};
    """
    return execute_query(session=session, query_text=query_text)


def membership_exists(session, project_id: int, user_id: int) -> Tuple[Any, Any]:
    query_text = f"SELECT COUNT(*) as cnt FROM project_memberships WHERE project_id = {project_id} AND user_id = {user_id}"
    return execute_query(session=session, query_text=query_text)


def get_user_clustering_states_by_clustering_id(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    """
    ユーザーの画像ごとのクラスタリング状態を取得（clustering_idベース）
    """
    query_text = f"""
        SELECT
            i.clustering_id,
            uics.executed_clustering_count
        FROM user_image_clustering_states uics
        JOIN images i ON uics.image_id = i.id
        WHERE uics.user_id = {user_id}
          AND uics.project_id = {project_id}
          AND i.project_id = {project_id}
          AND i.is_created_caption = TRUE
    """
    return execute_query(session=session, query_text=query_text)


def copy_clustering_states_by_clustering_id(session, source_user_id: int, target_user_id: int, project_id: int) -> Tuple[Any, Any]:
    """
    コピー元のexecuted_clustering_countをコピー先にコピー
    clustering_idで画像を紐付ける
    """
    query_text = f"""
        UPDATE user_image_clustering_states AS target_uics
        JOIN images AS target_img ON target_uics.image_id = target_img.id
        JOIN images AS source_img ON target_img.clustering_id = source_img.clustering_id
        JOIN user_image_clustering_states AS source_uics 
            ON source_img.id = source_uics.image_id 
            AND source_uics.user_id = {source_user_id}
            AND source_uics.project_id = {project_id}
        SET 
            target_uics.is_clustered = 1,
            target_uics.executed_clustering_count = source_uics.executed_clustering_count,
            target_uics.clustered_at = CURRENT_TIMESTAMP(6)
        WHERE target_uics.user_id = {target_user_id}
          AND target_uics.project_id = {project_id}
          AND target_img.project_id = {project_id}
          AND source_img.project_id = {project_id}
          AND target_img.is_created_caption = TRUE
          AND source_img.is_created_caption = TRUE
    """
    return execute_query(session=session, query_text=query_text)


def get_image_counts_for_clustering_counts(session, user_id: int, project_id: int) -> Tuple[Any, Any]:
    query_text = f"""
        SELECT
            uics.executed_clustering_count AS exec_count,
            i.clustering_id AS clustering_id
        FROM user_image_clustering_states uics
        JOIN images i ON uics.image_id = i.id
        WHERE uics.user_id = {user_id}
          AND uics.project_id = {project_id}
          AND i.project_id = {project_id}
          AND i.is_created_caption = TRUE
    """
    return execute_query(session=session, query_text=query_text)
