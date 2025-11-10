import copy
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, status,Response
import sys
import os

from fastapi.responses import JSONResponse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, LoginUser,JoinUser
from config import INIT_CLUSTERING_STATUS,CONTINUOUS_CLUSTERING_STATUS,DEFAULT_IMAGE_PATH,DEFAULT_OUTPUT_PATH
from clustering.clustering_manager import ChromaDBManager, InitClusteringManager
from clustering.mongo_db_manager import MongoDBManager
from clustering.mongo_db_manager import MongoDBManager
from fastapi import BackgroundTasks, Query
from collections import defaultdict
from typing import List
from clustering.mongo_result_manager import ResultManager

#分割したエンドポイントの作成
#ログイン操作
action_endpoint = APIRouter()

@action_endpoint.get("/action/clustering/result/{mongo_result_id}",tags=["action"],description="初期クラスタリング結果を取得する")
def get_clustering_result(mongo_result_id:str):
    from clustering.mongo_result_manager import ResultManager
    result_manager = ResultManager(mongo_result_id)
    
    # ResultManagerのget_result()メソッドを使用
    result_data = result_manager.get_result()
    
    if result_data:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "success", "result": result_data}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "Clustering result not found"}
        )


@action_endpoint.get("/action/clustering/all_nodes/{mongo_result_id}",tags=["action"],description="指定されたmongo_result_idのall_nodesを取得する")
def get_all_nodes(mongo_result_id: str):
    """
    指定されたmongo_result_idに紐づくall_nodesを取得する
    
    Args:
        mongo_result_id (str): MongoDBの結果ID
        
    Returns:
        JSONResponse: all_nodesの情報
    """
    try:
        # 入力バリデーション
        if not mongo_result_id or not mongo_result_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "mongo_result_id is required", "data": None}
            )
        
        print(f"📋 get_all_nodes呼び出し: mongo_result_id={mongo_result_id}")
        
        # ResultManagerを初期化
        result_manager = ResultManager(mongo_result_id)
        
        # all_nodesを取得
        all_nodes_data = result_manager.get_all_nodes()
        
        if all_nodes_data is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "message": "all_nodes not found for the given mongo_result_id",
                    "data": None
                }
            )
        
        print(f"✅ all_nodes取得成功: {len(all_nodes_data)}個のノード")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "success",
                "data": all_nodes_data
            }
        )
        
    except Exception as e:
        print(f"❌ get_all_nodes処理中にエラー: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": f"Internal server error occurred: {str(e)}",
                "data": None
            }
        )


@action_endpoint.post("/action/clustering/copy",tags=["action"],description="他ユーザーのクラスタリング結果をコピーする")
def copy_clustering_data(
    source_user_id: int = Query(..., description="コピー元のユーザーID"),
    target_user_id: int = Query(..., description="コピー先のユーザーID"),
    project_id: int = Query(..., description="プロジェクトID")
):
    """
    他ユーザーのクラスタリング結果（all_nodesとresult）をコピーする
    
    Args:
        source_user_id: コピー元のユーザーID（init_clustering_stateが2のユーザー）
        target_user_id: コピー先のユーザーID
        project_id: プロジェクトID
        
    Returns:
        JSONResponse: コピー結果
    """
    connect_session = create_connect_session()
    
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    try:
        # 1. コピー元ユーザーのinit_clustering_stateが2（完了）かチェック
        source_check_query = f"""
            SELECT init_clustering_state, mongo_result_id
            FROM project_memberships
            WHERE user_id = {source_user_id} AND project_id = {project_id};
        """
        source_result, _ = execute_query(session=connect_session, query_text=source_check_query)
        
        if not source_result:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "source user membership not found", "data": None}
            )
        
        source_data = source_result.mappings().first()
        if source_data["init_clustering_state"] != INIT_CLUSTERING_STATUS.FINISHED:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "source user has not completed init clustering", "data": None}
            )
        
        source_mongo_result_id = source_data["mongo_result_id"]
        
        # 2. コピー先ユーザーのmongo_result_idを取得
        target_check_query = f"""
            SELECT mongo_result_id, init_clustering_state
            FROM project_memberships
            WHERE user_id = {target_user_id} AND project_id = {project_id};
        """
        target_result, _ = execute_query(session=connect_session, query_text=target_check_query)
        
        if not target_result:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "target user membership not found", "data": None}
            )
        
        target_data = target_result.mappings().first()
        target_mongo_result_id = target_data["mongo_result_id"]
        
        # 3. コピー元のall_nodesとresultを取得
        source_result_manager = ResultManager(source_mongo_result_id)
        source_all_nodes = source_result_manager.get_all_nodes()
        source_result_data = source_result_manager.get_result()
        
        if source_all_nodes is None or source_result_data is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "source clustering data not found", "data": None}
            )
        
        # 4. コピー先にデータをコピー（ディープコピーで完全に独立したデータを作成）
        target_result_manager = ResultManager(target_mongo_result_id)
        copied_all_nodes = copy.deepcopy(source_all_nodes)
        copied_result = copy.deepcopy(source_result_data)
        
        target_result_manager.update_result(copied_result, copied_all_nodes)
        
        # 5. コピー先ユーザーのinit_clustering_stateを2（完了）に更新
        update_state_query = f"""
            UPDATE project_memberships
            SET init_clustering_state = {INIT_CLUSTERING_STATUS.FINISHED}
            WHERE user_id = {target_user_id} AND project_id = {project_id};
        """
        _, _ = execute_query(session=connect_session, query_text=update_state_query)
        
        # 6. コピー先ユーザーの全画像をクラスタリング済みとしてマーク
        mark_clustered_query = f"""
            UPDATE user_image_clustering_states
            SET is_clustered = 1, clustered_at = CURRENT_TIMESTAMP(6)
            WHERE user_id = {target_user_id} AND project_id = {project_id} AND is_clustered = 0;
        """
        _, _ = execute_query(session=connect_session, query_text=mark_clustered_query)
        
        print(f"✅ ユーザー{source_user_id}のデータをユーザー{target_user_id}にコピー完了")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "succeeded to copy clustering data",
                "data": {
                    "source_user_id": source_user_id,
                    "target_user_id": target_user_id,
                    "project_id": project_id,
                    "source_mongo_result_id": source_mongo_result_id,
                    "target_mongo_result_id": target_mongo_result_id
                }
            }
        )
        
    except Exception as e:
        print(f"❌ copy_clustering_data処理中にエラー: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": f"failed to copy clustering data: {str(e)}",
                "data": None
            }
        )


@action_endpoint.get("/action/clustering/init/{project_id}", tags=["action"], description="初期クラスタリングを実装する")
def execute_init_clustering(
    project_id: int = None,
    user_id: int = None,
    background_tasks: BackgroundTasks = None
):
    if project_id is None or user_id is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "project_id and user_id is required", "data": None}
        )

    try:
        project_id = int(project_id)
        user_id = int(user_id)
    except:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "project_id or user_id is invalid", "data": None}
        )

    connect_session = create_connect_session()
    query_text = f"""
        SELECT project_memberships.init_clustering_state, project_memberships.mongo_result_id,projects.original_images_folder_path
        FROM project_memberships
        JOIN projects ON project_memberships.project_id = projects.id
        WHERE project_memberships.project_id = {project_id} AND project_memberships.user_id = {user_id};
    """

    result, _ = execute_query(session=connect_session, query_text=query_text)
    result_mappings = result.mappings().first()

    if result_mappings is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "project or membership not found", "data": None}
        )

    init_clustering_state = result_mappings["init_clustering_state"]
    original_images_folder_path = result_mappings["original_images_folder_path"]
    mongo_result_id = result_mappings["mongo_result_id"]

    if init_clustering_state == INIT_CLUSTERING_STATUS.EXECUTING or init_clustering_state ==INIT_CLUSTERING_STATUS.FINISHED:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "init clustering already started", "data": None}
        )

    # 対象画像の取得
    query_text = f"""
        SELECT clustering_id, chromadb_sentence_id, chromadb_image_id
        FROM images
        WHERE project_id = {project_id} AND is_created_caption = TRUE;
    """

    result, _ = execute_query(session=connect_session, query_text=query_text)
    if result is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to get images", "data": None}
        )

    rows = result.mappings().all()
    # 検索用辞書を作成
    by_clustering_id = {}
    by_chromadb_sentence_id = {}
    by_chromadb_image_id = {}

    for row in rows:
        cid = row["clustering_id"]
        sid = row["chromadb_sentence_id"]
        iid = row["chromadb_image_id"]
        
        by_clustering_id[cid] = {"sentence_id": sid, "image_id": iid}
        by_chromadb_sentence_id[sid] = {"clustering_id": cid, "image_id": iid}
        by_chromadb_image_id[iid] = {"clustering_id": cid,"sentence_id":sid}
    
    # バックグラウンド処理に渡す関数
    def run_clustering(cid_dict: dict, sid_dict: dict, iid_dict: dict, project_id: int, original_images_folder_path: str):
        try:
            # プロジェクト名を取得
            project_name_query = f"""
                SELECT name FROM projects WHERE id = {project_id}
            """
            project_result, _ = execute_query(session=connect_session, query_text=project_name_query)
            project_mapping = project_result.mappings().first() if project_result else None
            project_name = project_mapping['name'] if project_mapping else f"Project_{project_id}"
            
            print(f"🏷️ プロジェクト名を取得: {project_name} (project_id: {project_id})")
            
            cl_module = InitClusteringManager(
                sentence_name_db=ChromaDBManager('sentence_name_embeddings'),
                sentence_usage_db=ChromaDBManager('sentence_usage_embeddings'),
                sentence_category_db=ChromaDBManager('sentence_category_embeddings'),
                image_db=ChromaDBManager("image_embeddings"),
                images_folder_path=f"./{DEFAULT_IMAGE_PATH}/{original_images_folder_path}",
                output_base_path=f"./{DEFAULT_OUTPUT_PATH}/{project_id}",
            )
            
            target_sentence_ids = list(sid_dict.keys())
            target_image_ids = list(iid_dict.keys())
            
            # sentence_name_dbから直接sentence_idを使用してembeddingsを取得
            sentence_data = cl_module.sentence_name_db.get_data_by_sentence_ids(target_sentence_ids)
            embeddings = sentence_data['embeddings']
            cluster_num, _ = cl_module.get_optimal_cluster_num(embeddings=embeddings)
            
            result_dict,all_nodes = cl_module.clustering(
                sentence_name_db_data=sentence_data,
                image_db_data=cl_module.image_db.get_data_by_ids(target_image_ids),
                clustering_id_dict=cid_dict,
                sentence_id_dict=sid_dict,  # 元の形式に戻す
                image_id_dict=iid_dict,
                cluster_num=cluster_num,
                overall_folder_name=project_name,
                output_folder=True,
                output_json=True
            )
            
            # all_nodesを配列から辞書形式に変換（idをキーとして）
            all_nodes_dict = {}
            for node in all_nodes:
                if 'id' in node:
                    all_nodes_dict[node['id']] = node
            
            from clustering.mongo_result_manager import ResultManager
            result_manager = ResultManager(mongo_result_id)
            result_manager.update_result(result_dict, all_nodes_dict)
        except Exception as e:
            print(f"Error during clustering:{e}")
            
            # エラーが発生した場合は初期化状態を更新
            clustering_state = INIT_CLUSTERING_STATUS.FAILED
        else:
            clustering_state = INIT_CLUSTERING_STATUS.FINISHED
            
            # 初期クラスタリング成功時、該当ユーザの全画像をクラスタリング済みとしてマーク
            try:
                mark_clustered_query = f"""
                    UPDATE user_image_clustering_states
                    SET is_clustered = 1, executed_clustering_count = 0, clustered_at = CURRENT_TIMESTAMP(6)
                    WHERE user_id = {user_id} AND project_id = {project_id} AND is_clustered = 0;
                """
                _, _ = execute_query(session=connect_session, query_text=mark_clustered_query)
                print(f"✅ ユーザ{user_id}のプロジェクト{project_id}内の全画像をクラスタリング済み(executed_clustering_count=0)としてマークしました")
            except Exception as mark_error:
                print(f"⚠️ user_image_clustering_states更新エラー: {mark_error}")
        finally:
            
            # 初期化状態を更新
            update_query = f"""
                UPDATE project_memberships
                SET init_clustering_state = '{clustering_state}'
                WHERE project_id = {project_id} AND user_id = {user_id};
            """
            _, _ = execute_query(session=connect_session, query_text=update_query)
                
    # 非同期実行
    background_tasks.add_task(run_clustering, by_clustering_id, by_chromadb_sentence_id, by_chromadb_image_id, project_id, original_images_folder_path)
    
    # 初期化状態を更新
    update_query = f"""
        UPDATE project_memberships
        SET init_clustering_state = '{INIT_CLUSTERING_STATUS.EXECUTING}'
        WHERE project_id = {project_id} AND user_id = {user_id};
    """
    #初期化状態を更新
    _, _ = execute_query(session=connect_session, query_text=update_query)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "init clustering started in background", "data": project_id}
    )


@action_endpoint.get("/action/clustering/continuous/{project_id}", tags=["action"], description="継続的クラスタリングを実装する")
def execute_continuous_clustering(
    project_id: int = None,
    user_id: int = None,
    background_tasks: BackgroundTasks = None
):
    """
    継続的クラスタリング: 新規追加された未クラスタリング画像を既存の階層に追加する
    
    Args:
        project_id: プロジェクトID
        user_id: ユーザーID
        background_tasks: バックグラウンドタスク
        
    Returns:
        JSONResponse: 継続的クラスタリング開始結果
    """
    if project_id is None or user_id is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "project_id and user_id is required", "data": None}
        )

    try:
        project_id = int(project_id)
        user_id = int(user_id)
    except:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "project_id or user_id is invalid", "data": None}
        )

    connect_session = create_connect_session()
    
    # プロジェクトメンバーシップ情報を取得
    query_text = f"""
        SELECT 
            project_memberships.init_clustering_state,
            project_memberships.continuous_clustering_state,
            project_memberships.mongo_result_id,
            projects.original_images_folder_path
        FROM project_memberships
        JOIN projects ON project_memberships.project_id = projects.id
        WHERE project_memberships.project_id = {project_id} AND project_memberships.user_id = {user_id};
    """

    result, _ = execute_query(session=connect_session, query_text=query_text)
    result_mappings = result.mappings().first()

    if result_mappings is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "project or membership not found", "data": None}
        )

    init_clustering_state = result_mappings["init_clustering_state"]
    continuous_clustering_state = result_mappings["continuous_clustering_state"]
    mongo_result_id = result_mappings["mongo_result_id"]
    original_images_folder_path = result_mappings["original_images_folder_path"]

    # 初期クラスタリングが完了していない場合はエラー
    if init_clustering_state != INIT_CLUSTERING_STATUS.FINISHED:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "init clustering not completed yet", "data": None}
        )

    # 継続的クラスタリングが実行可能でない場合はエラー
    if continuous_clustering_state != CONTINUOUS_CLUSTERING_STATUS.EXECUTABLE:  # 2 = 実行可能
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "continuous clustering is not executable", "data": None}
        )

    # 未クラスタリング画像の取得（画像の詳細情報も含める）
    unclustered_images_query = f"""
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

    result, _ = execute_query(session=connect_session, query_text=unclustered_images_query)
    
    if result is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to get unclustered images", "data": None}
        )

    rows = result.mappings().all()
    
    for row in rows:
        print("row",row)
    
    if len(rows) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "no unclustered images found", "data": None}
        )

    # ユーザー情報を取得
    user_info_query = f"""
        SELECT id, name, email FROM users WHERE id = {user_id};
    """
    user_result, _ = execute_query(session=connect_session, query_text=user_info_query)
    user_info = user_result.mappings().first() if user_result else None

    # コンソールに詳細情報を出力
    print("=" * 80)
    print("=" * 80)
    print(f"\n📋 対象ユーザー情報:")
    print(f"  - ユーザーID: {user_id}")
    if user_info:
        print(f"  - ユーザー名: {user_info['name']}")
        print(f"  - メールアドレス: {user_info['email']}")
    for idx, row in enumerate(rows, 1):
        print(f"\n  [{idx}] 画像情報:")
        print(f"      - 画像ID: {row['image_id']}")
        print(f"      - 画像名: {row['image_name']}")
    print("\n" + "=" * 80)

    # バックグラウンド処理に渡す関数
    def run_continuous_clustering(unclustered_rows: list, project_id: int, user_id: int, mongo_result_id: str):
        try:
            print(f"\n🔄 継続的クラスタリング バックグラウンド処理開始")
            print(f"   プロジェクトID: {project_id}")
            print(f"   ユーザーID: {user_id}")
            print(f"   未クラスタリング画像数: {len(unclustered_rows)}")
            
            from clustering.mongo_result_manager import ResultManager
            from clustering.chroma_db_manager import ChromaDBManager
            from clustering.embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # ResultManagerとChromaDBManagerを初期化
            result_manager = ResultManager(mongo_result_id)
            image_db = ChromaDBManager("image_embeddings")
            
            # 現在のexecuted_clustering_countを取得して+1
            get_count_query = f"""
                SELECT executed_clustering_count FROM project_memberships
                WHERE user_id = {user_id} AND project_id = {project_id};
            """
            count_result, _ = execute_query(session=connect_session, query_text=get_count_query)
            current_count = count_result.mappings().first()['executed_clustering_count']
            new_count = current_count + 1
            
            print(f"📊 現在のクラスタリング回数: {current_count} → 新しい回数: {new_count}")
            
            # すべてのリーフフォルダを取得
            leaf_folders = result_manager.get_all_leaf_folders()
            print(f"📂 リーフフォルダ数: {len(leaf_folders)}")
            
            if len(leaf_folders) == 0:
                print("❌ リーフフォルダが見つかりません")
                return
            
            # 各リーフフォルダの画像埋め込みベクトルの平均を計算
            folder_embeddings = {}
            for folder in leaf_folders:
                folder_id = folder['id']
                
                # result内でフォルダIDを探索してdataを取得
                folder_data_result = result_manager.get_folder_data_from_result(folder_id)
                
                if not folder_data_result['success']:
                    print(f"  ⚠️ フォルダ {folder_id} ({folder['name']}) のデータ取得失敗: {folder_data_result.get('error', 'Unknown error')}")
                    continue
                
                # フォルダ内の画像のclustering_idを取得
                folder_data = folder_data_result['data']
                if not isinstance(folder_data, dict) or len(folder_data) == 0:
                    print(f"  ⚠️ フォルダ {folder_id} ({folder['name']}) は空です")
                    continue
                
                clustering_ids = list(folder_data.keys())
                print(f"  📁 フォルダ {folder['name']} ({folder_id}): {len(clustering_ids)}個の画像を含む")
                
                # clustering_idからchromadb_image_idを取得
                image_ids = []
                for cid in clustering_ids:
                    get_image_id_query = f"""
                        SELECT chromadb_image_id FROM images
                        WHERE clustering_id = '{cid}' AND project_id = {project_id};
                    """
                    img_result, _ = execute_query(session=connect_session, query_text=get_image_id_query)
                    if img_result:
                        img_mapping = img_result.mappings().first()
                        if img_mapping:
                            image_ids.append(img_mapping['chromadb_image_id'])
                
                if len(image_ids) == 0:
                    continue
                
                # ChromaDBから画像の埋め込みベクトルを取得
                try:
                    image_data = image_db.get_data_by_ids(image_ids)
                    embeddings = image_data['embeddings']
                    
                    # 平均埋め込みベクトルを計算
                    avg_embedding = np.mean(embeddings, axis=0)
                    folder_embeddings[folder_id] = avg_embedding
                    
                    print(f"  ✅ フォルダ {folder['name']} ({folder_id}): {len(embeddings)}個の画像の平均ベクトル計算完了")
                except Exception as e:
                    print(f"  ⚠️ フォルダ {folder_id} の埋め込みベクトル取得エラー: {e}")
                    continue
            
            print(f"\n📊 埋め込みベクトルを持つフォルダ数: {len(folder_embeddings)}")
            
            # 各未クラスタリング画像を処理
            for idx, row in enumerate(unclustered_rows, 1):
                try:
                    image_id = row['image_id']
                    image_name = row['image_name']
                    clustering_id = row['clustering_id']
                    chromadb_image_id = row['chromadb_image_id']
                    
                    print(f"\n  [{idx}/{len(unclustered_rows)}] 処理中: {image_name} (ID: {image_id})")
                    
                    # ChromaDBから画像の埋め込みベクトルを取得
                    try:
                        new_image_data = image_db.get_data_by_ids([chromadb_image_id])
                        new_image_embedding = new_image_data['embeddings'][0]
                    except Exception as e:
                        print(f"    ⚠️ 画像埋め込みベクトル取得エラー: {e}")
                        continue
                    
                    # 各フォルダとの類似度を計算
                    max_similarity = -1
                    best_folder_id = None
                    
                    for folder_id, folder_embedding in folder_embeddings.items():
                        similarity = cosine_similarity(
                            [new_image_embedding],
                            [folder_embedding]
                        )[0][0]
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_folder_id = folder_id
                    
                    if best_folder_id is None:
                        print(f"    ⚠️ 適切なフォルダが見つかりませんでした")
                        continue
                    
                    best_folder = next((f for f in leaf_folders if f['id'] == best_folder_id), None)
                    folder_name = best_folder['name'] if best_folder else best_folder_id
                    
                    print(f"    🎯 最も類似したフォルダ: {folder_name} (類似度: {max_similarity:.4f})")
                    
                    # 画像をフォルダに挿入
                    # imagesテーブルからimage_pathを取得
                    get_path_query = f"""
                        SELECT name FROM images WHERE id = {image_id};
                    """
                    path_result, _ = execute_query(session=connect_session, query_text=get_path_query)
                    image_path = path_result.mappings().first()['name']
                    
                    insert_result = result_manager.insert_image_to_leaf_folder(
                        clustering_id=clustering_id,
                        image_path=image_path,
                        target_folder_id=best_folder_id
                    )
                    
                    if insert_result['success']:
                        print(f"    ✅ フォルダに画像を挿入しました")
                        
                        # user_image_clustering_statesを更新
                        update_state_query = f"""
                            UPDATE user_image_clustering_states
                            SET is_clustered = 1, 
                                executed_clustering_count = {new_count}, 
                                clustered_at = CURRENT_TIMESTAMP(6)
                            WHERE user_id = {user_id} AND image_id = {image_id};
                        """
                        _, _ = execute_query(session=connect_session, query_text=update_state_query)
                        
                        # フォルダの埋め込みベクトルを再計算（新しい画像を追加したため）
                        print(f"    🔄 フォルダ埋め込みベクトルを再計算中...")
                        try:
                            # result内でフォルダIDを探索してdataを取得
                            folder_data_result = result_manager.get_folder_data_from_result(best_folder_id)
                            if folder_data_result['success']:
                                folder_data = folder_data_result['data']
                                clustering_ids = list(folder_data.keys())
                                
                                image_ids = []
                                for cid in clustering_ids:
                                    get_image_id_query = f"""
                                        SELECT chromadb_image_id FROM images
                                        WHERE clustering_id = '{cid}' AND project_id = {project_id};
                                    """
                                    img_result, _ = execute_query(session=connect_session, query_text=get_image_id_query)
                                    if img_result:
                                        img_mapping = img_result.mappings().first()
                                        if img_mapping:
                                            image_ids.append(img_mapping['chromadb_image_id'])
                                
                                if len(image_ids) > 0:
                                    updated_image_data = image_db.get_data_by_ids(image_ids)
                                    updated_embeddings = updated_image_data['embeddings']
                                    folder_embeddings[best_folder_id] = np.mean(updated_embeddings, axis=0)
                                    print(f"    ✅ フォルダ埋め込みベクトル再計算完了 ({len(image_ids)}個の画像)")
                                else:
                                    print(f"    ⚠️ フォルダに画像IDが見つかりません")
                            else:
                                print(f"    ⚠️ フォルダデータの再取得失敗: {folder_data_result.get('error', 'Unknown error')}")
                        except Exception as e:
                            print(f"    ⚠️ フォルダ埋め込みベクトル再計算エラー: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"    ❌ 画像挿入エラー: {insert_result.get('error', 'Unknown error')}")
                        
                except Exception as img_error:
                    print(f"    ❌ 画像処理中にエラー: {img_error}")
                    continue
            
            # project_membershipsのexecuted_clustering_countを更新
            update_count_query = f"""
                UPDATE project_memberships
                SET executed_clustering_count = {new_count}
                WHERE user_id = {user_id} AND project_id = {project_id};
            """
            _, _ = execute_query(session=connect_session, query_text=update_count_query)
            
            # 未クラスタリング画像が残っているか確認
            check_unclustered_query = f"""
                SELECT COUNT(*) as unclustered_count
                FROM images i
                LEFT JOIN user_image_clustering_states uics 
                    ON i.id = uics.image_id AND uics.user_id = {user_id}
                WHERE i.project_id = {project_id} 
                    AND i.is_created_caption = TRUE
                    AND (uics.is_clustered = 0 OR uics.is_clustered IS NULL);
            """
            check_result, _ = execute_query(session=connect_session, query_text=check_unclustered_query)
            remaining_unclustered = check_result.mappings().first()['unclustered_count']
            
            print(f"\n📊 クラスタリング完了後の状態確認:")
            print(f"   残りの未クラスタリング画像数: {remaining_unclustered}")
            
            # 未クラスタリング画像が残っていれば2（実行可能）、なければ0（実行不可能）
            new_state = 2 if remaining_unclustered > 0 else 0
            state_description = "実行可能" if new_state == 2 else "実行不可能"
            
            update_state_query = f"""
                UPDATE project_memberships
                SET continuous_clustering_state = {new_state}
                WHERE user_id = {user_id} AND project_id = {project_id};
            """
            _, _ = execute_query(session=connect_session, query_text=update_state_query)
            
            print(f"   continuous_clustering_state: {new_state} ({state_description})")
            print(f"\n✅ 継続的クラスタリング バックグラウンド処理完了")
            print(f"   処理した画像数: {len(unclustered_rows)}")
            print(f"   新しいクラスタリング回数: {new_count}")
            
        except Exception as e:
            print(f"❌ 継続的クラスタリング処理中にエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # エラー時も未クラスタリング画像の有無を確認して状態を設定
            try:
                check_unclustered_query = f"""
                    SELECT COUNT(*) as unclustered_count
                    FROM images i
                    LEFT JOIN user_image_clustering_states uics 
                        ON i.id = uics.image_id AND uics.user_id = {user_id}
                    WHERE i.project_id = {project_id} 
                        AND i.is_created_caption = TRUE
                        AND (uics.is_clustered = 0 OR uics.is_clustered IS NULL);
                """
                check_result, _ = execute_query(session=connect_session, query_text=check_unclustered_query)
                remaining_unclustered = check_result.mappings().first()['unclustered_count']
                
                # 未クラスタリング画像が残っていれば2（実行可能）、なければ0（実行不可能）
                new_state = 2 if remaining_unclustered > 0 else 0
                
                update_state_query = f"""
                    UPDATE project_memberships
                    SET continuous_clustering_state = {new_state}
                    WHERE user_id = {user_id} AND project_id = {project_id};
                """
                _, _ = execute_query(session=connect_session, query_text=update_state_query)
                print(f"⚠️ エラー後の状態更新: continuous_clustering_state = {new_state} (未クラスタリング画像: {remaining_unclustered})")
            except Exception as state_error:
                print(f"⚠️ エラー後の状態更新に失敗: {state_error}")
                
    # continuous_clustering_stateを1（実行中）に更新
    update_query = f"""
        UPDATE project_memberships
        SET continuous_clustering_state = 1
        WHERE project_id = {project_id} AND user_id = {user_id};
    """
    _, _ = execute_query(session=connect_session, query_text=update_query)
    
    # 非同期実行
    background_tasks.add_task(run_continuous_clustering, rows, project_id, user_id, mongo_result_id)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "continuous clustering started in background",
            "data": {
                "project_id": project_id,
                "user_id": user_id,
                "unclustered_image_count": len(rows)
            }
        }
    )


@action_endpoint.put("/action/clustering/move/{mongo_result_id}", tags=["action"], description="クラスタリング結果のフォルダ構造を変更する")
def move_clustering_items(
    mongo_result_id: str,
    source_type: str = Query(..., description="移動するソースのタイプ: 'folders' または 'images'"),
    sources: List[str] = Query(..., description="移動する要素の名前の配列"),
    destination_folder: str = Query(..., description="移動先のフォルダ名")
):
    """
    指定したmongo_result_idに紐ついたjsonを編集してフォルダ構造を作り替えてnosqlデータベースにコミットする
    
    Args:
        mongo_result_id: MongoDBの結果ID
        source_type: 移動するソースのタイプ ("folders" または "images")
        sources: 移動する要素の名前の配列
        destination_folder: 移動先のフォルダ名
    """
    
    # パラメータの検証
    if source_type not in ["folders", "images"]:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "source_type must be 'folders' or 'images'", "data": None}
        )
    
    if not sources or len(sources) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "sources must not be empty", "data": None}
        )
    
    if not destination_folder:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "destination_folder is required", "data": None}
        )

    result_manager = ResultManager(mongo_result_id)
    
    if source_type == "folders":
        try:
            result_manager.move_folder_node(sources, destination_folder)
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"フォルダ移動に失敗しました: {str(e)}", "data": None}
            )
            
    elif source_type == "images":
        try:
            for source in sources:
                result_manager.move_file_node(source, destination_folder)
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": f"ファイル移動に失敗しました: {str(e)}", "data": None}
            )
            
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "source_type must be 'folders' or 'images'", "data": None}
        )
    


    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "success", "data": None}
    )
    

@action_endpoint.delete("/action/folders/{mongo_result_id}", tags=["action"], description="指定されたフォルダを削除")
async def delete_folders(mongo_result_id: str, sources: List[str] = Query(...)):
    """
    指定されたフォルダIDリストを受け取り、結果から削除する
    
    Args:
        mongo_result_id (str): MongoDBの結果ID
        sources (List[str]): 削除対象のフォルダIDリスト
    
    Returns:
        JSONResponse: 削除処理の結果
    """
    try:
        # 入力バリデーション
        if not mongo_result_id or not mongo_result_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "mongo_result_id is required"}
            )
        
        if not sources or len(sources) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "sources parameter is required and must contain at least one folder ID"}
            )
        
        print(f"🗂️ delete_folders呼び出し: mongo_result_id={mongo_result_id}")
        print(f"📋 受け取ったフォルダIDリスト (sources): {sources}")
        print(f"📊 削除対象フォルダ数: {len(sources)}")
        
        # ResultManagerを初期化
        result_manager = ResultManager(mongo_result_id)
        
        # フォルダを結果から削除
        is_success = result_manager.remove_folders_from_result(sources)
        
        if is_success:
            print(f"✅ フォルダ削除成功: {len(sources)}個のフォルダを削除しました")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "success", 
                    "data": {
                        "deleted_folder_count": len(sources),
                        "deleted_folders": sources
                    }
                }
            )
        else:
            print(f"❌ フォルダ削除失敗: remove_folders_from_result returned False")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": "Failed to delete folders from result",
                    "data": {
                        "mongo_result_id": mongo_result_id,
                        "attempted_folder_ids": sources,
                        "error": "remove_folders_from_result operation failed"
                    }
                }
            )
            
    except Exception as e:
        print(f"❌ delete_folders処理中にエラーが発生: {str(e)}")
        return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": "Internal server error occurred during folder deletion",
                    "data": {
                        "mongo_result_id": mongo_result_id,
                        "attempted_folder_ids": sources,
                        "error": str(e)
                    }
                }
            )


@action_endpoint.get("/action/clustering/node/{mongo_result_id}/{node_id}", tags=["action"], description="指定されたノードの情報を取得する")
async def get_node_info(mongo_result_id: str, node_id: str):
    """
    指定されたnode_idのノード情報をall_nodesから取得する
    
    Args:
        mongo_result_id (str): MongoDBの結果ID（パスパラメータ）
        node_id (str): ノードID（パスパラメータ）
        
    Returns:
        JSONResponse: ノード情報
    """
    try:
        # 入力バリデーション
        if not mongo_result_id or not mongo_result_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "mongo_result_id is required"}
            )
        
        if not node_id or not node_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "node_id is required"}
            )
        
        print(f"🔍 get_node_info呼び出し: mongo_result_id={mongo_result_id}, node_id={node_id}")
        
        # ResultManagerを初期化してノード情報を取得
        result_manager = ResultManager(mongo_result_id)
        node_data = result_manager.get_node_info(node_id=node_id)
        
        # エラーの場合はHTTPExceptionを発生
        if not node_data["success"]:
            if "not found" in node_data.get("error", "").lower():
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "message": node_data["error"],
                        "data": {
                            "mongo_result_id": mongo_result_id,
                            "node_id": node_id
                        }
                    }
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "message": node_data["error"],
                        "data": {
                            "mongo_result_id": mongo_result_id,
                            "node_id": node_id
                        }
                    }
                )
        
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "success",
                "data": node_data["data"]  # all_nodesの値のみを返す
            }
        )
        
    except Exception as e:
        print(f"❌ get_node_info処理中にエラー: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Internal server error occurred during node retrieval",
                "data": {
                    "mongo_result_id": mongo_result_id,
                    "node_id": node_id,
                    "error": str(e)
                }
            }
        )


@action_endpoint.put("/action/folders/{mongo_result_id}/{node_id}", tags=["action"], description="フォルダまたはファイルの名前を変更")
async def rename_folder_or_file(
    mongo_result_id: str,
    node_id: str,
    name: str = Query(None, description="新しい名前"),
    is_leaf: bool = Query(None, description="リーフノード（ファイル）かどうか")
):
    """
    指定されたノードの名前を変更する
    
    Args:
        mongo_result_id (str): MongoDBの結果ID
        node_id (str): 変更対象のノードID
        name (str, optional): 新しい名前
        is_leaf (bool, optional): リーフノード（ファイル）かどうか
    
    Returns:
        JSONResponse: 名前変更処理の結果
    """
    try:
        # 入力バリデーション
        if not mongo_result_id or not mongo_result_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "mongo_result_id is required"}
            )
        
        if not node_id or not node_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "node_id is required"}
            )
        
        # nameとis_leafの両方がNoneの場合はエラー
        if name is None and is_leaf is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "At least one of 'name' or 'is_leaf' parameters is required"}
            )
        
        # nameが指定されている場合は空文字チェック
        if name is not None and not name.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "name parameter must not be empty when provided"}
            )
        
        print(f"🏷️ rename_folder_or_file呼び出し: mongo_result_id={mongo_result_id}, node_id={node_id}")
        print(f"📝 パラメータ: name={name}, is_leaf={is_leaf}")
        
        # ResultManagerを初期化
        result_manager = ResultManager(mongo_result_id)
        
        # 名前・is_leaf変更処理
        update_result = result_manager.rename_node(
            node_id=node_id, 
            new_name=name.strip() if name is not None else None, 
            is_leaf=is_leaf
        )
        
        print(f"✅ 更新結果: {update_result}")
        
        if update_result.get("success", False):
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "success",
                    "data": {
                        "node_id": node_id,
                        "updated_fields": update_result.get("updated_fields", {}),
                        "is_leaf": is_leaf
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": "Failed to update node",
                    "data": {
                        "mongo_result_id": mongo_result_id,
                        "node_id": node_id,
                        "attempted_name": name,
                        "attempted_is_leaf": is_leaf,
                        "error": update_result.get("error", "Unknown error")
                    }
                }
            )
            
    except Exception as e:
        print(f"❌ rename_folder_or_file処理中にエラーが発生: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Internal server error occurred during rename operation",
                "data": {
                    "mongo_result_id": mongo_result_id,
                    "node_id": node_id,
                    "attempted_name": name,
                    "error": str(e)
                }
            }
        )