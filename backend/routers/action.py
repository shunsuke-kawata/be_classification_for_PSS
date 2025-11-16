import copy
import json
import re
import traceback
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, status, Response, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from db_utils.commons import create_connect_session, execute_query
from db_utils import action_queries, images_queries
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, LoginUser, JoinUser
from config import (
    INIT_CLUSTERING_STATUS,
    CONTINUOUS_CLUSTERING_STATUS,
    DEFAULT_IMAGE_PATH,
    DEFAULT_OUTPUT_PATH,
    CAPTION_STOPWORDS,
    MAJOR_COLORS,
    MAJOR_SHAPES
)
from clustering.clustering_manager import ChromaDBManager, InitClusteringManager
from clustering.mongo_db_manager import MongoDBManager
from clustering.mongo_result_manager import ResultManager
from clustering.chroma_db_manager import ChromaDBManager
from clustering.embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from clustering.utils import Utils
from clustering.word_analysis import WordAnalyzer

#分割したエンドポイントの作成
#ログイン操作
action_endpoint = APIRouter()

@action_endpoint.get("/action/clustering/result/{mongo_result_id}",tags=["action"],description="初期クラスタリング結果を取得する")
def get_clustering_result(mongo_result_id:str):
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
        source_result, _ = action_queries.get_membership_init_and_mongo(connect_session, source_user_id, project_id)
        
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
        target_result, _ = action_queries.get_membership_init_and_mongo(connect_session, target_user_id, project_id)
        
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
        _, _ = action_queries.update_init_state(connect_session, target_user_id, project_id, INIT_CLUSTERING_STATUS.FINISHED)
        
        # 6. コピー先ユーザーの全画像をクラスタリング済みとしてマーク
        _, _ = action_queries.mark_user_images_clustered(connect_session, target_user_id, project_id)
        
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
    result, _ = action_queries.get_membership_and_project_info(connect_session, project_id, user_id)
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
    result, _ = action_queries.select_images_for_init(connect_session, project_id)
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
            # プロジェクト名を取得
            project_result, _ = action_queries.get_project_name(connect_session, project_id)
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
                _, _ = action_queries.mark_user_images_clustered_with_executed_count(connect_session, user_id, project_id, 0)
                print(f"✅ ユーザ{user_id}のプロジェクト{project_id}内の全画像をクラスタリング済み(executed_clustering_count=0)としてマークしました")
            except Exception as mark_error:
                print(f"⚠️ user_image_clustering_states更新エラー: {mark_error}")
        finally:
            
            # 初期化状態を更新
            _, _ = action_queries.update_init_state(connect_session, user_id, project_id, clustering_state)
                
    # 非同期実行
    background_tasks.add_task(run_clustering, by_clustering_id, by_chromadb_sentence_id, by_chromadb_image_id, project_id, original_images_folder_path)
    
    # 初期化状態を更新
    # 初期化状態を更新
    _, _ = action_queries.update_init_state(connect_session, user_id, project_id, INIT_CLUSTERING_STATUS.EXECUTING)
    
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
    result, _ = action_queries.get_membership_and_project_info(connect_session, project_id, user_id)
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

    # デバッグ情報を出力
    print(f"\n🔍 継続的クラスタリング状態チェック:")
    print(f"   init_clustering_state: {init_clustering_state} (期待値: {INIT_CLUSTERING_STATUS.FINISHED})")
    print(f"   continuous_clustering_state: {continuous_clustering_state} (期待値: {CONTINUOUS_CLUSTERING_STATUS.EXECUTABLE})")
    print(f"   mongo_result_id: {mongo_result_id}")
    
    # 初期クラスタリングが完了していない場合はエラー
    if init_clustering_state != INIT_CLUSTERING_STATUS.FINISHED:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "message": "init clustering not completed yet", 
                "data": {
                    "current_init_state": init_clustering_state,
                    "required_init_state": INIT_CLUSTERING_STATUS.FINISHED
                }
            }
        )

    # 継続的クラスタリングが実行可能でない場合はエラー
    if continuous_clustering_state != CONTINUOUS_CLUSTERING_STATUS.EXECUTABLE:  # 2 = 実行可能
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "message": "continuous clustering is not executable", 
                "data": {
                    "current_continuous_state": continuous_clustering_state,
                    "required_continuous_state": CONTINUOUS_CLUSTERING_STATUS.EXECUTABLE
                }
            }
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

    result, _ = action_queries.get_unclustered_images(connect_session, project_id, user_id)
    
    if result is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to get unclustered images", "data": None}
        )

    rows = result.mappings().all()
    
    print(f"\n📊 未クラスタリング画像の取得結果:")
    print(f"   取得した画像数: {len(rows)}")
    for row in rows:
        print(f"   - 画像ID: {row['image_id']}, 名前: {row['image_name']}")
    
    if len(rows) == 0:
        print(f"⚠️ 未クラスタリング画像が見つかりませんでした")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "message": "no unclustered images found", 
                "data": {
                    "project_id": project_id,
                    "user_id": user_id,
                    "unclustered_count": 0
                }
            }
        )

    # ユーザー情報を取得
    user_result, _ = action_queries.get_user_info(connect_session, user_id)
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
            
            # ResultManagerとChromaDBManagerを初期化
            result_manager = ResultManager(mongo_result_id)
            # 文章埋め込みベクトルと画像埋め込みベクトルの両方を使用
            sentence_name_db = ChromaDBManager("sentence_name_embeddings")
            image_db = ChromaDBManager("image_embeddings")
            
            # 現在のexecuted_clustering_countを取得して+1
            count_result, _ = action_queries.get_executed_clustering_count(connect_session, user_id, project_id)
            current_count = count_result.mappings().first()['executed_clustering_count']
            new_count = current_count + 1
            
            print(f"📊 現在のクラスタリング回数: {current_count} → 新しい回数: {new_count}")
            
            # すべてのリーフフォルダを取得
            leaf_folders = result_manager.get_all_leaf_folders()
            print(f"📂 リーフフォルダ数: {len(leaf_folders)}")
            
            if len(leaf_folders) == 0:
                print("❌ リーフフォルダが見つかりません")
                return
            
            # 各リーフフォルダの文章埋め込みベクトルと画像埋め込みベクトルの平均を計算
            folder_sentence_embeddings = {}
            folder_image_embeddings = {}
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
                
                # clustering_idからchromadb_sentence_idとchromadb_image_idを取得
                sentence_ids = []
                image_ids = []
                for cid in clustering_ids:
                    sent_result, _ = action_queries.get_chromadb_sentence_id_by_clustering_id(connect_session, cid, project_id)
                    if sent_result:
                        sent_mapping = sent_result.mappings().first()
                        if sent_mapping:
                            sentence_ids.append(sent_mapping['chromadb_sentence_id'])
                    
                    img_result, _ = action_queries.get_chromadb_image_id_by_clustering_id(connect_session, cid, project_id)
                    if img_result:
                        img_mapping = img_result.mappings().first()
                        if img_mapping:
                            image_ids.append(img_mapping['chromadb_image_id'])

                # ChromaDBから文章の埋め込みベクトルを取得
                if len(sentence_ids) > 0:
                    try:
                        sentence_data = sentence_name_db.get_data_by_ids(sentence_ids)
                        sentence_embeddings = sentence_data['embeddings']
                        avg_sentence_embedding = np.mean(sentence_embeddings, axis=0)
                        folder_sentence_embeddings[folder_id] = avg_sentence_embedding
                        print(f"  ✅ フォルダ {folder['name']} ({folder_id}): {len(sentence_embeddings)}個の文章の平均ベクトル計算完了")
                    except Exception as e:
                        print(f"  ⚠️ フォルダ {folder_id} の文章埋め込みベクトル取得エラー: {e}")
                
                # ChromaDBから画像の埋め込みベクトルを取得
                if len(image_ids) > 0:
                    try:
                        image_data = image_db.get_data_by_ids(image_ids)
                        image_embeddings = image_data['embeddings']
                        avg_image_embedding = np.mean(image_embeddings, axis=0)
                        folder_image_embeddings[folder_id] = avg_image_embedding
                        print(f"  ✅ フォルダ {folder['name']} ({folder_id}): {len(image_embeddings)}個の画像の平均ベクトル計算完了")
                    except Exception as e:
                        print(f"  ⚠️ フォルダ {folder_id} の画像埋め込みベクトル取得エラー: {e}")
            
            print(f"\n📊 文章埋め込みベクトルを持つフォルダ数: {len(folder_sentence_embeddings)}")
            print(f"📊 画像埋め込みベクトルを持つフォルダ数: {len(folder_image_embeddings)}")
            
            # 各未クラスタリング画像を処理
            for idx, row in enumerate(unclustered_rows, 1):
                try:
                    image_id = row['image_id']
                    image_name = row['image_name']
                    clustering_id = row['clustering_id']
                    chromadb_sentence_id = row['chromadb_sentence_id']
                    chromadb_image_id = row['chromadb_image_id']
                    
                    print(f"\n  [{idx}/{len(unclustered_rows)}] 処理中: {image_name} (ID: {image_id})")
                    
                    # ChromaDBから文章の埋め込みベクトルを取得
                    new_sentence_embedding = None
                    try:
                        new_sentence_data = sentence_name_db.get_data_by_ids([chromadb_sentence_id])
                        new_sentence_embedding = new_sentence_data['embeddings'][0]
                    except Exception as e:
                        print(f"    ⚠️ 文章埋め込みベクトル取得エラー: {e}")
                    
                    # ChromaDBから画像の埋め込みベクトルを取得
                    new_image_embedding = None
                    try:
                        new_image_data = image_db.get_data_by_ids([chromadb_image_id])
                        new_image_embedding = new_image_data['embeddings'][0]
                    except Exception as e:
                        print(f"    ⚠️ 画像埋め込みベクトル取得エラー: {e}")
                    
                    # 両方のベクトルが取得できなかった場合はスキップ
                    if new_sentence_embedding is None and new_image_embedding is None:
                        print(f"    ⚠️ 埋め込みベクトルの取得に失敗しました")
                        continue
                    
                    # 各フォルダとの類似度を計算（文章と画像の両方）
                    max_similarity = -1
                    best_folder_id = None
                    best_similarity_type = None  # 'sentence' or 'image'
                    
                    # 文章ベクトルで類似度計算
                    if new_sentence_embedding is not None:
                        for folder_id, folder_embedding in folder_sentence_embeddings.items():
                            similarity = cosine_similarity(
                                [new_sentence_embedding],
                                [folder_embedding]
                            )[0][0]
                            
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_folder_id = folder_id
                                best_similarity_type = 'sentence'
                    
                    # 画像ベクトルで類似度計算
                    if new_image_embedding is not None:
                        for folder_id, folder_embedding in folder_image_embeddings.items():
                            similarity = cosine_similarity(
                                [new_image_embedding],
                                [folder_embedding]
                            )[0][0]
                            
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_folder_id = folder_id
                                best_similarity_type = 'image'
                    
                    if best_folder_id is None:
                        print(f"    ⚠️ 適切なフォルダが見つかりませんでした")
                        continue
                    
                    print(f"    📊 最高類似度: {max_similarity:.4f} (タイプ: {best_similarity_type})")
                    
                    # 類似度閾値チェック：閾値を下回る場合は新しいフォルダを作成
                    SIMILARITY_THRESHOLD = 0.4  # 類似度閾値（調整可能）
                    
                    if max_similarity < SIMILARITY_THRESHOLD:
                        print(f"    ⚠️ 最高類似度 {max_similarity:.4f} が閾値 {SIMILARITY_THRESHOLD} を下回っています")
                        print(f"    🆕 新しいリーフフォルダを作成します...")
                        
                        # キャプションから新フォルダ名を生成
                        try:
                            caption_res, _ = images_queries.select_caption_by_clustering_id(connect_session, clustering_id)
                            if caption_res:
                                caption_row = caption_res.mappings().first()
                                if caption_row and 'caption' in caption_row and caption_row['caption']:
                                    caption = caption_row['caption']
                                    # キャプションから特徴的な単語を抽出してフォルダ名を生成
                                    # 最初の文節（.の前）から単語を抽出
                                    first_sentence = caption.split('.')[0] if '.' in caption else caption
                                    # 2-3個の特徴的な単語を抽出（ストップワード除外）

                                    words = WordAnalyzer.extract_words(first_sentence)
                                    # 最大3単語でフォルダ名を作成
                                    new_folder_name = ','.join(words[:3]) if len(words) > 0 else f"new_category_{idx}"
                                else:
                                    new_folder_name = f"new_category_{idx}"
                            else:
                                new_folder_name = f"new_category_{idx}"
                        except Exception as name_e:
                            print(f"    ⚠️ フォルダ名生成エラー: {name_e}")
                            new_folder_name = f"new_category_{idx}"
                        
                        # imagesテーブルからimage_pathを取得
                        path_result, _ = action_queries.get_image_name_by_id(connect_session, image_id)
                        image_path = path_result.mappings().first()['name']
                        
                        # トップレベル（parent_id=None）に新しいリーフフォルダを作成
                        create_result = result_manager.create_new_leaf_folder(
                            folder_name=new_folder_name,
                            parent_id=None,  # トップレベルに作成
                            initial_clustering_id=clustering_id,
                            initial_image_path=image_path
                        )
                        
                        if create_result['success']:
                            new_folder_id = create_result['folder_id']
                            print(f"    ✅ 新しいフォルダを作成しました: {new_folder_name} (ID: {new_folder_id})")
                            
                            # user_image_clustering_statesを更新
                            _, _ = action_queries.update_user_image_state_for_image(connect_session, user_id, image_id, new_count)
                            
                            # 新しいフォルダの埋め込みベクトルを追加（両方）
                            if new_sentence_embedding is not None:
                                folder_sentence_embeddings[new_folder_id] = new_sentence_embedding
                            if new_image_embedding is not None:
                                folder_image_embeddings[new_folder_id] = new_image_embedding
                            
                            # leaf_foldersリストにも追加
                            leaf_folders.append({
                                'id': new_folder_id,
                                'name': new_folder_name,
                                'parent_id': None,
                                'is_leaf': True
                            })
                            
                            print(f"    ℹ️ 類似度が低いため、後続のフォルダ特徴分析はスキップします")
                            continue  # 後続の処理をスキップ
                        else:
                            print(f"    ❌ フォルダ作成エラー: {create_result.get('error', 'Unknown error')}")
                            print(f"    → 既存のフォルダに配置を試みます")
                            # エラーの場合は既存フォルダへの配置処理に進む
                    
                    best_folder = next((f for f in leaf_folders if f['id'] == best_folder_id), None)
                    folder_name = best_folder['name'] if best_folder else best_folder_id
                    
                    print(f"    🎯 最も類似したフォルダ: {folder_name} (類似度: {max_similarity:.4f})")
                    
                    # --- 分類基準を使った振り分けロジック ---
                    # 後で使用するための変数を初期化
                    classification_criteria = {}
                    classification_words_found = []
                    target_folder_id_by_criteria = None
                    
                    # --- 指定したフォルダと同じ階層にあるフォルダを取得 ---
                    try:
                        # all_nodesから指定フォルダ（best_folder）の情報を取得
                        all_nodes = result_manager.get_all_nodes()
                        best_node = all_nodes.get(best_folder_id) if all_nodes else None
                        
                        if best_node:
                            parent_id_of_best = best_node.get('parent_id')
                            print(f"    📍 指定フォルダのparent_id: {parent_id_of_best}")
                            
                            # 同じparent_idを持つフォルダを取得
                            sibling_folders = []
                            for node_id, node_data in all_nodes.items():
                                if node_data.get('parent_id') == parent_id_of_best:
                                    sibling_folders.append({
                                        'id': node_id,
                                        'name': node_data.get('name'),
                                        'parent_id': node_data.get('parent_id'),
                                        'is_leaf': node_data.get('is_leaf', False)
                                    })
                            
                            print(f"    📂 同じ階層のフォルダ一覧 (count={len(sibling_folders)}):")
                            for sib in sibling_folders:
                                print(f"       - ID: {sib['id']}, Name: {sib['name']}, is_leaf: {sib['is_leaf']}")
                            
                            # --- is_leafフォルダのキャプション一覧を取得 ---
                            sibling_leaf_folders = [f for f in sibling_folders if f['is_leaf']]
                            print(f"\n    📝 is_leafフォルダのキャプション収集開始 ({len(sibling_leaf_folders)}個のフォルダ)")
                            
                            all_captions = []  # 全キャプションを格納
                            folder_captions_map = {}  # フォルダごとのキャプション
                            
                            for sib_folder in sibling_leaf_folders:
                                sib_folder_id = sib_folder['id']
                                sib_folder_name = sib_folder['name']
                                
                                # フォルダ内のclustering_idを取得
                                folder_data_result = result_manager.get_folder_data_from_result(sib_folder_id)
                                
                                if folder_data_result['success']:
                                    folder_data = folder_data_result['data']
                                    clustering_ids = list(folder_data.keys())
                                    
                                    folder_captions = []
                                    for cid in clustering_ids:
                                        try:
                                            caption_res, _ = images_queries.select_caption_by_clustering_id(connect_session, cid)
                                            if caption_res:
                                                caption_row = caption_res.mappings().first()
                                                if caption_row and 'caption' in caption_row and caption_row['caption']:
                                                    caption = caption_row['caption']
                                                    folder_captions.append(caption)
                                                    all_captions.append(caption)
                                        except Exception as cap_e:
                                            print(f"       ⚠️ キャプション取得エラー (clustering_id: {cid}): {cap_e}")
                                    
                                    folder_captions_map[sib_folder_id] = {
                                        'folder_name': sib_folder_name,
                                        'caption_count': len(folder_captions),
                                        'captions': folder_captions
                                    }
                            
                            print(f"    ✅ 収集完了: 全{len(all_captions)}個のキャプション")
                            
                            # --- 頻出単語リストの作成 ---
                            print(f"\n    📊 頻出単語分析開始...")
                            
                            from collections import Counter
                            import re
                            
                            # 全キャプションから単語を抽出
                            all_words = []
                            for caption in all_captions:
                                # 小文字化して単語に分割
                                words = re.findall(r'\b[a-z]+\b', caption.lower())
                                all_words.extend(words)
                            
                            # ストップワードを除外
                            stopwords_set = set(CAPTION_STOPWORDS)
                            filtered_words = [word for word in all_words if word not in stopwords_set]
                            
                            # 単語の出現回数をカウント
                            word_counter = Counter(filtered_words)
                            
                            # 頻出順にソート（重複なし）
                            frequent_words = word_counter.most_common()
                            
                            # --- 各フォルダの固有単語分析 ---
                            print(f"\n    🔍 各フォルダの特徴的な単語を抽出中...")
                            
                            # 各フォルダの単語カウンターを作成（文の位置によるバイアス付き）
                            folder_word_counters = {}
                            for sib_folder_id, folder_info in folder_captions_map.items():
                                folder_words = []
                                for caption in folder_info['captions']:
                                    # キャプションを文に分割（.で区切る）
                                    sentences = caption.split('.')
                                    
                                    for sentence_idx, sentence in enumerate(sentences):
                                        if not sentence.strip():  # 空の文はスキップ
                                            continue
                                        
                                        # 文の位置による重み（1文目: 1.0, 2文目: 0.85, 3文目: 0.7, それ以降: 0.6）
                                        # 極端にならないように調整
                                        if sentence_idx == 0:
                                            position_weight = 1.0
                                        elif sentence_idx == 1:
                                            position_weight = 0.85
                                        elif sentence_idx == 2:
                                            position_weight = 0.7
                                        else:
                                            position_weight = 0.6
                                        
                                        words = re.findall(r'\b[a-z]+\b', sentence.lower())
                                        filtered_sentence_words = [w for w in words if w not in stopwords_set]
                                        
                                        # 重み付きで単語をカウント（重みに応じて複数回追加）
                                        for word in filtered_sentence_words:
                                            # 重みを考慮するため、fractional countとして扱う
                                            # Counterは整数しか扱えないので、後でスコア計算時に適用
                                            folder_words.append((word, position_weight))
                                
                                # 重み付きカウンターを作成
                                weighted_counter = {}
                                for word, weight in folder_words:
                                    weighted_counter[word] = weighted_counter.get(word, 0.0) + weight
                                
                                folder_word_counters[sib_folder_id] = weighted_counter
                            
                            # 各フォルダの特徴的な単語を抽出（TF-IDF風のスコアリング）
                            folder_unique_words = {}
                            TOP_N_UNIQUE_WORDS = 10  # 各フォルダから上位N個の特徴的な単語を抽出
                            
                            for target_folder_id, target_counter in folder_word_counters.items():
                                # 各単語のスコアを計算（重み付きカウント対応）
                                word_scores = {}
                                
                                for word, count_in_target in target_counter.items():
                                    # このフォルダでの重み付き出現回数
                                    tf = count_in_target
                                    
                                    # 他のフォルダでの重み付き出現回数の合計
                                    count_in_others = sum(
                                        other_counter.get(word, 0.0) 
                                        for other_id, other_counter in folder_word_counters.items() 
                                        if other_id != target_folder_id
                                    )
                                    
                                    # スコア計算: (このフォルダでの重み付き出現回数) / (他のフォルダでの重み付き出現回数 + 1)
                                    # +1は0除算を防ぐため
                                    idf_like_score = tf / (count_in_others + 1.0)
                                    
                                    # 最終スコア: 重み付き出現回数 × IDF風スコア
                                    final_score = tf * idf_like_score
                                    
                                    word_scores[word] = {
                                        'score': final_score,
                                        'count_in_folder': tf,
                                        'count_in_others': count_in_others,
                                        'ratio': idf_like_score
                                    }
                                
                                # スコア順にソート
                                sorted_words = sorted(
                                    word_scores.items(), 
                                    key=lambda x: x[1]['score'], 
                                    reverse=True
                                )
                                
                                # 上位N個を取得
                                top_unique = sorted_words[:TOP_N_UNIQUE_WORDS]
                                
                                folder_unique_words[target_folder_id] = {
                                    'folder_name': folder_captions_map[target_folder_id]['folder_name'],
                                    'unique_words': [
                                        {
                                            'word': word,
                                            'score': round(info['score'], 2),
                                            'count_in_folder': info['count_in_folder'],
                                            'count_in_others': info['count_in_others'],
                                            'ratio': round(info['ratio'], 2)
                                        }
                                        for word, info in top_unique
                                    ]
                                }
                            
                            # --- フォルダ間の共通カテゴリ分析 ---
                            print(f"\n    🔍 フォルダ間の共通カテゴリ分析を開始...")
                            
                            # 各フォルダから上位10個の特徴的単語を取得
                            folder_top_words_list = {}
                            for folder_id, unique_info in folder_unique_words.items():
                                top_10_words = [w['word'] for w in unique_info['unique_words'][:10]]
                                folder_top_words_list[folder_id] = top_10_words
                            
                            # 全フォルダに共通する単語を特定
                            if len(folder_top_words_list) > 0:
                                # 各フォルダの単語セットを作成
                                folder_word_sets = [set(words) for words in folder_top_words_list.values()]
                                # 全フォルダに共通する単語を取得
                                common_to_all_folders = set.intersection(*folder_word_sets) if len(folder_word_sets) > 1 else set()
                                
                                print(f"\n    🔍 全フォルダに共通する単語を除外...")
                                print(f"       - 全フォルダ数: {len(folder_word_sets)}")
                                print(f"       - 全フォルダに共通する単語数: {len(common_to_all_folders)}")
                                
                                if len(common_to_all_folders) > 0:
                                    common_words_display = ', '.join(sorted(list(common_to_all_folders)))
                                    print(f"       - 共通単語: {common_words_display}")
                                    
                                    # 各フォルダのトップ10単語から共通単語を除外
                                    for folder_id in folder_top_words_list.keys():
                                        original_count = len(folder_top_words_list[folder_id])
                                        folder_top_words_list[folder_id] = [
                                            w for w in folder_top_words_list[folder_id] 
                                            if w not in common_to_all_folders
                                        ]
                                        removed_count = original_count - len(folder_top_words_list[folder_id])
                                        if removed_count > 0:
                                            folder_name = folder_unique_words[folder_id]['folder_name']
                                            print(f"       - 📁 {folder_name}: {removed_count}個の共通単語を除外")
                                else:
                                    print(f"       ℹ️ 全フォルダに共通する単語はありません")
                            
                            # WordAnalyzerを初期化（既存のWordNetメソッドを使用）
                            from sentence_transformers import SentenceTransformer
                            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                            word_analyzer = WordAnalyzer(embedding_model)
                            
                            # --- 新しいロジック: 全フォルダで同じカテゴリを持つ単語のみを抽出 ---
                            print(f"\n    🔍 全フォルダ間で共通するカテゴリの単語を分析...")
                            
                            folder_ids_list = list(folder_unique_words.keys())
                            
                            # 各単語がどのカテゴリに属するかをフォルダごとに分析
                            # {folder_id: {word: [(category, score, target_word), ...]}}
                            folder_word_categories = {}
                            
                            for folder_id in folder_ids_list:
                                folder_word_categories[folder_id] = {}
                                folder_words = folder_top_words_list[folder_id]
                                
                                for word in folder_words:
                                    # この単語と他の全フォルダの単語を比較してカテゴリを取得
                                    word_category_info = []  # [(category, score, other_folder_word), ...]
                                    
                                    for other_folder_id in folder_ids_list:
                                        if other_folder_id == folder_id:
                                            continue
                                        
                                        other_folder_words = folder_top_words_list[other_folder_id]
                                        
                                        for other_word in other_folder_words:
                                            common_categories, category_score = word_analyzer.get_common_category(word, other_word)
                                            
                                            if len(common_categories) > 0 and category_score >= 3.0:
                                                # スコア3.0以上の共通カテゴリのみ
                                                for cat in common_categories[:1]:  # 最上位カテゴリのみ
                                                    word_category_info.append((cat, category_score, other_word, other_folder_id))
                                    
                                    if len(word_category_info) > 0:
                                        folder_word_categories[folder_id][word] = word_category_info
                            
                            # 全フォルダで共通して出現するカテゴリを特定
                            print(f"\n    🔍 全フォルダで共通するカテゴリを特定中...")
                            
                            from collections import defaultdict
                            category_occurrence = defaultdict(lambda: {
                                'folders': set(),
                                'words_by_folder': defaultdict(list),  # {folder_id: [(word, avg_score)]}
                                'word_category_scores': defaultdict(list)  # {word: [scores]}
                            })
                            
                            # 各フォルダの各単語が属するカテゴリを集計
                            for folder_id, word_cats in folder_word_categories.items():
                                for word, cat_info_list in word_cats.items():
                                    if len(cat_info_list) == 0:
                                        continue
                                    
                                    # このwordが最も属するカテゴリを決定（スコアの平均が最も高いカテゴリ）
                                    cat_scores = defaultdict(list)
                                    for cat, score, other_word, other_folder_id in cat_info_list:
                                        cat_scores[cat].append(score)
                                    
                                    # 各カテゴリの平均スコアを計算
                                    best_category = None
                                    best_avg_score = -1
                                    for cat, scores in cat_scores.items():
                                        avg_score = sum(scores) / len(scores)
                                        if avg_score > best_avg_score:
                                            best_avg_score = avg_score
                                            best_category = cat
                                    
                                    if best_category:
                                        category_occurrence[best_category]['folders'].add(folder_id)
                                        category_occurrence[best_category]['words_by_folder'][folder_id].append((word, best_avg_score))
                                        category_occurrence[best_category]['word_category_scores'][word].append(best_avg_score)
                            
                            # 全フォルダに出現するカテゴリのみをフィルタリング
                            num_folders = len(folder_ids_list)
                            common_categories_across_all_folders = {}
                            
                            for category, info in category_occurrence.items():
                                if len(info['folders']) == num_folders:
                                    # 全フォルダに出現するカテゴリ
                                    common_categories_across_all_folders[category] = info
                            
                            print(f"       ✅ 全{num_folders}個のフォルダに共通するカテゴリ数: {len(common_categories_across_all_folders)}")
                            
                            # 分類基準の推定
                            classification_criteria = {}
                            
                            if len(common_categories_across_all_folders) > 0:
                                # 各カテゴリの平均スコアと単語数で評価
                                sorted_categories = []
                                for category, info in common_categories_across_all_folders.items():
                                    # 各単語の平均スコアを計算
                                    word_scores = []
                                    words_with_scores = []  # [(word, avg_score)]
                                    
                                    for word, scores_list in info['word_category_scores'].items():
                                        avg_score = sum(scores_list) / len(scores_list)
                                        word_scores.append(avg_score)
                                        words_with_scores.append((word, avg_score))
                                    
                                    # スコア順にソート
                                    words_with_scores.sort(key=lambda x: x[1], reverse=True)
                                    
                                    category_avg_score = sum(word_scores) / len(word_scores) if len(word_scores) > 0 else 0.0
                                    
                                    sorted_categories.append((category, {
                                        'words_with_scores': words_with_scores,
                                        'avg_score': category_avg_score,
                                        'word_count': len(words_with_scores),
                                        'folders': info['folders']
                                    }))
                                
                                # 平均スコア順にソート
                                sorted_categories.sort(key=lambda x: x[1]['avg_score'], reverse=True)
                                
                                print(f"\n    📋 推定される分類基準（全フォルダ共通カテゴリ）:")
                                
                                for rank, (category, info) in enumerate(sorted_categories[:10], 1):
                                    words_display = ', '.join([f"{w}({s:.2f})" for w, s in info['words_with_scores'][:5]])
                                    folders_list = sorted([folder_unique_words[fid]['folder_name'] for fid in info['folders']])
                                    
                                    # 各フォルダでこのカテゴリに属する単語を表示
                                    folder_words_display = []
                                    for fid in info['folders']:
                                        fname = folder_unique_words[fid]['folder_name']
                                        # このフォルダでこのカテゴリに属する単語
                                        fwords = [w for w, s in info['words_with_scores'] if any(
                                            w == fw[0] for fw in common_categories_across_all_folders[category]['words_by_folder'].get(fid, [])
                                        )]
                                        if len(fwords) > 0:
                                            folder_words_display.append(f"{fname}:{fwords[0]}")
                                    
                                    classification_criteria[category] = {
                                        'rank': rank,
                                        'category': category,
                                        'words': [w for w, s in info['words_with_scores']],
                                        'words_with_scores': info['words_with_scores'],
                                        'word_count': info['word_count'],
                                        'avg_score': round(info['avg_score'], 2),
                                        'folders': folders_list,
                                        'folder_words': folder_words_display
                                    }
                                    
                                    print(f"\n       {rank}. カテゴリ: {category}")
                                    print(f"          - 平均スコア: {info['avg_score']:.2f}")
                                    print(f"          - 単語数: {info['word_count']}")
                                    print(f"          - 全フォルダ数: {len(info['folders'])}")
                                    print(f"          - 該当単語 (スコア順上位5): {words_display}")
                                    print(f"          - フォルダ別単語: {', '.join(folder_words_display[:5])}")
                                
                                # 最も支配的なカテゴリを分類基準として特定
                                if len(sorted_categories) > 0:
                                    top_category = sorted_categories[0][0]
                                    top_info = sorted_categories[0][1]
                                    top_words = ', '.join([w for w, s in top_info['words_with_scores'][:5]])
                                    
                                    print(f"\n    🎯 最も可能性の高い分類基準:")
                                    print(f"       カテゴリ: {top_category}")
                                    print(f"       平均スコア: {top_info['avg_score']:.2f}")
                                    print(f"       該当単語: {top_words}")
                                    
                            else:
                                print(f"       ⚠️ 全フォルダに共通するカテゴリが見つかりませんでした")
                            
                            # デバッグ用JSON出力データを作成
                            debug_output = {
                                'summary': {
                                    'total_captions': len(all_captions),
                                    'total_words_before_filtering': len(all_words),
                                    'total_words_after_filtering': len(filtered_words),
                                    'unique_words': len(word_counter),
                                    'sibling_leaf_folder_count': len(sibling_leaf_folders),
                                    'common_to_all_folders_count': len(common_to_all_folders) if 'common_to_all_folders' in locals() else 0,
                                    'common_categories_count': len(common_categories_across_all_folders) if 'common_categories_across_all_folders' in locals() else 0,
                                    'classification_criteria_count': len(classification_criteria)
                                },
                                'common_to_all_folders': sorted(list(common_to_all_folders)) if 'common_to_all_folders' in locals() else [],
                                'common_categories_across_all_folders': {
                                    cat: {
                                        'words': [w for w, s in info['words_with_scores']] if 'words_with_scores' in info else [],
                                        'avg_score': info.get('avg_score', 0.0),
                                        'folders': list(info.get('folders', []))
                                    }
                                    for cat, info in (dict(sorted_categories) if 'sorted_categories' in locals() and sorted_categories else {}).items()
                                } if 'sorted_categories' in locals() else {},
                                'classification_criteria': classification_criteria,
                                'folder_captions': folder_captions_map,
                                'frequent_words': [
                                    {
                                        'word': word,
                                        'count': count,
                                        'rank': idx + 1
                                    }
                                    for idx, (word, count) in enumerate(frequent_words)
                                ],
                                'top_20_words': [
                                    {
                                        'word': word,
                                        'count': count
                                    }
                                    for word, count in frequent_words[:20]
                                ],
                                'folder_unique_words': folder_unique_words
                            }
                            
                            # JSON形式で出力
                            import json
                            from datetime import datetime
                            
                            # JSONファイルに保存
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            json_filename = f"sibling_captions_analysis_{timestamp}.json"
                            json_filepath = os.path.join("./", json_filename)
                            
                            try:
                                with open(json_filepath, 'w', encoding='utf-8') as f:
                                    json.dump(debug_output, f, indent=2, ensure_ascii=False)
                                print(f"\n    💾 JSONファイルに保存しました: {json_filepath}")
                            except Exception as json_e:
                                print(f"    ⚠️ JSONファイル保存エラー: {json_e}")
                            
                            # コンソールにも出力（簡易版）
                            print(f"\n    📋 デバッグ用JSON出力サマリー:")
                            print(f"       - 総キャプション数: {debug_output['summary']['total_captions']}")
                            print(f"       - ユニークな単語数: {debug_output['summary']['unique_words']}")
                            print(f"       - フォルダ数: {debug_output['summary']['sibling_leaf_folder_count']}")
                            
                            print(f"\n    🔝 Top 20 頻出単語:")
                            for idx, (word, count) in enumerate(frequent_words[:20], 1):
                                print(f"       {idx:2d}. {word:20s} : {count:4d}回")
                            
                            print(f"\n    🎯 各フォルダの特徴的な単語 (Top {TOP_N_UNIQUE_WORDS}):")
                            for folder_id, unique_info in folder_unique_words.items():
                                folder_name = unique_info['folder_name']
                                print(f"\n       📁 {folder_name} (ID: {folder_id}):")
                                for rank, word_info in enumerate(unique_info['unique_words'], 1):
                                    print(f"          {rank:2d}. {word_info['word']:20s} | "
                                          f"スコア: {word_info['score']:6.2f} | "
                                          f"このフォルダ: {word_info['count_in_folder']:6.2f}回 | "
                                          f"他フォルダ: {word_info['count_in_others']:6.2f}回 | "
                                          f"比率: {word_info['ratio']:5.2f}")
                            
                            
                            # --- 分類基準を使った新規画像の振り分けロジック ---
                            print(f"\n    🔍 分類基準を使った振り分けロジックを開始...")
                            
                            # 新規画像のキャプションを取得
                            try:
                                new_image_caption_res, _ = images_queries.select_caption_by_clustering_id(connect_session, clustering_id)
                                new_image_caption = None
                                if new_image_caption_res:
                                    caption_row = new_image_caption_res.mappings().first()
                                    if caption_row and 'caption' in caption_row and caption_row['caption']:
                                        new_image_caption = caption_row['caption'].lower()
                                
                                if new_image_caption:
                                    print(f"       📝 新規画像のキャプション: {new_image_caption[:100]}...")
                                    
                                    # 分類基準から最上位カテゴリのみを使用
                                    if len(classification_criteria) > 0:
                                        # rankでソート（すでにrankがついている）
                                        sorted_criteria = sorted(
                                            classification_criteria.items(),
                                            key=lambda x: x[1].get('rank', 999)
                                        )
                                        
                                        # 最上位カテゴリのみを使用
                                        top_category, top_info = sorted_criteria[0]
                                        top_category_words_with_scores = top_info.get('words_with_scores', [])
                                        
                                        print(f"\n       🎯 分類基準カテゴリ: {top_category}")
                                        print(f"          該当単語: {', '.join([f'{w}({s:.2f})' for w, s in top_category_words_with_scores[:5]])}")
                                        
                                        # 新規画像のキャプションから単語を抽出
                                        new_image_words = set(re.findall(r'\b[a-z]+\b', new_image_caption))
                                        # ストップワードを除外
                                        new_image_words = new_image_words - stopwords_set
                                        
                                        # このカテゴリの単語リスト
                                        top_category_words = set([w for w, s in top_category_words_with_scores])
                                        
                                        # キャプション内に分類基準単語が含まれているか確認（完全一致）
                                        found_exact_words = new_image_words & top_category_words
                                        
                                        # WordNetを使ってこのカテゴリに属する単語を探す
                                        category_matched_words = []  # [(word, avg_score)]
                                        
                                        # 除外する単語セット: ストップワード + 全フォルダ共通単語
                                        exclude_words_for_matching = stopwords_set.copy()
                                        if 'common_to_all_folders' in locals() and len(common_to_all_folders) > 0:
                                            exclude_words_for_matching.update(common_to_all_folders)
                                        
                                        for new_word in new_image_words:
                                            if new_word in top_category_words:
                                                # すでに完全一致で見つかっている
                                                continue
                                            
                                            # 除外単語リストに含まれていればスキップ
                                            if new_word in exclude_words_for_matching:
                                                continue
                                            
                                            # 新規単語がこのカテゴリに属するかチェック
                                            scores_for_word = []
                                            for category_word, cat_word_score in top_category_words_with_scores:
                                                common_categories, category_score = word_analyzer.get_common_category(new_word, category_word)
                                                
                                                # 最上位の共通カテゴリがtop_categoryと一致するかチェック
                                                if len(common_categories) > 0 and common_categories[0] == top_category and category_score >= 3.0:
                                                    scores_for_word.append(category_score)
                                                    print(f"       🔗 '{new_word}' は '{category_word}' と同じカテゴリ '{top_category}' (スコア: {category_score:.2f})")
                                            
                                            if len(scores_for_word) > 0:
                                                # この単語のカテゴリとの平均スコアを計算
                                                avg_score = sum(scores_for_word) / len(scores_for_word)
                                                category_matched_words.append((new_word, avg_score))
                                        
                                        # スコア順にソート
                                        category_matched_words.sort(key=lambda x: x[1], reverse=True)
                                        
                                        if len(found_exact_words) > 0:
                                            print(f"\n       ✅ キャプション内に分類基準単語を発見（完全一致）: {', '.join(found_exact_words)}")
                                        
                                        if len(category_matched_words) > 0:
                                            print(f"\n       🔍 キャプション内に分類基準カテゴリ '{top_category}' に属する単語を発見:")
                                            top_matched = [f"{w}({s:.2f})" for w, s in category_matched_words[:5]]
                                            print(f"          {', '.join(top_matched)}")
                                        
                                        # 完全一致 + カテゴリマッチを統合
                                        all_found_words = found_exact_words.copy()
                                        for word, score in category_matched_words:
                                            all_found_words.add(word)
                                        
                                        if len(all_found_words) > 0:
                                            print(f"\n       📂 既存の兄弟フォルダと照合します...")
                                            
                                            # 候補フォルダをスコア付きで格納
                                            folder_candidates = []
                                            
                                            # まず完全一致の単語を優先してチェック
                                            for word in found_exact_words:
                                                for sib_folder in sibling_leaf_folders:
                                                    sib_folder_name = sib_folder['name'].lower()
                                                    
                                                    # 完全一致: フォルダ名 == 単語 または フォルダ名に単語が含まれる
                                                    if sib_folder_name == word or word in sib_folder_name.split(','):
                                                        # このフォルダのTF-IDFスコアを取得
                                                        folder_score = 0.0
                                                        if sib_folder['id'] in folder_unique_words:
                                                            for w_info in folder_unique_words[sib_folder['id']]['unique_words']:
                                                                if w_info['word'] == word:
                                                                    folder_score = w_info['score']
                                                                    break
                                                        
                                                        folder_candidates.append({
                                                            'folder': sib_folder,
                                                            'word': word,
                                                            'score': folder_score + 1000,  # 完全一致を優先
                                                            'match_type': 'exact'
                                                        })
                                                        print(f"       🎯 候補フォルダ発見（完全一致）: '{word}' → '{sib_folder['name']}' (スコア: {folder_score:.2f})")
                                            
                                            # 完全一致候補がなければ、カテゴリマッチ単語をチェック
                                            if len(folder_candidates) == 0:
                                                # カテゴリに最も近い単語を優先（スコア順にソート済み）
                                                for word, word_category_score in category_matched_words:
                                                    for sib_folder in sibling_leaf_folders:
                                                        sib_folder_name = sib_folder['name'].lower()
                                                        if sib_folder_name == word or word in sib_folder_name.split(','):
                                                            folder_score = 0.0
                                                            if sib_folder['id'] in folder_unique_words:
                                                                for w_info in folder_unique_words[sib_folder['id']]['unique_words']:
                                                                    if w_info['word'] == word:
                                                                        folder_score = w_info['score']
                                                                        break
                                                            
                                                            # カテゴリスコアとフォルダスコアを組み合わせて評価
                                                            combined_score = folder_score + (word_category_score * 0.1)
                                                            
                                                            folder_candidates.append({
                                                                'folder': sib_folder,
                                                                'word': word,
                                                                'score': combined_score,
                                                                'match_type': 'category',
                                                                'category_score': word_category_score
                                                            })
                                                            print(f"       🎯 候補フォルダ発見（カテゴリ一致）: '{word}' (カテゴリスコア: {word_category_score:.2f}) → '{sib_folder['name']}' (総合スコア: {combined_score:.2f})")
                                            
                                            # 候補から最適なフォルダを選択
                                            matched_folder = None
                                            matched_word = None
                                            
                                            if len(folder_candidates) > 0:
                                                # スコアが高い順にソート
                                                folder_candidates.sort(key=lambda x: x['score'], reverse=True)
                                                best_candidate = folder_candidates[0]
                                                matched_folder = best_candidate['folder']
                                                matched_word = best_candidate['word']
                                                
                                                print(f"       ⭐ 最適フォルダを選択: '{matched_word}' → '{matched_folder['name']}' (スコア: {best_candidate['score']:.2f}, タイプ: {best_candidate['match_type']})")
                                            
                                            if matched_folder:
                                                # 既存フォルダに挿入
                                                target_folder_id_by_criteria = matched_folder['id']
                                                print(f"       📂 既存フォルダへ挿入予定: {matched_folder['name']} (ID: {target_folder_id_by_criteria})")
                                            else:
                                                # 新規フォルダ作成（完全一致を優先、なければカテゴリマッチで最もスコアの高い単語を使用）
                                                new_folder_word = None
                                                if len(found_exact_words) > 0:
                                                    new_folder_word = list(found_exact_words)[0]
                                                else:
                                                    # カテゴリマッチから最もカテゴリに近い単語を取得（最高スコア）
                                                    if len(category_matched_words) > 0:
                                                        new_folder_word = category_matched_words[0][0]  # 最高スコアの単語
                                                
                                                if new_folder_word:
                                                    print(f"       🆕 新規フォルダ作成予定: フォルダ名='{new_folder_word}'")
                                                    
                                                    # 親フォルダIDを取得（best_folderと同じ階層）
                                                    parent_id_for_new = parent_id_of_best if 'parent_id_of_best' in locals() else None
                                                    
                                                    # 画像パスを取得
                                                    path_result_temp, _ = action_queries.get_image_name_by_id(connect_session, image_id)
                                                    image_path_temp = path_result_temp.mappings().first()['name']
                                                    
                                                    # 新規フォルダ作成
                                                    create_result = result_manager.create_new_leaf_folder(
                                                        folder_name=new_folder_word,
                                                        parent_id=parent_id_for_new,
                                                        initial_clustering_id=clustering_id,
                                                        initial_image_path=image_path_temp
                                                    )
                                                    
                                                    if create_result['success']:
                                                        new_folder_id = create_result['folder_id']
                                                        target_folder_id_by_criteria = new_folder_id
                                                        print(f"       ✅ 新規フォルダ作成成功: '{new_folder_word}' (ID: {new_folder_id})")
                                                        
                                                        # user_image_clustering_statesを更新
                                                        _, _ = action_queries.update_user_image_state_for_image(connect_session, user_id, image_id, new_count)
                                                        
                                                        # 新しいフォルダの埋め込みベクトルを追加
                                                        if new_sentence_embedding is not None:
                                                            folder_sentence_embeddings[new_folder_id] = new_sentence_embedding
                                                        if new_image_embedding is not None:
                                                            folder_image_embeddings[new_folder_id] = new_image_embedding
                                                        
                                                        # leaf_foldersリストにも追加
                                                        leaf_folders.append({
                                                            'id': new_folder_id,
                                                            'name': new_folder_word,
                                                            'parent_id': parent_id_for_new,
                                                            'is_leaf': True
                                                        })
                                                        
                                                        # sibling_leaf_foldersにも追加
                                                        sibling_leaf_folders.append({
                                                            'id': new_folder_id,
                                                            'name': new_folder_word,
                                                            'parent_id': parent_id_for_new,
                                                            'is_leaf': True
                                                        })
                                                        
                                                        print(f"       ℹ️ 新規フォルダ作成により、後続の既存フォルダ挿入処理はスキップします")
                                                        # 新規フォルダ作成が成功したので、後続の挿入処理をスキップ
                                                        continue
                                                    else:
                                                        print(f"       ❌ 新規フォルダ作成失敗: {create_result.get('error', 'Unknown error')}")
                                                        print(f"       → フォールバックとして最も類似したフォルダに挿入します")
                                                        target_folder_id_by_criteria = None
                                                    print(f"       ℹ️ 新規フォルダ作成により、後続の既存フォルダ挿入処理はスキップします")
                                                    # 新規フォルダ作成が成功したので、後続の挿入処理をスキップ
                                                    continue
                                                else:
                                                    print(f"       ❌ 新規フォルダ作成失敗: {create_result.get('error', 'Unknown error')}")
                                                    print(f"       → フォールバックとして最も類似したフォルダに挿入します")
                                                    target_folder_id_by_criteria = None
                                        else:
                                            print(f"       ℹ️ キャプション内に分類基準単語が含まれていません")
                                            print(f"       → 最も類似したフォルダ（{folder_name}）に挿入します")
                                    else:
                                        print(f"       ℹ️ 分類基準が生成されていません")
                                        print(f"       → 最も類似したフォルダ（{folder_name}）に挿入します")
                                else:
                                    print(f"       ⚠️ 新規画像のキャプションが取得できませんでした")
                                    print(f"       → 最も類似したフォルダ（{folder_name}）に挿入します")
                            
                            except Exception as criteria_e:
                                print(f"       ⚠️ 分類基準による振り分け処理でエラー: {criteria_e}")
                                traceback.print_exc()
                                print(f"       → フォールバックとして最も類似したフォルダに挿入します")
                            
                        else:
                            print(f"    ⚠️ 指定フォルダ {best_folder_id} がall_nodesに見つかりません")
                    except Exception as sib_e:
                        print(f"    ⚠️ 同階層フォルダ取得エラー: {sib_e}")
                        traceback.print_exc()
                    
                    # 最終的な挿入先フォルダを決定
                    final_target_folder_id = target_folder_id_by_criteria if target_folder_id_by_criteria else best_folder_id
                    
                    print(f"\n    📌 最終挿入先フォルダ: ID={final_target_folder_id}")
                    
                    # 画像をフォルダに挿入
                    # imagesテーブルからimage_pathを取得
                    path_result, _ = action_queries.get_image_name_by_id(connect_session, image_id)
                    image_path = path_result.mappings().first()['name']
                    
                    insert_result = result_manager.insert_image_to_leaf_folder(
                        clustering_id=clustering_id,
                        image_path=image_path,
                        target_folder_id=final_target_folder_id
                    )

                    if insert_result['success']:
                        print(f"    ✅ フォルダに画像を挿入しました")

                        # user_image_clustering_statesを更新
                        _, _ = action_queries.update_user_image_state_for_image(connect_session, user_id, image_id, new_count)

                        # フォルダの埋め込みベクトルを再計算（新しい画像を追加したため）
                        print(f"    🔄 フォルダ埋め込みベクトルを再計算中...")
                        try:
                            # result内でフォルダIDを探索してdataを取得
                            folder_data_result = result_manager.get_folder_data_from_result(final_target_folder_id)
                            if folder_data_result['success']:
                                folder_data = folder_data_result['data']
                                clustering_ids = list(folder_data.keys())

                                # 文章埋め込みベクトルの再計算
                                sentence_ids = []
                                for cid in clustering_ids:
                                    sent_result, _ = action_queries.get_chromadb_sentence_id_by_clustering_id(connect_session, cid, project_id)
                                    if sent_result:
                                        sent_mapping = sent_result.mappings().first()
                                        if sent_mapping:
                                            sentence_ids.append(sent_mapping['chromadb_sentence_id'])

                                if len(sentence_ids) > 0:
                                    updated_sentence_data = sentence_name_db.get_data_by_ids(sentence_ids)
                                    updated_sentence_embeddings = updated_sentence_data['embeddings']
                                    folder_sentence_embeddings[final_target_folder_id] = np.mean(updated_sentence_embeddings, axis=0)
                                    print(f"    ✅ フォルダ文章埋め込みベクトル再計算完了 ({len(sentence_ids)}個の文章)")
                                
                                # 画像埋め込みベクトルの再計算
                                image_ids = []
                                for cid in clustering_ids:
                                    img_result, _ = action_queries.get_chromadb_image_id_by_clustering_id(connect_session, cid, project_id)
                                    if img_result:
                                        img_mapping = img_result.mappings().first()
                                        if img_mapping:
                                            image_ids.append(img_mapping['chromadb_image_id'])

                                if len(image_ids) > 0:
                                    updated_image_data = image_db.get_data_by_ids(image_ids)
                                    updated_image_embeddings = updated_image_data['embeddings']
                                    folder_image_embeddings[final_target_folder_id] = np.mean(updated_image_embeddings, axis=0)
                                    print(f"    ✅ フォルダ画像埋め込みベクトル再計算完了 ({len(image_ids)}個の画像)")
                            else:
                                print(f"    ⚠️ フォルダデータの再取得失敗: {folder_data_result.get('error', 'Unknown error')}")
                        except Exception as e:
                            print(f"    ⚠️ フォルダ埋め込みベクトル再計算エラー: {e}")
                            traceback.print_exc()
                    else:
                        print(f"    ❌ 画像挿入エラー: {insert_result.get('error', 'Unknown error')}")
                        
                except Exception as img_error:
                    print(f"    ❌ 画像処理中にエラー: {img_error}")
                    continue
            
            # project_membershipsのexecuted_clustering_countを更新
            _, _ = action_queries.update_project_executed_clustering_count(connect_session, user_id, project_id, new_count)
            
            # 未クラスタリング画像が残っているか確認
            check_result, _ = action_queries.get_unclustered_count_for_project(connect_session, user_id, project_id)
            remaining_unclustered = check_result.mappings().first()['unclustered_count']
            
            print(f"\n📊 クラスタリング完了後の状態確認:")
            print(f"   残りの未クラスタリング画像数: {remaining_unclustered}")
            
            # 未クラスタリング画像が残っていれば2（実行可能）、なければ0（実行不可能）
            new_state = 2 if remaining_unclustered > 0 else 0
            state_description = "実行可能" if new_state == 2 else "実行不可能"
            
            _, _ = action_queries.update_continuous_state(connect_session, user_id, project_id, new_state)
            
            print(f"   continuous_clustering_state: {new_state} ({state_description})")
            print(f"\n✅ 継続的クラスタリング バックグラウンド処理完了")
            print(f"   処理した画像数: {len(unclustered_rows)}")
            print(f"   新しいクラスタリング回数: {new_count}")
            
        except Exception as e:
            print(f"❌ 継続的クラスタリング処理中にエラー: {str(e)}")
            traceback.print_exc()
            
            # エラー時も未クラスタリング画像の有無を確認して状態を設定
            try:
                check_result, _ = action_queries.get_unclustered_count_for_project(connect_session, user_id, project_id)
                remaining_unclustered = check_result.mappings().first()['unclustered_count']
                
                # 未クラスタリング画像が残っていれば2（実行可能）、なければ0（実行不可能）
                new_state = 2 if remaining_unclustered > 0 else 0
                
                _, _ = action_queries.update_continuous_state(connect_session, user_id, project_id, new_state)
                print(f"⚠️ エラー後の状態更新: continuous_clustering_state = {new_state} (未クラスタリング画像: {remaining_unclustered})")
            except Exception as state_error:
                print(f"⚠️ エラー後の状態更新に失敗: {state_error}")
                
    # continuous_clustering_stateを1（実行中）に更新
    _, _ = action_queries.update_continuous_state(connect_session, user_id, project_id, 1)
    
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


@action_endpoint.get("/action/clustering/captions/{mongo_result_id}", tags=["action"], description="指定フォルダ内のクラスタリングIDに対応するキャプションを取得する")
async def get_captions_for_folder(mongo_result_id: str, folder_id: str = Query(..., description="フォルダの node_id")):
    """
    mongo_result_id (パス) と folder_id (クエリ) を受け取り、そのフォルダに含まれる画像の clustering_id を取得し、
    images テーブルから caption を取得して {clustering_id: caption} のマップを返す。
    """
    try:
        if not mongo_result_id or not mongo_result_id.strip():
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "mongo_result_id is required"})

        if not folder_id or not folder_id.strip():
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "folder_id is required"})

        # ResultManagerを初期化してフォルダ内のclustering_id一覧を取得
        result_manager = ResultManager(mongo_result_id)
        clustering_ids_result = result_manager.get_leaf_folder_image_clustering_ids(folder_id)

        # debug logs for tracing
        print(f"🔍 get_captions_for_folder: clustering_ids_result={clustering_ids_result}")

        if not clustering_ids_result.get('success', False):
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": clustering_ids_result.get('error', 'folder not found'), "data": None})

        clustering_ids = clustering_ids_result.get('data', [])
        print(f"🔍 get_captions_for_folder: clustering_ids (count={len(clustering_ids)}): {clustering_ids[:50]}")

        captions_map = {}

        # DBセッションを作成して1つずつcaptionを取得（将来的にINクエリへ最適化可）
        connect_session = create_connect_session()
        if connect_session is None:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database"})

        for cid in clustering_ids:
            try:
                res, _ = images_queries.select_caption_by_clustering_id(connect_session, cid)
                if res is None:
                    captions_map[cid] = None
                    continue
                mapping = res.mappings().first()
                captions_map[cid] = mapping['caption'] if mapping and 'caption' in mapping else None
            except Exception as q_e:
                # 個別取得に失敗しても他の結果は返す
                captions_map[cid] = None

        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "success", "data": {"folder_id": folder_id, "captions": captions_map}})

    except Exception as e:
        print(f"❌ get_captions_for_folderエラー: {e}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": f"Internal server error: {str(e)}"})


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


@action_endpoint.get("/action/clustering/download/{project_id}", tags=["action"], description="分類結果をダウンロードする")
async def download_classification_result(
    project_id: int,
    user_id: int = Query(..., description="ユーザーID")
):
    """
    分類結果をZIPファイルとしてダウンロードする
    
    ZIPファイルの内容:
    - result.json: 分類結果の階層構造
    - all_nodes.json: 全ノードの情報
    - images/: 分類結果に基づいたフォルダ構造の画像ファイル
    
    Args:
        project_id: プロジェクトID
        user_id: ユーザーID
        
    Returns:
        FileResponse: ZIPファイル
    """
    connect_session = create_connect_session()
    
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    try:
        # プロジェクト情報とmongo_result_idを取得
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
        
        result, _ = action_queries.get_project_info_and_mongo(connect_session, project_id, user_id)
        result_mapping = result.mappings().first()
        
        if result_mapping is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "project or membership not found", "data": None}
            )
        
        project_name = result_mapping['project_name']
        original_images_folder_path = result_mapping['original_images_folder_path']
        mongo_result_id = result_mapping['mongo_result_id']
        init_clustering_state = result_mapping['init_clustering_state']
        
        # 初期クラスタリングが完了していない場合はエラー
        if init_clustering_state != INIT_CLUSTERING_STATUS.FINISHED:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "clustering not completed yet", "data": None}
            )
        
        print(f"📦 ダウンロード処理開始:")
        print(f"   プロジェクト: {project_name}")
        print(f"   ユーザーID: {user_id}")
        print(f"   mongo_result_id: {mongo_result_id}")
        
        # ResultManagerから分類結果データを取得
        result_manager = ResultManager(mongo_result_id)
        export_data = result_manager.export_classification_data()
        
        if not export_data['success']:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": f"failed to export classification data: {export_data.get('error', 'Unknown error')}",
                    "data": None
                }
            )
        
        result_dict = export_data['result']
        all_nodes_dict = export_data['all_nodes']
        
        # 画像フォルダのパス
        source_images_path = Path(f"./{DEFAULT_IMAGE_PATH}/{original_images_folder_path}")
        
        if not source_images_path.exists():
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": f"source images folder not found: {source_images_path}",
                    "data": None
                }
            )
        
        print(f"   画像フォルダ: {source_images_path}")
        
        # ダウンロードパッケージを作成
        try:
            zip_path = Utils.create_classification_download_package(
                result_dict=result_dict,
                all_nodes_dict=all_nodes_dict,
                source_images_path=source_images_path,
                project_name=project_name
            )
            
            print(f"   ZIPファイル作成完了: {zip_path}")
            
            # ZIPファイルをダウンロード
            return FileResponse(
                path=str(zip_path),
                media_type='application/zip',
                filename=f"{project_name}.zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{project_name}.zip"'
                }
            )
            
        except Exception as create_error:
            print(f"❌ ZIPファイル作成エラー: {create_error}")
            traceback.print_exc()
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "message": f"failed to create download package: {str(create_error)}",
                    "data": None
                }
            )
        
    except Exception as e:
        print(f"❌ download_classification_result処理中にエラー: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": f"Internal server error: {str(e)}",
                "data": None
            }
        )


@action_endpoint.get("/action/clustering/counts/{project_id}", tags=["action"], description="プロジェクト内の画像のクラスタリング回数情報を取得する")
async def get_clustering_counts(
    project_id: int,
    user_id: int = Query(..., description="ユーザーID")
):
    """
    プロジェクト内の全画像のクラスタリング回数情報を取得する
    
    Returns:
        {
            "available_counts": [0, 1, 2, ...],  # 実行された回数のリスト
            "image_counts": {
                "clustering_id_1": 0,  # 各画像のクラスタリング回数
                "clustering_id_2": 1,
                ...
            }
        }
    """
    connect_session = create_connect_session()
    
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )
    
    try:
        # プロジェクトメンバーシップを確認（COUNTで存在確認する。
        # 一部環境で project_memberships に id カラムが無い可能性があるため、単純な存在確認を使う）
        membership_result, _ = action_queries.membership_exists(connect_session, project_id, user_id)

        # execute_query が失敗して None を返す場合を安全に扱う
        if membership_result is None:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": "failed to query project_memberships", "data": None}
            )

        membership_row = membership_result.mappings().first()
        if membership_row is None or membership_row.get('cnt', 0) == 0:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "project membership not found", "data": None}
            )
        
        result, _ = action_queries.get_image_counts_for_clustering_counts(connect_session, user_id, project_id)
        rows = result.mappings().all()

        # executed_clustering_count ごとに clustering_id の配列を作成
        grouped_by_count: dict[str, list] = {}
        # clustering_id -> executed_clustering_count の辞書
        image_counts: dict = {}
        available_counts_set = set()

        for row in rows:
            clustering_id = row.get('clustering_id')
            count = row.get('exec_count')

            # clustering_id または count が無い場合はスキップ
            if clustering_id is None or count is None:
                continue

            # image_counts マップ
            image_counts[clustering_id] = int(count)

            # grouped map: key を文字列にして返す（例: '0', '1', ...）
            key = str(int(count))
            if key not in grouped_by_count:
                grouped_by_count[key] = []
            # 重複を避けて追加
            if clustering_id not in grouped_by_count[key]:
                grouped_by_count[key].append(clustering_id)

            available_counts_set.add(int(count))

        # 利用可能な回数をソートしたリストに変換
        available_counts = sorted(list(available_counts_set))
        
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "success",
                "data": {
                    "available_counts": available_counts,
                    "image_counts": image_counts,
                    "grouped_image_ids": grouped_by_count
                }
            }
        )
        
    except Exception as e:
        print(f"❌ get_clustering_counts処理中にエラー: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": f"Internal server error: {str(e)}",
                "data": None
            }
        )