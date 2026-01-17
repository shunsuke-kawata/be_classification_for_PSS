import copy
import json
import math
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
    MAJOR_SHAPES,
    TFIDF_SCORE_THRESHOLDS
)
from clustering.clustering_manager import ChromaDBManager, InitClusteringManager
from clustering.mongo_db_manager import MongoDBManager
from clustering.mongo_result_manager import ResultManager
from clustering.chroma_db_manager import ChromaDBManager
from clustering.embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from clustering.utils import Utils
from clustering.word_analysis import WordAnalyzer
from clustering.continuous_clustering_reporter import ContinuousClusteringReporter

#分割したエンドポイントの作成
#ログイン操作
action_endpoint = APIRouter()

def add_parent_ids_hierarchical(clustering_dict: dict, parent_id: str = None) -> dict:
    """
    全ての要素にparent_idを追加する再帰関数（階層分類用）
    InitClusteringManager._add_parent_ids()と同じロジック
    
    Args:
        clustering_dict: クラスタリング結果の辞書
        parent_id: 親要素のID（最上位階層の場合はNone）
        
    Returns:
        parent_idが追加された辞書
    """
    result = {}
    
    for key, value in clustering_dict.items():
        # 値が辞書の場合のみ処理
        if isinstance(value, dict):
            # 現在の要素のコピーを作成
            new_value = value.copy()
            
            # parent_idを追加
            new_value['parent_id'] = parent_id
            
            # dataフィールドがある場合、再帰的に処理
            if 'data' in new_value and isinstance(new_value['data'], dict):
                new_value['data'] = add_parent_ids_hierarchical(new_value['data'], key)
            
            result[key] = new_value
        else:
            # 文字列やその他の値の場合はそのまま
            result[key] = value
    
    return result

@action_endpoint.get("/action/clustering/result/{mongo_result_id}",tags=["action"],description="初期クラスタリング結果を取得する")
def get_clustering_result(mongo_result_id:str):
    print(f"🔍 get_clustering_result called with mongo_result_id: {mongo_result_id}")
    
    result_manager = ResultManager(mongo_result_id)
    
    # ResultManagerのget_result()メソッドを使用
    result_data = result_manager.get_result()
    
    if result_data:
        print(f"✅ Found result data for mongo_result_id: {mongo_result_id}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "success", "result": result_data}
        )
    else:
        print(f"❌ Result not found for mongo_result_id: {mongo_result_id}")
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
        
        # 6. コピー元のexecuted_clustering_countをコピー先に反映
        # clustering_idで画像を紐付けて、コピー元と同じexecuted_clustering_countを設定
        _, _ = action_queries.copy_clustering_states_by_clustering_id(connect_session, source_user_id, target_user_id, project_id)
        
        print(f"✅ ユーザー{source_user_id}のデータをユーザー{target_user_id}にコピー完了（executed_clustering_countも含む）")
        
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
    use_hierarchical: bool = False,
    background_tasks: BackgroundTasks = None
):
    # エンドポイント呼び出し時のトグル値を出力
    print(f"\n{'='*80}")
    print(f"🔔 execute_init_clustering エンドポイント呼び出し")
    print(f"  - project_id: {project_id}")
    print(f"  - user_id: {user_id}")
    print(f"  - use_hierarchical: {use_hierarchical}")
    print(f"{'='*80}\n")
    
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
    def run_clustering(cid_dict: dict, sid_dict: dict, iid_dict: dict, project_id: int, original_images_folder_path: str, use_hierarchical: bool = False):
        try:
            # トグルの値を出力
            print(f"🔄 use_hierarchical = {use_hierarchical}")
            
            # クラスタリングに使用する画像情報を出力
            print(f"\n📸 クラスタリング対象画像情報:")
            print(f"  - 画像数: {len(cid_dict)}")
            print(f"  - Sentence ID数: {len(sid_dict)}")
            print(f"  - Image ID数: {len(iid_dict)}")
            print(f"\n📋 Clustering ID リスト (最初の10件):")
            for i, (clustering_id, info) in enumerate(list(cid_dict.items())[:10]):
                print(f"  [{i+1}] {clustering_id}")
                print(f"      -> sentence_id: {info.get('sentence_id')}")
                print(f"      -> image_id: {info.get('image_id')}")
            if len(cid_dict) > 10:
                print(f"  ... 他 {len(cid_dict) - 10} 件")
            print()
            
            # プロジェクト名を取得
            project_result, _ = action_queries.get_project_name(connect_session, project_id)
            project_mapping = project_result.mappings().first() if project_result else None
            project_name = project_mapping['name'] if project_mapping else f"Project_{project_id}"
            
            print(f"🏷️ プロジェクト名を取得: {project_name} (project_id: {project_id})")
            
            # 通常のクラスタリング処理を実行
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
            
            # トグルの値に応じてクラスタリングメソッドを選択
            if use_hierarchical:
                print(f"\n🔄 use_hierarchical = True: clustering_dummy()を実行します\n")
                result_dict, all_nodes = cl_module.clustering_dummy(
                    sentence_name_db_data=sentence_data,
                    image_db_data=cl_module.image_db.get_data_by_ids(target_image_ids),
                    clustering_id_dict=cid_dict,
                    sentence_id_dict=sid_dict,
                    image_id_dict=iid_dict,
                    cluster_num=cluster_num,
                    overall_folder_name=project_name,
                    output_folder=True,
                    output_json=True
                )
            else:
                print(f"\n🔄 use_hierarchical = False: clustering()を実行します\n")
                result_dict, all_nodes = cl_module.clustering(
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
            
            # MongoDBを更新（ダミー・通常両方で実行）
            print(f"\n💾 MongoDBを更新:")
            print(f"  - mongo_result_id: {mongo_result_id}")
            print(f"  - result_dict keys: {list(result_dict.keys())[:5]}...")
            print(f"  - all_nodes_dict size: {len(all_nodes_dict)}")
            
            result_manager = ResultManager(mongo_result_id)
            result_manager.update_result(result_dict, all_nodes_dict)
            
            print(f"✅ MongoDB更新完了")
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
            _, _ = action_queries.update_init_state(connect_session, user_id, project_id, clustering_state)
                
    # 非同期実行
    background_tasks.add_task(run_clustering, by_clustering_id, by_chromadb_sentence_id, by_chromadb_image_id, project_id, original_images_folder_path, use_hierarchical)
    #実行中に変更
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
            
            # プロジェクト情報を取得（レポート用）
            project_result, _ = action_queries.get_project_name(connect_session, project_id)
            project_info = project_result.mappings().first() if project_result else None
            project_name = project_info['name'] if project_info else f"project_{project_id}"
            
            # ユーザー情報を取得（レポート用）
            user_result, _ = action_queries.get_user_info(connect_session, user_id)
            user_info = user_result.mappings().first() if user_result else None
            user_name = user_info['name'] if user_info else f"user_{user_id}"
            
            # レポーター初期化
            reporter = ContinuousClusteringReporter(
                project_name=project_name,
                user_name=user_name,
                output_base_dir=DEFAULT_OUTPUT_PATH
            )
            
            # 実行時刻を記録
            from datetime import datetime
            execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 全画像のレポートデータを格納するリスト
            all_reports_data = []
            
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
            
            # 類似度閾値を定義（レポート生成でも使用）
            SIMILARITY_THRESHOLD = 0.4  # 類似度閾値（調整可能）
            
            # 各未クラスタリング画像を処理
            for idx, row in enumerate(unclustered_rows, 1):
                try:
                    image_id = row['image_id']
                    image_name = row['image_name']
                    clustering_id = row['clustering_id']
                    chromadb_sentence_id = row['chromadb_sentence_id']
                    chromadb_image_id = row['chromadb_image_id']
                    caption = row.get('caption', '')
                    
                    print(f"\n  [{idx}/{len(unclustered_rows)}] 処理中: {image_name} (ID: {image_id})")
                    
                    # レポートデータを初期化
                    report_data = {
                        'execution_time': execution_time,
                        'project_name': project_name,
                        'user_name': user_name,
                        'clustering_count': new_count,
                        'image_id': image_id,
                        'image_name': image_name,
                        'clustering_id': clustering_id,
                        'chromadb_sentence_id': chromadb_sentence_id,
                        'chromadb_image_id': chromadb_image_id,
                        'caption': caption,
                        'sentence_embedding_available': False,
                        'image_embedding_available': False,
                        'total_folders_checked': len(leaf_folders),
                        'similarity_scores': [],
                        'errors': [],
                        'new_folder_created': False,
                        'classification_criteria_used': False,
                        'additional_info': {},
                        'feature_analysis': {},
                        'sibling_folders_info': {},
                        'processing_steps': [],  # 処理ステップの記録
                        'decision_step': None,   # 最終決定ステップ
                        'decision_reason': None  # 決定理由
                    }
                    
                    # ChromaDBから文章の埋め込みベクトルを取得
                    new_sentence_embedding = None
                    try:
                        new_sentence_data = sentence_name_db.get_data_by_ids([chromadb_sentence_id])
                        new_sentence_embedding = new_sentence_data['embeddings'][0]
                        report_data['sentence_embedding_available'] = True
                    except Exception as e:
                        print(f"    ⚠️ 文章埋め込みベクトル取得エラー: {e}")
                        report_data['errors'].append(f"文章埋め込みベクトル取得エラー: {str(e)}")
                    
                    # ChromaDBから画像の埋め込みベクトルを取得
                    new_image_embedding = None
                    try:
                        new_image_data = image_db.get_data_by_ids([chromadb_image_id])
                        new_image_embedding = new_image_data['embeddings'][0]
                        report_data['image_embedding_available'] = True
                    except Exception as e:
                        print(f"    ⚠️ 画像埋め込みベクトル取得エラー: {e}")
                        report_data['errors'].append(f"画像埋め込みベクトル取得エラー: {str(e)}")
                    
                    # 両方のベクトルが取得できなかった場合はスキップ
                    if new_sentence_embedding is None and new_image_embedding is None:
                        print(f"    ⚠️ 埋め込みベクトルの取得に失敗しました")
                        continue
                    
                    # 各フォルダとの類似度を計算（文章と画像の両方）
                    max_similarity = -1
                    best_folder_id = None
                    best_similarity_type = None  # 'sentence' or 'image'
                    all_similarity_scores = []  # 全フォルダとの類似度を記録
                    
                    # 文章ベクトルで類似度計算
                    if new_sentence_embedding is not None:
                        for folder_id, folder_embedding in folder_sentence_embeddings.items():
                            similarity = cosine_similarity(
                                [new_sentence_embedding],
                                [folder_embedding]
                            )[0][0]
                            
                            # フォルダ名を取得
                            folder_obj = next((f for f in leaf_folders if f['id'] == folder_id), None)
                            folder_name = folder_obj['name'] if folder_obj else folder_id
                            
                            all_similarity_scores.append({
                                'folder_id': folder_id,
                                'folder_name': folder_name,
                                'similarity': float(similarity),
                                'type': 'sentence'
                            })
                            
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
                            
                            # フォルダ名を取得
                            folder_obj = next((f for f in leaf_folders if f['id'] == folder_id), None)
                            folder_name = folder_obj['name'] if folder_obj else folder_id
                            
                            all_similarity_scores.append({
                                'folder_id': folder_id,
                                'folder_name': folder_name,
                                'similarity': float(similarity),
                                'type': 'image'
                            })
                            
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_folder_id = folder_id
                                best_similarity_type = 'image'
                    
                    # 類似度スコアをソートしてレポートデータに保存
                    all_similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
                    report_data['similarity_scores'] = all_similarity_scores
                    
                    if best_folder_id is None:
                        print(f"    ⚠️ 適切なフォルダが見つかりませんでした")
                        report_data['errors'].append("適切なフォルダが見つかりませんでした")
                        report_data['decision_step'] = 'NO_FOLDER_FOUND'
                        report_data['decision_reason'] = '類似度計算で適切なフォルダが見つかりませんでした'
                        
                        # レポート生成
                        try:
                            reporter.generate_image_report(report_data)
                        except Exception as report_e:
                            print(f"    ⚠️ レポート生成エラー: {report_e}")
                            traceback.print_exc()
                        
                        # レポートデータをリストに追加
                        all_reports_data.append(report_data)
                        continue
                    
                    print(f"    📊 最高類似度: {max_similarity:.4f} (タイプ: {best_similarity_type})")
                    
                    # 類似度閾値チェック：閾値を下回る場合は新しいフォルダを作成
                    report_data['similarity_threshold'] = SIMILARITY_THRESHOLD
                    
                    if max_similarity < SIMILARITY_THRESHOLD:
                        print(f"    ⚠️ 最高類似度 {max_similarity:.4f} が閾値 {SIMILARITY_THRESHOLD} を下回っています")
                        print(f"    🆕 新しいリーフフォルダを作成します...")
                        report_data['processing_steps'].append(f'類似度閾値チェック: {max_similarity:.4f} < {SIMILARITY_THRESHOLD}')
                        report_data['decision_step'] = 'NEW_FOLDER_CREATION'
                        report_data['decision_reason'] = f'最高類似度({max_similarity:.4f})が閾値({SIMILARITY_THRESHOLD})を下回ったため新規フォルダを作成'
                        
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
                            
                            # レポートデータに新規フォルダ作成情報を記録
                            report_data['new_folder_created'] = True
                            report_data['new_folder_name'] = new_folder_name
                            report_data['new_folder_id'] = new_folder_id
                            report_data['final_folder_name'] = new_folder_name
                            report_data['final_folder_id'] = new_folder_id
                            report_data['final_similarity'] = max_similarity
                            report_data['final_similarity_type'] = best_similarity_type
                            
                            # レポート生成
                            try:
                                reporter.generate_image_report(report_data)
                            except Exception as report_e:
                                print(f"    ⚠️ レポート生成エラー: {report_e}")
                                traceback.print_exc()
                            
                            # レポートデータをリストに追加
                            all_reports_data.append(report_data)
                            
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
                            report_data['errors'].append(f"新規フォルダ作成失敗: {create_result.get('error', 'Unknown error')}")
                            # エラーの場合は既存フォルダへの配置処理に進む（レポートは後で生成）
                    
                    best_folder = next((f for f in leaf_folders if f['id'] == best_folder_id), None)
                    folder_name = best_folder['name'] if best_folder else best_folder_id
                    
                    print(f"    🎯 最も類似したフォルダ: {folder_name} (類似度: {max_similarity:.4f})")
                    
                    # --- 分類基準を使った振り分けロジック ---
                    # 後で使用するための変数を初期化
                    classification_criteria = {}
                    classification_words_found = []
                    target_folder_id_by_criteria = None
                    sibling_leaf_folders = []  # 初期化して未定義エラーを防止
                    
                    # --- 指定したフォルダと同じ階層にあるフォルダを取得 ---
                    try:
                        # all_nodesから指定フォルダ（best_folder）の情報を取得
                        all_nodes = result_manager.get_all_nodes()
                        best_node = all_nodes.get(best_folder_id) if all_nodes else None
                        
                        if not best_node:
                            print(f"    ⚠️ 指定フォルダ {best_folder_id} がall_nodesに見つかりません")
                            # best_folder_idのみを含むリストとして扱う
                            sibling_leaf_folders = [best_folder] if best_folder else []
                        else:
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
                            
                            print(f"    📂 同階層フォルダ: {len(sibling_folders)}個")
                            
                            # --- is_leafフォルダのキャプション一覧を取得 ---
                            sibling_leaf_folders = [f for f in sibling_folders if f['is_leaf']]
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
                            
                            # ストップワードを準備
                            stopwords_set = set(CAPTION_STOPWORDS)
                            
                            # folder_captions_mapが空の場合は処理をスキップ
                            if len(folder_captions_map) == 0:
                                print(f"    ⚠️ キャプションが取得できませんでした。フォルダ特徴分析をスキップします")
                            else:
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
                                            
                                            # 文の位置による重み（1文目: 1.0, 2文目: 0.85, 3文目: 0.7、それ以降: 0.6）
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
                                
                                # folder_word_countersが空の場合のチェック
                                if len(folder_word_counters) == 0:
                                    print(f"    ⚠️ 単語カウンターが空です。分析をスキップします")
                                else:
                                    # 各フォルダの特徴的な単語を抽出（改善版: フォルダ代表性スコア）
                                    folder_unique_words = {}
                                    TOP_N_UNIQUE_WORDS = 10  # 各フォルダから上位N個の特徴的な単語を抽出
                                    
                                    # === グローバル統計の計算 ===
                                    num_folders = len(folder_word_counters)
                                    
                                    # 各単語が何個のフォルダに出現するか
                                    word_folder_count = {}
                                    # 各単語の全フォルダでの総出現回数
                                    word_total_count = {}
                                    
                                    for counter in folder_word_counters.values():
                                        for word, count in counter.items():
                                            if word not in word_folder_count:
                                                word_folder_count[word] = 0
                                                word_total_count[word] = 0.0
                                            word_folder_count[word] += 1
                                            word_total_count[word] += count
                                    
                                    # === 各フォルダの単語スコアを計算 ===
                                    for target_folder_id, target_counter in folder_word_counters.items():
                                        # このフォルダの総画像数を取得
                                        folder_data_result = result_manager.get_folder_data_from_result(target_folder_id)
                                        total_images_in_folder = len(folder_data_result['data']) if folder_data_result['success'] else 1
                                        
                                        # folder_captions_mapに存在しない場合はスキップ
                                        if target_folder_id not in folder_captions_map:
                                            print(f"  ⚠️ フォルダ {target_folder_id} のキャプションが見つかりません。スキップします。")
                                            continue
                                        
                                        # このフォルダ内で単語を含む画像数をカウント（一貫性計算用）
                                        # 重み付きカウントではなく、純粋な画像数
                                        word_image_count = {}
                                        for caption in folder_captions_map[target_folder_id]['captions']:
                                            words_in_caption = set(re.findall(r'\b[a-z]+\b', caption.lower())) - stopwords_set
                                            for word in words_in_caption:
                                                word_image_count[word] = word_image_count.get(word, 0) + 1
                                        
                                        word_scores = {}
                                        
                                        for word, count_in_target in target_counter.items():
                                            # === 指標1: TF（Term Frequency）- 文位置重み付き ===
                                            tf = count_in_target / max(total_images_in_folder, 1)
                                            
                                            # === 指標2: フォルダ集中度（Concentration） ===
                                            # この単語の全体出現のうち、このフォルダに何%集中しているか
                                            concentration = count_in_target / max(word_total_count.get(word, 1), 0.001)
                                            
                                            # === 指標3: フォルダ内一貫性（Consistency） ===
                                            # フォルダ内の何%の画像にこの単語が出現するか
                                            num_images_with_word = word_image_count.get(word, 0)
                                            consistency = num_images_with_word / max(total_images_in_folder, 1)
                                            
                                            # === 指標4: グローバル希少性（IDF） ===
                                            num_folders_with_word = word_folder_count.get(word, 1)
                                            base_idf = math.log((num_folders + 1) / (num_folders_with_word + 1))
                                            
                                            # === 代表性スコア（Representativeness Score） ===
                                            # フォルダの内容を表す単語
                                            score_repr = tf * concentration * (consistency ** 0.5) * 1000
                                            
                                            # === 識別性スコア（Distinctiveness Score） ===
                                            # 他フォルダと区別する単語
                                            score_dist = tf * base_idf * concentration * 100
                                            
                                            # === 最終スコア: 代表性70% + 識別性30% ===
                                            final_score = 0.7 * score_repr + 0.3 * score_dist
                                            
                                            word_scores[word] = {
                                                'score': final_score,
                                                'score_repr': score_repr,
                                                'score_dist': score_dist,
                                                'tf': tf,
                                                'concentration': concentration,
                                                'consistency': consistency,
                                                'base_idf': base_idf,
                                                'count_in_folder': count_in_target,
                                                'num_images_with_word': num_images_with_word,
                                                'num_folders_with_word': num_folders_with_word,
                                                'total_count_all_folders': word_total_count.get(word, 0),
                                                'total_images': total_images_in_folder
                                            }
                                        
                                        # スコア順にソート
                                        sorted_words = sorted(
                                            word_scores.items(), 
                                            key=lambda x: x[1]['score'], 
                                            reverse=True
                                        )
                                        
                                        # 上位N個を取得
                                        top_unique = sorted_words[:TOP_N_UNIQUE_WORDS]
                                        
                                        # folder_captions_mapに存在しない場合はデフォルト値を使用
                                        folder_display_name = folder_captions_map.get(target_folder_id, {}).get('folder_name', str(target_folder_id))
                                        
                                        # 各単語の上位語を取得
                                        from nltk.corpus import wordnet as wn
                                        
                                        unique_words_with_hypernyms = []
                                        for word, info in top_unique:
                                            # WordNetから上位語を取得
                                            hypernym = 'N/A'
                                            try:
                                                synsets = wn.synsets(word)
                                                if synsets:
                                                    # 最初のsynsetの最も一般的な上位語を取得
                                                    hypernyms = synsets[0].hypernyms()
                                                    if hypernyms:
                                                        # 最初の上位語の名前を取得（.name()から単語部分のみ抽出）
                                                        hypernym = hypernyms[0].name().split('.')[0]
                                            except Exception as e:
                                                print(f"      ⚠️ '{word}'の上位語取得エラー: {e}")
                                            
                                            unique_words_with_hypernyms.append({
                                                'word': word,
                                                'hypernym': hypernym,
                                                'score': round(info['score'], 2),
                                                'score_repr': round(info['score_repr'], 2),
                                                'score_dist': round(info['score_dist'], 2),
                                                'tf': round(info['tf'], 4),
                                                'concentration': round(info['concentration'], 4),
                                                'consistency': round(info['consistency'], 4),
                                                'base_idf': round(info['base_idf'], 4),
                                                'count_in_folder': info['count_in_folder'],
                                                'num_images_with_word': info['num_images_with_word'],
                                                'num_folders_with_word': info['num_folders_with_word'],
                                                'total_count_all_folders': info['total_count_all_folders'],
                                                'total_images': info['total_images']
                                            })
                                        
                                        folder_unique_words[target_folder_id] = {
                                            'folder_name': folder_display_name,
                                            'unique_words': unique_words_with_hypernyms
                                        }
                                    
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
                                        
                                        if len(common_to_all_folders) > 0:
                                            print(f"\n    🔍 全フォルダ共通単語: {len(common_to_all_folders)}個を除外")
                                            
                                            # 各フォルダのトップ10単語から共通単語を除外
                                            for folder_id in folder_top_words_list.keys():
                                                folder_top_words_list[folder_id] = [
                                                    w for w in folder_top_words_list[folder_id] 
                                                    if w not in common_to_all_folders
                                                ]
                                    
                                    # WordAnalyzerを初期化（既存のWordNetメソッドを使用）
                                    from sentence_transformers import SentenceTransformer
                                    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                                    word_analyzer = WordAnalyzer(embedding_model)
                                    
                                    # --- 全フォルダで同じカテゴリを持つ単語のみを抽出 ---
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
                                    from collections import defaultdict
                                    category_occurrence = defaultdict(lambda: {
                                        'folders': set(),
                                        'words_by_folder': defaultdict(list),
                                        'word_category_scores': defaultdict(list)
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
                                            common_categories_across_all_folders[category] = info
                                    
                                    print(f"\n    📊 共通カテゴリ: {len(common_categories_across_all_folders)}個")
                                    
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
                                        
                                        for rank, (category, info) in enumerate(sorted_categories[:5], 1):
                                            classification_criteria[category] = {
                                                'rank': rank,
                                                'category': category,
                                                'words': [w for w, s in info['words_with_scores']],
                                                'words_with_scores': info['words_with_scores'],
                                                'word_count': info['word_count'],
                                                'avg_score': round(info['avg_score'], 2),
                                                'folders': sorted([folder_unique_words[fid]['folder_name'] for fid in info['folders']])
                                            }
                                        
                                        # 最も支配的なカテゴリを分類基準として特定
                                        if len(sorted_categories) > 0:
                                            top_category = sorted_categories[0][0]
                                    else:
                                        print(f"       ⚠️ 全フォルダに共通するカテゴリが見つかりませんでした")
                            
                            # デバッグ用JSON出力データを作成
                            debug_output = {
                                'summary': {
                                    'total_captions': len(all_captions),
                                    'sibling_leaf_folder_count': len(sibling_leaf_folders),
                                    'common_categories_count': len(common_categories_across_all_folders) if 'common_categories_across_all_folders' in locals() else 0,
                                    'classification_criteria_count': len(classification_criteria)
                                },
                                'classification_criteria': classification_criteria,
                                'folder_unique_words': folder_unique_words
                            }
                            
                            # JSON形式で出力（デバッグ用 - コメントアウト）
                            # import json
                            # from datetime import datetime
                            # 
                            # # JSONファイルに保存
                            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # json_filename = f"sibling_captions_analysis_{timestamp}.json"
                            # json_filepath = os.path.join("./", json_filename)
                            # 
                            # try:
                            #     with open(json_filepath, 'w', encoding='utf-8') as f:
                            #         json.dump(debug_output, f, indent=2, ensure_ascii=False)
                            # except Exception as json_e:
                            #     print(f"    ⚠️ JSON保存エラー: {json_e}")
                            
                            # 新規画像のキャプションを取得
                            try:
                                new_image_caption_res, _ = images_queries.select_caption_by_clustering_id(connect_session, clustering_id)
                                new_image_caption = None
                                if new_image_caption_res:
                                    caption_row = new_image_caption_res.mappings().first()
                                    if caption_row and 'caption' in caption_row and caption_row['caption']:
                                        new_image_caption = caption_row['caption'].lower()
                                
                                if new_image_caption:
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
                                        
                                        # 新規画像のキャプションから単語を抽出
                                        new_image_words = set(re.findall(r'\b[a-z]+\b', new_image_caption))
                                        # ストップワードを除外
                                        new_image_words = new_image_words - stopwords_set
                                        
                                        # 除外する単語セット: ストップワード + 全フォルダ共通単語
                                        exclude_words_for_matching = stopwords_set.copy()
                                        common_words_count = 0
                                        if 'common_to_all_folders' in locals() and len(common_to_all_folders) > 0:
                                            exclude_words_for_matching.update(common_to_all_folders)
                                            common_words_count = len(common_to_all_folders)
                                        
                                        # Step 0: 高スコア単語による即座のフォルダ決定
                                        high_score_matches = []
                                        
                                        # 画像類似度と文章類似度の閾値設定
                                        IMAGE_SIMILARITY_THRESHOLD = 0.85
                                        SENTENCE_SIMILARITY_THRESHOLD = 0.75
                                        
                                        # 各フォルダの高スコア単語（閾値以上）をチェック
                                        for sib_folder in sibling_leaf_folders:
                                            sib_folder_id = sib_folder['id']
                                            sib_folder_name = sib_folder['name']
                                            
                                            if sib_folder_id not in folder_unique_words:
                                                continue
                                            
                                            folder_high_score_words = []
                                            # このフォルダの高スコア単語を抽出
                                            for w_info in folder_unique_words[sib_folder_id]['unique_words']:
                                                if w_info['score'] >= TFIDF_SCORE_THRESHOLDS['high']:
                                                    folder_high_score_words.append(w_info)
                                            
                                            if len(folder_high_score_words) == 0:
                                                continue
                                            
                                            # キャプション内の単語と照合
                                            matched_high_score_words = []
                                            for w_info in folder_high_score_words:
                                                if w_info['word'] in new_image_words and w_info['word'] not in exclude_words_for_matching:
                                                    matched_high_score_words.append(w_info)
                                            
                                            if len(matched_high_score_words) > 0:
                                                # 最高スコアの単語を選択
                                                best_match = max(matched_high_score_words, key=lambda x: x['score'])
                                                
                                                # フォルダ内のclustering_idを取得
                                                folder_data_result = result_manager.get_folder_data_from_result(sib_folder_id)
                                                if not folder_data_result['success']:
                                                    print(f"             ⚠️ フォルダデータ取得失敗")
                                                    continue
                                                
                                                folder_data = folder_data_result['data']
                                                clustering_ids_in_folder = list(folder_data.keys())
                                                
                                                # 各画像との類似度を計算
                                                image_similarities = []
                                                for cid in clustering_ids_in_folder:
                                                    try:
                                                        # clustering_idから画像IDと文章IDを取得
                                                        img_result, _ = action_queries.get_chromadb_image_id_by_clustering_id(connect_session, cid, project_id)
                                                        sent_result, _ = action_queries.get_chromadb_sentence_id_by_clustering_id(connect_session, cid, project_id)
                                                        
                                                        if not img_result or not sent_result:
                                                            continue
                                                        
                                                        img_mapping = img_result.mappings().first()
                                                        sent_mapping = sent_result.mappings().first()
                                                        
                                                        if not img_mapping or not sent_mapping:
                                                            continue
                                                        
                                                        folder_image_id = img_mapping['chromadb_image_id']
                                                        folder_sentence_id = sent_mapping['chromadb_sentence_id']
                                                        
                                                        # 画像埋め込みベクトルを取得
                                                        folder_img_data = image_db.get_data_by_ids([folder_image_id])
                                                        folder_img_embedding = folder_img_data['embeddings'][0] if folder_img_data['embeddings'] else None
                                                        
                                                        # 文章埋め込みベクトルを取得
                                                        folder_sent_data = sentence_name_db.get_data_by_ids([folder_sentence_id])
                                                        folder_sent_embedding = folder_sent_data['embeddings'][0] if folder_sent_data['embeddings'] else None
                                                        
                                                        if folder_img_embedding is None or folder_sent_embedding is None:
                                                            continue
                                                        
                                                        if new_image_embedding is None or new_sentence_embedding is None:
                                                            continue
                                                        
                                                        # 画像類似度を計算
                                                        img_sim = float(np.dot(new_image_embedding, folder_img_embedding) / (
                                                            np.linalg.norm(new_image_embedding) * np.linalg.norm(folder_img_embedding)
                                                        ))
                                                        
                                                        # 文章類似度を計算
                                                        sent_sim = float(np.dot(new_sentence_embedding, folder_sent_embedding) / (
                                                            np.linalg.norm(new_sentence_embedding) * np.linalg.norm(folder_sent_embedding)
                                                        ))
                                                        
                                                        image_similarities.append({
                                                            'clustering_id': cid,
                                                            'image_similarity': img_sim,
                                                            'sentence_similarity': sent_sim
                                                        })
                                                        
                                                    except Exception as sim_e:
                                                        continue
                                                
                                                # 画像類似度でソート（降順）
                                                image_similarities.sort(key=lambda x: x['image_similarity'], reverse=True)
                                                
                                                # 画像類似度が高い順に、両方の閾値をクリアするものを探す
                                                best_matching_image = None
                                                for sim_info in image_similarities:
                                                    if (sim_info['image_similarity'] >= IMAGE_SIMILARITY_THRESHOLD and 
                                                        sim_info['sentence_similarity'] >= SENTENCE_SIMILARITY_THRESHOLD):
                                                        best_matching_image = sim_info
                                                        break
                                                
                                                # 両方の閾値をクリアする画像が見つかった場合のみ、このフォルダをマッチ候補に追加
                                                if best_matching_image is not None:
                                                    high_score_matches.append({
                                                        'folder': sib_folder,
                                                        'matched_words': matched_high_score_words,
                                                        'best_word': best_match['word'],
                                                        'best_score': best_match['score'],
                                                        'image_similarity': best_matching_image['image_similarity'],
                                                        'sentence_similarity': best_matching_image['sentence_similarity'],
                                                        'matching_clustering_id': best_matching_image['clustering_id']
                                                    })
                                        
                                        # 高スコアマッチが見つかった場合、即座にフォルダを決定
                                        if len(high_score_matches) > 0:
                                            # 複数マッチがある場合の選択ロジック
                                            if len(high_score_matches) == 1:
                                                best_match = high_score_matches[0]
                                            else:
                                                # 複数ある場合、画像類似度が最も高いものを選択
                                                best_match = max(high_score_matches, key=lambda x: x['image_similarity'])
                                            
                                            print(f"\n       ⭐ Step 0: 高スコアマッチ '{best_match['best_word']}' → '{best_match['folder']['name']}'")
                                            
                                            # 処理ステップを記録
                                            report_data['processing_steps'].append('Step 0: 高スコア単語による即座のフォルダ決定')
                                            report_data['decision_step'] = 'STEP_0_HIGH_SCORE_MATCH'
                                            report_data['decision_reason'] = f"単語'{best_match['best_word']}'が高スコア({best_match['best_score']:.2f})でフォルダ'{best_match['folder']['name']}'にマッチ"
                                            report_data['matched_word'] = best_match['best_word']
                                            report_data['matched_score'] = best_match['best_score']
                                            
                                            # 既存フォルダに挿入
                                            target_folder_id_by_criteria = best_match['folder']['id']
                                        
                                        # 高スコアマッチがない場合のみ、従来のカテゴリマッチングを実行
                                        if len(high_score_matches) == 0:
                                        
                                            # Step 1: 新規画像のキャプション内の単語から、分類基準カテゴリ(top_category)に属する単語をフィルタリング
                                            category_words_in_caption = []
                                            excluded_by_filter = []
                                            checked_but_not_matched = []
                                            word_matching_details = []
                                            for new_word in sorted(list(new_image_words)):
                                                if new_word in exclude_words_for_matching:
                                                    excluded_by_filter.append(new_word)
                                                    continue
                                                
                                                max_score_for_word = -1
                                                belongs_to_category = False
                                                best_match_word = None
                                                match_details = []
                                                
                                                for category_word, cat_word_score in top_category_words_with_scores:
                                                    common_categories, category_score = word_analyzer.get_common_category(new_word, category_word)
                                                    
                                                    if len(common_categories) > 0:
                                                        match_details.append({
                                                            'category_word': category_word,
                                                            'common_categories': common_categories,
                                                            'category_score': category_score,
                                                            'matched': common_categories[0] == top_category and category_score >= 3.0
                                                        })
                                                    
                                                    if len(common_categories) > 0 and common_categories[0] == top_category and category_score >= 3.0:
                                                        belongs_to_category = True
                                                        if category_score > max_score_for_word:
                                                            max_score_for_word = category_score
                                                            best_match_word = category_word
                                                
                                                word_matching_details.append({
                                                    'word': new_word,
                                                    'matched': belongs_to_category,
                                                    'max_score': max_score_for_word,
                                                    'best_match_word': best_match_word,
                                                    'match_details': match_details
                                                })
                                                
                                                if belongs_to_category:
                                                    category_words_in_caption.append((new_word, max_score_for_word))
                                                else:
                                                    checked_but_not_matched.append(new_word)
                                            
                                            # スコア順にソート
                                            category_words_in_caption.sort(key=lambda x: x[1], reverse=True)
                                            
                                            print(f"\n       ⭐ Step 1: {len(category_words_in_caption)}個のカテゴリ単語を抽出")
                                            report_data['processing_steps'].append(f'Step 1: カテゴリ\'{top_category}\'に属する単語を{len(category_words_in_caption)}個抽出')
                                            report_data['processing_steps'].append(f'Step 1: カテゴリ\'{top_category}\'に属する単語を{len(category_words_in_caption)}個抽出')
                                            
                                            if len(category_words_in_caption) == 0:
                                                print(f"       ⚠️ カテゴリ '{top_category}' に属する単語が見つかりませんでした")
                                            
                                            # Step 2: フィルタリングした単語と既存フォルダの特徴的な単語を照合
                                            if len(category_words_in_caption) > 0:
                                                print(f"\n       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                                print(f"       🔍 Step 2: 既存フォルダの特徴的な単語と照合...")
                                                print(f"          📋 照合対象の単語: {[w for w, s in category_words_in_caption]}")
                                                print(f"          📂 照合対象のフォルダ数: {len(sibling_leaf_folders)}個")
                                                print(f"\n       🔎 フォルダ照合の詳細:")
                                                
                                                folder_candidates = []
                                                words_checked_count = {}
                                                folders_checked_for_word = {}
                                                
                                                # フィルタリングした各単語について、既存フォルダの特徴的な単語に含まれるか確認
                                                for new_word, word_category_score in category_words_in_caption:
                                                    words_checked_count[new_word] = 0
                                                    folders_checked_for_word[new_word] = []
                                                    print(f"\n          🔍 '{new_word}' を各フォルダの特徴的単語と照合中...")
                                                    # 各兄弟フォルダの特徴的な単語リストをチェック
                                                    for sib_folder in sibling_leaf_folders:
                                                        sib_folder_id = sib_folder['id']
                                                        sib_folder_name = sib_folder['name']
                                                        words_checked_count[new_word] += 1
                                                        
                                                        # このフォルダの特徴的な単語を取得
                                                        if sib_folder_id in folder_unique_words:
                                                            folder_unique_list = folder_unique_words[sib_folder_id]['unique_words']
                                                            folder_words_list = [w['word'] for w in folder_unique_list]
                                                            folders_checked_for_word[new_word].append({
                                                                'folder_name': sib_folder_name,
                                                                'folder_id': sib_folder_id,
                                                                'unique_words': folder_words_list[:5]  # 上位5個
                                                            })
                                                            
                                                            # 特徴的な単語リストに含まれているか確認
                                                            for w_info in folder_unique_list:
                                                                if w_info['word'] == new_word:
                                                                    # 一致した！
                                                                    folder_tf_idf_score = w_info['score']
                                                                    
                                                                    # 総合スコア = フォルダのTF-IDFスコア + カテゴリスコア
                                                                    combined_score = folder_tf_idf_score + (word_category_score * 0.1)
                                                                    
                                                                    folder_candidates.append({
                                                                        'folder': sib_folder,
                                                                        'word': new_word,
                                                                        'folder_score': folder_tf_idf_score,
                                                                        'category_score': word_category_score,
                                                                        'combined_score': combined_score
                                                                    })
                                                                    
                                                                    print(f"             ✅ マッチ！フォルダ '{sib_folder_name}'")
                                                                    print(f"                └─ TF-IDF: {folder_tf_idf_score:.2f}, カテゴリ: {word_category_score:.2f}, 総合: {combined_score:.2f}")
                                                                    break
                                                    
                                                    # この単語でマッチしなかったフォルダの情報を出力
                                                    if words_checked_count[new_word] == len(sibling_leaf_folders) and len([c for c in folder_candidates if c['word'] == new_word]) == 0:
                                                        print(f"             ❌ どのフォルダの特徴的単語にも含まれていません")
                                                        print(f"             📊 確認したフォルダ: {words_checked_count[new_word]}個")
                                                
                                                print(f"\n       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                                print(f"       📊 Step 2 結果サマリー:")
                                                print(f"          🎯 マッチした候補数: {len(folder_candidates)}個")
                                                report_data['processing_steps'].append(f'Step 2: 既存フォルダとの照合で{len(folder_candidates)}個の候補を発見')
                                                if len(folder_candidates) > 0:
                                                    print(f"          📋 候補リスト:")
                                                    for idx, candidate in enumerate(sorted(folder_candidates, key=lambda x: x['combined_score'], reverse=True)[:5], 1):
                                                        print(f"             {idx}. '{candidate['word']}' → '{candidate['folder']['name']}' (総合: {candidate['combined_score']:.2f})")
                                                print(f"       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                                
                                                # Step 3: 候補から最適なフォルダを選択
                                                print(f"\n       🎯 Step 3: 最適なフォルダを選択...")
                                                matched_folder = None
                                                matched_word = None
                                                
                                                if len(folder_candidates) > 0:
                                                    # 総合スコアが高い順にソート
                                                    folder_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
                                                    best_candidate = folder_candidates[0]
                                                    matched_folder = best_candidate['folder']
                                                    matched_word = best_candidate['word']
                                                    
                                                    print(f"\n       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                                    print(f"       ⭐ 最終決定:")
                                                    print(f"          🎯 選択された単語: '{matched_word}'")
                                                    print(f"          📂 挿入先フォルダ: '{matched_folder['name']}'")
                                                    print(f"          📊 スコア詳細:")
                                                    print(f"             - TF-IDFスコア: {best_candidate['folder_score']:.2f}")
                                                    print(f"             - カテゴリスコア: {best_candidate['category_score']:.2f}")
                                                    print(f"             - 総合スコア: {best_candidate['combined_score']:.2f}")
                                                    print(f"          🔢 フォルダID: {matched_folder['id']}")
                                                    print(f"       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                                                    
                                                    # 既存フォルダに挿入
                                                    target_folder_id_by_criteria = matched_folder['id']
                                                
                                                else:
                                                    # 既存フォルダにマッチする単語が見つからなかった → 新規フォルダ作成
                                                    # カテゴリに最も属する単語（最高スコア）を使用
                                                    new_folder_word = category_words_in_caption[0][0]
                                                    
                                                    print(f"\n       ℹ️ 既存フォルダに一致する単語が見つかりませんでした")
                                                    print(f"       🆕 新規フォルダ作成予定: フォルダ名='{new_folder_word}'")
                                                    
                                                    report_data['processing_steps'].append('Step 3: 既存フォルダにマッチせず、新規フォルダ作成を決定')
                                                    report_data['decision_step'] = 'STEP_3_NEW_FOLDER_REQUIRED'
                                                    report_data['decision_reason'] = f'カテゴリ単語は見つかったが、既存フォルダにマッチせず、\'{new_folder_word}\'で新規フォルダ作成'
                                                    
                                                    report_data['processing_steps'].append('Step 3: 既存フォルダにマッチせず、新規フォルダ作成を決定')
                                                    report_data['decision_step'] = 'STEP_3_NEW_FOLDER_REQUIRED'
                                                    report_data['decision_reason'] = f'カテゴリ単語は見つかったが、既存フォルダにマッチせず、\'{new_folder_word}\'で新規フォルダ作成'
                                                    
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
                                            else:
                                                print(f"       ℹ️ キャプション内に分類基準カテゴリに属する単語が含まれていません")
                                                print(f"       → 最も類似したフォルダ（{folder_name}）に挿入します")
                                        else:
                                            print(f"       ℹ️ 分類基準が生成されていません")
                                            print(f"       → 最も類似したフォルダ（{folder_name}）に挿入します")
                                    else:
                                        print(f"       ⚠️ 新規画像のキャプションが取得できませんでした")
                                        print(f"       → 最も類似したフォルダ（{folder_name}）に挿入します")
                                    
                                    # レポートデータに兄弟フォルダのTF-IDFスコア表を追加
                                    if 'folder_unique_words' in locals() and len(folder_unique_words) > 0:
                                        report_data['sibling_folder_tfidf_scores'] = folder_unique_words
                                        report_data['classification_criteria_process_executed'] = True
                                        if 'classification_criteria' in locals():
                                            report_data['classification_criteria_details'] = classification_criteria

                            except Exception as criteria_e:
                                print(f"       ⚠️ 分類基準による振り分け処理でエラー: {criteria_e}")
                                traceback.print_exc()
                                print(f"       → フォールバックとして最も類似したフォルダに挿入します")
                    except Exception as sib_e:
                        print(f"    ⚠️ 同階層フォルダ取得エラー: {sib_e}")
                        traceback.print_exc()
                    
                    # フォルダ選択がうまくいかなかった場合、全ての画像から最も類似したものを探してそのフォルダに挿入
                    if target_folder_id_by_criteria is None:
                        print(f"\n    🔍 全画像から最も類似した画像を検索中...")
                        
                        try:
                            # 兄弟フォルダが定義されている場合はそれを使用、なければ全リーフフォルダを使用
                            folders_to_check = sibling_leaf_folders if 'sibling_leaf_folders' in locals() and len(sibling_leaf_folders) > 0 else leaf_folders
                            print(f"       対象フォルダ数: {len(folders_to_check)}個 ({'兄弟フォルダのみ' if 'sibling_leaf_folders' in locals() and len(sibling_leaf_folders) > 0 else '全リーフフォルダ'})")
                            
                            max_image_similarity = -1
                            max_sentence_similarity = -1
                            best_matching_folder_id = None
                            best_matching_image_info = {}
                            all_image_similarities = []
                            
                            # 各フォルダ内の全画像と比較
                            for folder in folders_to_check:
                                folder_id = folder['id']
                                folder_name = folder['name']
                                
                                # フォルダ内のclustering_idを取得
                                folder_data_result = result_manager.get_folder_data_from_result(folder_id)
                                if not folder_data_result['success']:
                                    print(f"       ⚠️ フォルダ {folder_name} (ID: {folder_id}) のデータ取得失敗")
                                    continue
                                
                                folder_data = folder_data_result['data']
                                clustering_ids = list(folder_data.keys())
                                
                                if len(clustering_ids) == 0:
                                    print(f"       ⚠️ フォルダ {folder_name} (ID: {folder_id}) は空です")
                                    continue
                                
                                # フォルダ内の全画像と比較
                                for cid in clustering_ids:
                                    try:
                                        # chromadb_image_idとchromadb_sentence_idを取得
                                        img_result, _ = action_queries.get_chromadb_image_id_by_clustering_id(connect_session, cid, project_id)
                                        if not img_result:
                                            continue
                                        
                                        img_mapping = img_result.mappings().first()
                                        if not img_mapping:
                                            continue
                                        
                                        chromadb_img_id = img_mapping['chromadb_image_id']
                                        
                                        sent_result, _ = action_queries.get_chromadb_sentence_id_by_clustering_id(connect_session, cid, project_id)
                                        if not sent_result:
                                            continue
                                        
                                        sent_mapping = sent_result.mappings().first()
                                        if not sent_mapping:
                                            continue
                                        
                                        chromadb_sent_id = sent_mapping['chromadb_sentence_id']
                                        
                                        # ChromaDBから埋め込みベクトルを取得
                                        existing_sentence_data = sentence_db.get_data_by_ids([chromadb_sent_id])
                                        existing_sentence_embedding = existing_sentence_data['embeddings'][0]
                                        
                                        existing_image_data = image_db.get_data_by_ids([chromadb_img_id])
                                        existing_image_embedding = existing_image_data['embeddings'][0]
                                        
                                        # 文章の類似度を計算
                                        sentence_similarity = 0.0
                                        if new_sentence_embedding is not None:
                                            sentence_similarity = cosine_similarity(
                                                [new_sentence_embedding],
                                                [existing_sentence_embedding]
                                            )[0][0]
                                        
                                        # 画像の類似度を計算
                                        image_similarity = 0.0
                                        if new_image_embedding is not None:
                                            image_similarity = cosine_similarity(
                                                [new_image_embedding],
                                                [existing_image_embedding]
                                            )[0][0]
                                        
                                        # 記録用
                                        all_image_similarities.append({
                                            'folder_id': folder_id,
                                            'folder_name': folder_name,
                                            'clustering_id': cid,
                                            'sentence_similarity': float(sentence_similarity),
                                            'image_similarity': float(image_similarity)
                                        })
                                        
                                        # 画像類似度で最大値を更新
                                        if image_similarity > max_image_similarity:
                                            max_image_similarity = image_similarity
                                            max_sentence_similarity = sentence_similarity
                                            best_matching_folder_id = folder_id
                                            best_matching_image_info = {
                                                'folder_name': folder_name,
                                                'clustering_id': cid,
                                                'sentence_similarity': float(sentence_similarity),
                                                'image_similarity': float(image_similarity)
                                            }
                                    
                                    except Exception as embed_e:
                                        continue
                            
                            # 最も類似した画像が見つかった場合
                            if best_matching_folder_id is not None:
                                target_folder_id_by_criteria = best_matching_folder_id
                                
                                print(f"\n    ✅ 最も類似した画像を発見")
                                print(f"       フォルダ: {best_matching_image_info['folder_name']} (ID: {best_matching_folder_id})")
                                print(f"       画像ID: {best_matching_image_info['clustering_id']}")
                                print(f"       画像類似度: {max_image_similarity:.4f}")
                                print(f"       文章類似度: {max_sentence_similarity:.4f}")
                                
                                # レポートデータに記録
                                report_data['most_similar_image_matching_used'] = True
                                report_data['best_matching_image_similarity'] = float(max_image_similarity)
                                report_data['best_matching_sentence_similarity'] = float(max_sentence_similarity)
                                report_data['best_matching_folder_id'] = best_matching_folder_id
                                report_data['best_matching_folder_name'] = best_matching_image_info['folder_name']
                                report_data['best_matching_clustering_id'] = best_matching_image_info['clustering_id']
                                report_data['all_image_similarities'] = all_image_similarities[:100]  # 最大100件まで記録
                            else:
                                print(f"    ⚠️ 適切な画像が見つかりませんでした")
                        
                        except Exception as image_matching_e:
                            print(f"    ⚠️ 類似画像マッチングエラー: {image_matching_e}")
                            traceback.print_exc()
                    
                    # 最終的な挿入先フォルダを決定
                    final_target_folder_id = target_folder_id_by_criteria if target_folder_id_by_criteria else best_folder_id
                    
                    # 最終フォルダ情報をレポートデータに記録
                    final_folder_obj = next((f for f in leaf_folders if f['id'] == final_target_folder_id), None)
                    final_folder_name = final_folder_obj['name'] if final_folder_obj else final_target_folder_id
                    
                    report_data['final_folder_id'] = final_target_folder_id
                    report_data['final_folder_name'] = final_folder_name
                    report_data['final_similarity'] = max_similarity
                    report_data['final_similarity_type'] = best_similarity_type
                    
                    # 既存フォルダに追加する場合、フォルダ平均との類似度を計算
                    try:
                        # 文章特徴量の類似度
                        if final_target_folder_id in folder_sentence_embeddings and new_sentence_embedding is not None:
                            folder_sent_vec = folder_sentence_embeddings[final_target_folder_id]
                            new_image_sent_vec = new_sentence_embedding
                            sentence_similarity_with_folder = float(np.dot(folder_sent_vec, new_image_sent_vec) / 
                                                                          (np.linalg.norm(folder_sent_vec) * np.linalg.norm(new_image_sent_vec)))
                            report_data['folder_average_sentence_similarity'] = sentence_similarity_with_folder
                            print(f"    📊 フォルダ平均との文章類似度: {sentence_similarity_with_folder:.4f}")
                        
                        # 画像特徴量の類似度
                        if final_target_folder_id in folder_image_embeddings and new_image_embedding is not None:
                            folder_img_vec = folder_image_embeddings[final_target_folder_id]
                            new_image_img_vec = new_image_embedding
                            image_similarity_with_folder = float(np.dot(folder_img_vec, new_image_img_vec) / 
                                                                       (np.linalg.norm(folder_img_vec) * np.linalg.norm(new_image_img_vec)))
                            report_data['folder_average_image_similarity'] = image_similarity_with_folder
                            print(f"    📊 フォルダ平均との画像類似度: {image_similarity_with_folder:.4f}")
                    except Exception as sim_calc_e:
                        print(f"    ⚠️ フォルダ平均との類似度計算エラー: {sim_calc_e}")
                    
                    if target_folder_id_by_criteria:
                        report_data['classification_criteria_used'] = True
                    
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
                        
                        # レポート生成
                        try:
                            reporter.generate_image_report(report_data)
                        except Exception as report_e:
                            print(f"    ⚠️ レポート生成エラー: {report_e}")
                            traceback.print_exc()
                        
                        # レポートデータをリストに追加
                        all_reports_data.append(report_data)

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
                        report_data['errors'].append(f"画像挿入エラー: {insert_result.get('error', 'Unknown error')}")
                        
                        # レポート生成
                        try:
                            reporter.generate_image_report(report_data)
                        except Exception as report_e:
                            print(f"    ⚠️ レポート生成エラー: {report_e}")
                            traceback.print_exc()
                        
                        # レポートデータをリストに追加
                        all_reports_data.append(report_data)
                        
                except Exception as img_error:
                    print(f"    ❌ 画像処理中にエラー: {img_error}")
                    traceback.print_exc()
                    
                    # エラー時でもレポートを生成
                    if 'report_data' in locals():
                        report_data['errors'].append(f"画像処理エラー: {str(img_error)}")
                        
                        try:
                            reporter.generate_image_report(report_data)
                        except Exception as report_e:
                            print(f"    ⚠️ レポート生成エラー: {report_e}")
                            traceback.print_exc()
                        
                        all_reports_data.append(report_data)
                    
                    continue
            
            # サマリーレポート生成
            if len(all_reports_data) > 0:
                try:
                    print(f"\n📊 サマリーレポート生成中...")
                    reporter.generate_summary_report(all_reports_data)
                    print(f"✅ サマリーレポート生成完了")
                    
                    print(f"\n📊 評価指標レポート生成中...")
                    # フォルダデータを取得
                    all_nodes = result_manager.get_all_nodes()
                    reporter.generate_metrics_report(
                        all_reports_data,
                        folder_data=all_nodes,
                        similarity_threshold=SIMILARITY_THRESHOLD
                    )
                    print(f"✅ 評価指標レポート生成完了")
                except Exception as summary_e:
                    print(f"⚠️ レポート生成エラー: {summary_e}")
                    traceback.print_exc()
            
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
    

@action_endpoint.post("/action/folders/{mongo_result_id}", tags=["action"], description="新しいフォルダを作成")
def create_folder(
    mongo_result_id: str,
    parent_folder_id: str = Query(..., description="親フォルダのID"),
    is_leaf: bool = Query(..., description="リーフフォルダかどうか（True: ファイル用, False: カテゴリ用）")
):
    """
    指定された親フォルダの配下に新しいフォルダを作成する
    
    Args:
        mongo_result_id: MongoDBの結果ID
        parent_folder_id: 親フォルダのID
        is_leaf: リーフフォルダかどうか（True: ファイル用, False: カテゴリ用）
        
    Returns:
        JSONResponse: 作成されたフォルダの情報
    """
    try:
        # 入力バリデーション
        if not mongo_result_id or not mongo_result_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "mongo_result_id is required", "data": None}
            )
        
        if not parent_folder_id or not parent_folder_id.strip():
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "parent_folder_id is required", "data": None}
            )
        
        print(f"📁 フォルダ作成リクエスト:")
        print(f"   mongo_result_id: {mongo_result_id}")
        print(f"   parent_folder_id: {parent_folder_id}")
        print(f"   is_leaf: {is_leaf}")
        
        # ResultManagerを初期化
        result_manager = ResultManager(mongo_result_id)
        
        # 新しいフォルダIDを生成
        new_folder_id = Utils.generate_uuid()
        
        # フォルダ名をis_leafに応じて設定
        folder_prefix = "leaf" if is_leaf else "category"
        folder_name = f"{folder_prefix}-{new_folder_id[:8]}"
        
        print(f"   生成されたフォルダID: {new_folder_id}")
        print(f"   フォルダ名: {folder_name}")
        
        # all_nodesとresultを取得
        all_nodes = result_manager.get_all_nodes()
        result_data = result_manager.get_result()
        
        if all_nodes is None or result_data is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "result or all_nodes not found", "data": None}
            )
        
        # 親フォルダの存在を確認
        if parent_folder_id not in all_nodes:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"parent folder not found: {parent_folder_id}", "data": None}
            )
        
        # 新しいフォルダノードを作成
        new_folder_node = {
            "type": "folder",
            "id": new_folder_id,
            "name": folder_name,
            "parent_id": parent_folder_id,
            "is_leaf": is_leaf
        }
        
        # all_nodesに追加
        all_nodes[new_folder_id] = new_folder_node
        print(f"   📝 all_nodesに追加: {new_folder_id}")
        
        # resultに空のデータを追加
        new_folder_data = {
            "type": "folder",
            "name": folder_name,
            "is_leaf": is_leaf,
            "data": {}  # 空のフォルダとして作成
        }
        
        # result内で親フォルダを再帰的に探索して追加
        def add_folder_to_parent_recursive(node: dict, target_parent_id: str) -> bool:
            """
            resultを再帰的に探索して親フォルダを見つけ、新しいフォルダを追加する
            
            Args:
                node: 現在のノード
                target_parent_id: 親フォルダのID
                
            Returns:
                bool: 追加に成功したかどうか
            """
            for folder_id, folder_data in node.items():
                if folder_id == target_parent_id:
                    # 親フォルダが見つかった
                    if isinstance(folder_data, dict):
                        if "data" not in folder_data:
                            folder_data["data"] = {}
                        folder_data["data"][new_folder_id] = new_folder_data
                        print(f"   ✅ 親フォルダ {target_parent_id} に追加しました")
                        return True
                elif isinstance(folder_data, dict) and "data" in folder_data and isinstance(folder_data["data"], dict):
                    # 再帰的に探索
                    if add_folder_to_parent_recursive(folder_data["data"], target_parent_id):
                        return True
            return False
        
        # トップレベルから探索
        if not add_folder_to_parent_recursive(result_data, parent_folder_id):
            # トップレベルに親がある場合（resultの直下）
            if parent_folder_id in result_data:
                parent_data = result_data[parent_folder_id]
                if "data" not in parent_data:
                    parent_data["data"] = {}
                parent_data["data"][new_folder_id] = new_folder_data
                print(f"   ✅ トップレベルの親フォルダ {parent_folder_id} に追加しました")
            else:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"message": f"parent folder not found in result: {parent_folder_id}", "data": None}
                )
        
        # 親フォルダを辿ってresult内のデータを更新
        print(f"   🔄 親フォルダを辿ってresult更新中...")
        parent_path = result_manager.get_parents(new_folder_id)
        print(f"   📂 親パス: {parent_path}")
        
        # 更新をMongoDBに保存
        result_manager.update_result(result_data, all_nodes)
        print(f"   💾 MongoDBに保存完了")
        
        print(f"✅ フォルダ作成成功: {folder_name} (ID: {new_folder_id})")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "success",
                "data": {
                    "folder_id": new_folder_id,
                    "folder_name": folder_name,
                    "parent_id": parent_folder_id,
                    "is_leaf": is_leaf
                }
            }
        )
        
    except Exception as e:
        print(f"❌ フォルダ作成エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": f"フォルダ作成に失敗しました: {str(e)}",
                "data": None
            }
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