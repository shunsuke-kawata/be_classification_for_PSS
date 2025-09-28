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
from config import CLUSTERING_STATUS,DEFAULT_IMAGE_PATH,DEFAULT_OUTPUT_PATH
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

    if init_clustering_state == CLUSTERING_STATUS.EXECUTING or init_clustering_state ==CLUSTERING_STATUS.FINISHED:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "init clustering already started", "data": None}
        )

    # 対象画像の取得
    query_text = f"""
        SELECT clustering_id,chromadb_sentence_id,chromadb_image_id
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
        by_chromadb_image_id[iid] = {"clustering_id": cid, "sentence_id": sid}
    
    # バックグラウンド処理に渡す関数
    def run_clustering(cid_dict:dict,sid_dict:dict,iid_dict:dict,project_id:int, original_images_folder_path:str):
        try:
            cl_module = InitClusteringManager(
                sentence_db=ChromaDBManager('sentence_embeddings'),
                image_db=ChromaDBManager("image_embeddings"),
                images_folder_path=f"./{DEFAULT_IMAGE_PATH}/{original_images_folder_path}",
                output_base_path=f"./{DEFAULT_OUTPUT_PATH}/{project_id}",
            )
            
            target_sentence_ids = list(sid_dict.keys())
            target_image_ids = list(iid_dict.keys())
            embeddings = cl_module.sentence_db.get_data_by_ids(target_sentence_ids)['embeddings']
            cluster_num, _ = cl_module.get_optimal_cluster_num(embeddings=embeddings)
            result_dict,all_nodes = cl_module.clustering(
                sentence_db_data=cl_module.sentence_db.get_data_by_ids(target_sentence_ids),
                image_db_data=cl_module.image_db.get_data_by_ids(target_image_ids),
                clustering_id_dict=cid_dict,
                sentence_id_dict=sid_dict,
                image_id_dict=iid_dict,
                cluster_num=cluster_num,
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
            clustering_state = CLUSTERING_STATUS.FAILED
        else:
            clustering_state = CLUSTERING_STATUS.FINISHED
        finally:
            
            # 初期化状態を更新
            update_query = f"""
                UPDATE project_memberships
                SET init_clustering_state = '{clustering_state}'
                WHERE project_id = {project_id} AND user_id = {user_id};
            """
            _, _ = execute_query(session=connect_session, query_text=update_query)
                
    # 非同期実行
    print(original_images_folder_path)
    background_tasks.add_task(run_clustering, by_clustering_id, by_chromadb_sentence_id,by_chromadb_image_id,project_id, original_images_folder_path)
    
    # 初期化状態を更新
    update_query = f"""
        UPDATE project_memberships
        SET init_clustering_state = '{CLUSTERING_STATUS.EXECUTING}'
        WHERE project_id = {project_id} AND user_id = {user_id};
    """
    #初期化状態を更新
    _, _ = execute_query(session=connect_session, query_text=update_query)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "init clustering started in background", "data": project_id}
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
    指定されたフォルダIDリストを受け取り、IDを出力する
    
    Args:
        mongo_result_id (str): MongoDBの結果ID
        sources (List[str]): 削除対象のフォルダIDリスト
    
    Returns:
        JSONResponse: 500エラー
    """
    print(f"🗂️ delete_folders呼び出し: mongo_result_id={mongo_result_id}")
    print(f"📋 受け取ったフォルダIDリスト (sources): {sources}")
    print(f"📊 削除対象フォルダ数: {len(sources)}")
    

    result_manager = ResultManager(mongo_result_id)
    # フォルダIDの詳細を出力

    result_manager.remove_folders_from_result(sources)
        
    
    print("📝 この処理は開発中のため、IDの出力のみを行い500エラーを返します")
    
    # 500エラーを返す
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal Server Error - Folder deletion not implemented",
            "data": {
                "mongo_result_id": mongo_result_id,
                "received_folder_ids": sources,
                "folder_count": len(sources),
                "error": "Function is in development mode - only logging IDs"
            }
        }
    )