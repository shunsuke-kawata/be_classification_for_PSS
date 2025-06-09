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
from fastapi import BackgroundTasks
from collections import defaultdict

#分割したエンドポイントの作成
#ログイン操作
action_endpoint = APIRouter()

@action_endpoint.get("/action/clustering/result/{mongo_result_id}",tags=["action"],description="初期クラスタリング結果を取得する")
def get_init_clustering_result(mongo_result_id:str):
    mdb_module = MongoDBManager()
    
    result = mdb_module.find_documents(collection_name='clustering_results',query={"mongo_result_id": mongo_result_id})

    def convert_objectid(doc):
        doc["_id"] = str(doc["_id"])
        return doc
    items = result
    items = [convert_objectid(doc) for doc in items]
    
    if(len(items)==1):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "success", "data": items[0]}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "The data could not be uniquely identified.", "data": None}
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
            result_dict = cl_module.clustering(
                sentence_db_data=cl_module.sentence_db.get_data_by_ids(target_sentence_ids),
                image_db_data=cl_module.image_db.get_data_by_ids(target_image_ids),
                clustering_id_dict=cid_dict,
                sentence_id_dict=sid_dict,
                image_id_dict=iid_dict,
                cluster_num=cluster_num,
                output_folder=True,
                output_json=True
            )
            
            mongo_module = MongoDBManager()
            mongo_module.update_document(
                collection_name='clustering_results',
                query={"mongo_result_id": mongo_result_id},
                update={"mongo_result_id":mongo_result_id,"result":result_dict},
                )
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