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
    
    items_to_move = {}
    
    if not items_to_move:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "message": f"No {source_type} found to move", 
                "data": {
                    "searched_ids": sources
                }
            }
        )
        if source_type == "folders":
            print(f"   - フォルダ: {key}")
            print(f"     - is_leaf: {value.get('is_leaf')}")
            print(f"     - parent_id: {value.get('parent_id')}")
            if "data" in value:
                print(f"     - 子要素数: {len(value['data'])}")
                print(f"     - 子要素: {list(value['data'].keys())}")
        else:
            print(f"   - 画像: {key}")
    
    # 挿入後の予想状態を表示
    print(f"🔮 挿入後の予想状態:")
    new_destination_data = destination_data.copy()
    new_destination_data.update(items_to_move)
    print(f"   - 挿入後の要素数: {len(new_destination_data)}")
    print(f"   - 挿入後の要素: {list(new_destination_data.keys())}")
        
    # 移動先に同じ名前の要素が存在しないかチェック
    print(f"🔍 移動先の名前重複チェック中...")
    
    if "data" not in destination_node:
        destination_node["data"] = {}
    
    conflicting_items = []
    for key in items_to_move.keys():
        if key in destination_node["data"]:
            conflicting_items.append(key)
    
    if conflicting_items:
        print(f"❌ エラー: 移動先に同じ名前の要素が既に存在します: {conflicting_items}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": f"Items with same names already exist in destination: {conflicting_items}", "data": None}
        )
    
    print(f"✅ 名前重複チェック完了: 重複なし")
    
    # 実際のMongoDBデータ書き換え処理
    print(f"💾 MongoDBデータ書き換え処理開始...")
    
    # ロールバック用の元データを保持
    original_result_structure = copy.deepcopy(result_structure)
    print(f"📋 ロールバック用の元データを保持しました")
    
    def rollback_data():
        """データを元の状態に戻す"""
        try:
            print(f"🔄 ロールバック処理開始...")
            result_manager._mongo_module.update_document(
                collection_name='clustering_results',
                query={"mongo_result_id": mongo_result_id},
                update={"result": original_result_structure}
            )
            print(f"✅ ロールバック完了: データを元の状態に復元しました")
        except Exception as e:
            print(f"❌ ロールバックエラー: {e}")
    
    try:
        # 1. 挿入するデータの保持
        print(f"📦 挿入するデータを保持中...")
        items_to_insert = copy.deepcopy(items_to_move)
        print(f"   - 保持したデータ数: {len(items_to_insert)}個")
        print(f"   - 保持したデータ: {list(items_to_insert.keys())}")
        
        # 2. 挿入先にデータの追加
        print(f"📥 挿入先にデータを追加中...")
        print(f"   - 挿入先ID: {destination_id}")
        print(f"   - 挿入先の現在の状態: {destination_node.get('is_leaf')}")
        
        # 既存のdestination_nodeを使用（再取得は不要）
        if "data" not in destination_node:
            destination_node["data"] = {}
            print(f"   - dataフィールドを初期化しました")
        
        print(f"   - 挿入前の要素数: {len(destination_node['data'])}")
        print(f"   - 挿入前の要素: {list(destination_node['data'].keys())}")
        
        # 移動先に要素を追加
        for key, value in items_to_insert.items():
            destination_node["data"][key] = value
            print(f"   ✅ 追加: {key} → {destination_id}")
        
        print(f"   - 挿入後の要素数: {len(destination_node['data'])}")
        print(f"   - 挿入後の要素: {list(destination_node['data'].keys())}")
        print(f"✅ 挿入先への追加完了: {len(items_to_insert)}個の要素を追加")
        
        # 3. 挿入元のデータを削除
        print(f"🗑️  挿入元のデータを削除中...")
        
        def remove_items_from_source(node_dict, target_ids, item_type):
            """移動元から要素を削除"""
            removed_count = 0
            for key, value in list(node_dict.items()):
                if key in target_ids:
                    if item_type == "folders" and isinstance(value, dict):
                        del node_dict[key]
                        removed_count += 1
                        print(f"   ✅ 削除: {key} (フォルダ)")
                    elif item_type == "images" and isinstance(value, str):
                        del node_dict[key]
                        removed_count += 1
                        print(f"   ✅ 削除: {key} (画像)")
                
                # 再帰的に検索・削除
                if isinstance(value, dict) and "data" in value:
                    sub_removed = remove_items_from_source(value["data"], target_ids, item_type)
                    removed_count += sub_removed
            
            return removed_count
        
        removed_count = remove_items_from_source(result_structure, sources, source_type)
        print(f"✅ 挿入元からの削除完了: {removed_count}個の要素を削除")
        
        # 削除後の挿入先の状態を確認
        print(f"🔍 削除後の挿入先の状態確認...")
        print(f"   - 挿入先の要素数: {len(destination_node['data'])}")
        print(f"   - 挿入先の要素: {list(destination_node['data'].keys())}")
        
        json.dump(result_structure, open("c.json", "w"), indent=4)
        # 4. MongoDBに保存
        print(f"💾 MongoDBに保存中...")
        result_manager._mongo_module.update_document(
            collection_name='clustering_results',
            query={"mongo_result_id": mongo_result_id},
            update={"result": result_structure}
        )
        print(f"✅ MongoDB保存完了")
        
        # 保存後の確認
        print(f"🔍 保存後の確認中...")
        saved_result = result_manager._mongo_module.find_one_document(
            collection_name='clustering_results',
            query={"mongo_result_id": mongo_result_id}
        )
        if not saved_result:
            print(f"❌ エラー: 保存後の確認でデータが見つかりません")
            raise Exception("Data not found after save")
        
        saved_structure = saved_result.get("result", {})
        print(f"   - 保存されたデータの要素数: {len(saved_structure)}")
        print(f"   - 保存されたデータの要素: {list(saved_structure.keys())}")
        
        # 5. レスポンスの返却
        print(f"✅ すべての処理が完了しました")
        print(f"   - 移動対象数: {len(items_to_insert)}個")
        print(f"   - 移動タイプ: {source_type}")
        print(f"   - 移動先: {destination_folder}")
        print(f"   - 移動対象ID: {list(items_to_insert.keys())}")
        print(f"   - 削除数: {removed_count}個")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Successfully moved {len(items_to_insert)} {source_type} to '{destination_folder}'",
                "data": {
                    "moved_items": list(items_to_insert.keys()),
                    "destination_folder": destination_folder,
                    "source_type": source_type,
                    "moved_count": len(items_to_insert),
                    "removed_count": removed_count,
                    "operation_completed": True
                }
            }
        )
        
    except Exception as e:
        print(f"❌ 処理中にエラーが発生しました: {e}")
        print(f"🔄 ロールバック処理を実行します...")
        rollback_data()
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": f"Operation failed and rolled back: {str(e)}",
                "data": {
                    "error": str(e),
                    "rolled_back": True,
                    "original_state_restored": True
                }
            }
        )