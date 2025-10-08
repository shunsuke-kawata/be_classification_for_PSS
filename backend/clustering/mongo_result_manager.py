import json
from typing import Dict, List, Optional, Any
from .mongo_db_manager import MongoDBManager
class ResultManager:
    """
    クラスタリング結果のdictを扱うためのユーティリティクラス
    """
    
    def __init__(self, mongo_result_id:str,clustering_results: str="clustering_results"):
        self._mongo_result_id = mongo_result_id
        self._clustering_results = clustering_results
        self._mongo_module = MongoDBManager()
    
    @property
    def mongo_result_id(self)->str:
        return self._mongo_result_id
    
            
    def get_result(self)->dict:
        result = self._mongo_module.find_one_document(self._clustering_results,{"mongo_result_id":self._mongo_result_id})
        
        if not result:
            return None
        
        return result['result']
    
    def get_all_nodes(self)->dict:
        result = self._mongo_module.find_one_document(self._clustering_results,{"mongo_result_id":self._mongo_result_id})
        if not result:
            return None
        return result['all_nodes']
    
    def update_result(self, result_dict: dict, all_nodes_dict: dict) -> None:
        """
        クラスタリング結果を更新する
        
        Args:
            result_dict (dict): 更新するresult辞書
            all_nodes_dict (dict): 更新するall_nodes辞書
        """
        self._mongo_module.update_document(
            collection_name=self._clustering_results,
            query={"mongo_result_id": self._mongo_result_id},
            update={"mongo_result_id": self._mongo_result_id, "result": result_dict, "all_nodes": all_nodes_dict}
        )

    def find_node(self,node_id:str)->dict:
        result = self._mongo_module.find_one_document(self._clustering_results,{"mongo_result_id":self._mongo_result_id})
        if not result:
            return None
        return result['all_nodes'].get(node_id)
    
    def get_full_node_path(self,node_id:str)->str:
        """
        mongo_result_id に紐づくドキュメントから
        all_nodes のキー node_id にマッチするノードだけを返す
        """
        # all_nodesを取得してから指定されたnode_idのノードを返す
        all_nodes = self.get_all_nodes()
        if not all_nodes:
            return None
        return all_nodes.get(node_id)
    
    def get_parents(self, target_node_id: str) -> List[str]:
        """
        ノードのIDから、ルートまでの完全なパスを取得する
        
        Args:
            target_node_id (str): ファイルノードのID
            
        Returns:
            List[str]: ルートからファイルまでのパス（node_idの配列）
                      例: [root_id, parent_folder_id, ..., target_node_id]
        """
        print(f"=== ResultManager.get_parents() デバッグ ===")
        print(f"🔍 target_node_id: {target_node_id}")
                
        all_nodes = self.get_all_nodes()
        if not all_nodes:
            print(f"❌ all_nodes が見つかりません")
            return []
        
        print(f"🔍 all_nodes contains {len(all_nodes)} nodes")
        
        # ファイルノードを取得
        file_node = all_nodes.get(target_node_id)
        if not file_node:
            print(f"❌ target_node_id {target_node_id} がall_nodesに見つかりません")
            print(f"🔍 available node_ids (first 10): {list(all_nodes.keys())[:10]}")
            return []
        
        print(f"🔍 file_node: {file_node}")
        
        # パスを構築（ルートから順番に）
        path = []
        current_id = target_node_id
        
        # ファイルノードからルートまで遡る
        while current_id:
            current_node = all_nodes.get(current_id)
            if not current_node:
                print(f"❌ current_id {current_id} がall_nodesに見つかりません")
                break
            
            # 現在のノードIDをパスに追加（先頭に挿入）
            path.insert(0, current_id)
            print(f"🔍 added to path: {current_id}, current path: {path}")
            
            # 親ノードのIDを取得
            parent_id = current_node.get('parent_id')
            print(f"🔍 parent_id: {parent_id}")
            current_id = parent_id
        
        print(f"🔍 final path: {path}")
        return path
    
    
    def move_file_node(self,target_node_id:str, destination_folder_id:str)->None:
        target_node = self.find_node(target_node_id)
        if not target_node:
            raise ValueError(f"Node with id {target_node_id} not found")
        
        source_folder_id = target_node['parent_id'] 
        target_filename = target_node['name']
        destination_parents = self.get_parents(destination_folder_id)
        
        # ファイル移動処理を実行
        self._perform_file_move(
            target_node_id=target_node_id,
            target_filename=target_filename,
            destination_folder_id=destination_folder_id,
            destination_parents=destination_parents,
            source_folder_id=source_folder_id
        )
    
    def _perform_file_move(self, target_node_id: str, target_filename: str, 
                          destination_folder_id: str, destination_parents: List[str], source_folder_id: str) -> None:
        """
        ファイル移動の実際の処理を実行
        1. destination_folderのdataに target_node_id:filename を追加
        2. target_nodeのparent_idをdestination_folder_idに更新
        3. source_folderからtarget_node_idを削除
        """
        # 1. destination_folderのdataに target_node_id:filename を追加
        destination_data_path = f"result.{'.data.'.join(destination_parents)}.data.{target_node_id}"
        destination_update = {destination_data_path: target_filename}
        
        self._mongo_module.update_document(
            collection_name=self._clustering_results,
            query={"mongo_result_id": self._mongo_result_id},
            update=destination_update,
            upsert=False
        )
        
        # 2. all_nodesでtarget_nodeのparent_idをdestination_folder_idに更新
        parent_update_path = f"all_nodes.{target_node_id}.parent_id"
        parent_update = {parent_update_path: destination_folder_id}
        
        self._mongo_module.update_document(
            collection_name=self._clustering_results,
            query={"mongo_result_id": self._mongo_result_id},
            update=parent_update,
            upsert=False
        )
        
        # 3. source_folderからtarget_node_idを削除
        source_parents = self.get_parents(source_folder_id)
        source_data_path = f"result.{'.data.'.join(source_parents)}.data.{target_node_id}"
        
        # MongoDBの$unsetオペレータを使用してフィールドを削除
        collection = self._mongo_module.get_collection(self._clustering_results)
        collection.update_one(
            {"mongo_result_id": self._mongo_result_id},
            {"$unset": {source_data_path: ""}}
        )

    def delete_file_node(self,node_id:str)->None:
        target_node = self.find_node(node_id)
        if not target_node:
            raise ValueError(f"Node with id {node_id} not found")
        source_folder_id = target_node['parent_id']
        source_folder = self.find_node(source_folder_id)
        if not source_folder:
            raise ValueError(f"Node with id {source_folder_id} not found")
        source_folder_file_data = source_folder.get('data',None)
        if (source_folder_file_data is None):
            raise ValueError(f"Node with id {source_folder_id} has no data")
        del source_folder_file_data[node_id]

    def move_folder_node(self, target_folder_ids: List[str], destination_folder_id: str) -> None:
        """
        フォルダノードを移動する
        Args:
            target_folder_ids (List[str]): 移動するフォルダのIDの配列
            destination_folder_id (str): 移動先フォルダのID
        """
        destination_parents = self.get_parents(destination_folder_id)
        
        # 各フォルダに対して移動処理を実行
        for target_folder_id in target_folder_ids:
            self._perform_folder_move(
                target_folder_id=target_folder_id,
                destination_folder_id=destination_folder_id,
                destination_parents=destination_parents
            )
    
    def _perform_folder_move(self, target_folder_id: str, 
                            destination_folder_id: str, destination_parents: List[str]) -> None:
        """
        フォルダ移動の実際の処理を実行
        1. target_folderの情報を取得
        2. destination_folderのdataに target_folder_idとその中身を追加
        3. target_folderのparent_idをdestination_folder_idに更新
        4. source_folderからtarget_folder_idを削除
        """
        # 1. target_folderの情報を取得
        target_node = self.find_node(target_folder_id)
        if not target_node:
            raise ValueError(f"Node with id {target_folder_id} not found")
        
        source_folder_id = target_node['parent_id']
        
        # 2. 移動するフォルダの完全なデータを取得
        target_folder_parents = self.get_parents(target_folder_id)
        target_folder_data_path = f"result.{'.data.'.join(target_folder_parents)}"
        
        # target_folderの完全なデータ構造を取得
        target_folder_query = {"mongo_result_id": self._mongo_result_id}
        target_folder_projection = {target_folder_data_path: 1, "_id": 0}
        
        target_folder_result = self._mongo_module.find_one_with_projection(
            self._clustering_results,
            target_folder_query,
            target_folder_projection
        )
        
        if not target_folder_result:
            raise ValueError(f"Could not retrieve folder data for {target_folder_id}")
        
        # 移動するフォルダの完全なデータ構造を抽出
        folder_data_parts = target_folder_data_path.split('.')
        folder_data = target_folder_result
        for part in folder_data_parts:
            if part in folder_data:
                folder_data = folder_data[part]
            else:
                raise ValueError(f"Could not navigate to path: {target_folder_data_path}")
        
        # 3. destination_folderのdataに target_folder_idとその中身を追加
        destination_data_path = f"result.{'.data.'.join(destination_parents)}.data.{target_folder_id}"
        destination_update = {destination_data_path: folder_data}
        
        self._mongo_module.update_document(
            collection_name=self._clustering_results,
            query={"mongo_result_id": self._mongo_result_id},
            update=destination_update,
            upsert=False
        )
        
        # 4. all_nodesでtarget_folderのparent_idをdestination_folder_idに更新
        parent_update_path = f"all_nodes.{target_folder_id}.parent_id"
        parent_update = {parent_update_path: destination_folder_id}
        
        self._mongo_module.update_document(
            collection_name=self._clustering_results,
            query={"mongo_result_id": self._mongo_result_id},
            update=parent_update,
            upsert=False
        )
        
        # 5. source_folderからtarget_folder_idを削除
        source_parents = self.get_parents(source_folder_id)
        source_data_path = f"result.{'.data.'.join(source_parents)}.data.{target_folder_id}"
        
        # MongoDBの$unsetオペレータを使用してフィールドを削除
        collection = self._mongo_module.get_collection(self._clustering_results)
        collection.update_one(
            {"mongo_result_id": self._mongo_result_id},
            {"$unset": {source_data_path: ""}}
        )

    def remove_node_from_all_nodes(self, node_id: str) -> bool:
        """
        all_nodesから指定されたノードを削除する
        
        Args:
            node_id (str): 削除するノードのID
            
        Returns:
            bool: 削除に成功したかどうか
        """
        try:
            # all_nodesから該当ノードを削除
            all_nodes_path = f"all_nodes.{node_id}"
            
            collection = self._mongo_module.get_collection(self._clustering_results)
            result = collection.update_one(
                {"mongo_result_id": self._mongo_result_id},
                {"$unset": {all_nodes_path: ""}}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"❌ all_nodesからノード削除中にエラー: {e}")
            return False

    def remove_folders_from_result(self, folder_ids: List[str]) -> bool:
        """
        resultから指定された複数のフォルダを削除する
        
        Args:
            folder_ids (List[str]): 削除するフォルダのIDの配列
            
        Returns:
            bool: 全ての削除に成功したかどうか
        """
        try:
            all_success = True
            
            # 各フォルダに対して削除処理を実行
            for folder_id in folder_ids:
                success = self._perform_folder_removal(folder_id)
                if not success:
                    all_success = False
                    print(f"⚠️ フォルダ {folder_id} の削除に失敗しました")
            
            return all_success
            
        except Exception as e:
            print(f"❌ 複数フォルダ削除中にエラー: {e}")
            return False

    def _perform_folder_removal(self, folder_id: str) -> bool:
        """
        単体のフォルダをresultから削除する実際の処理
        
        Args:
            folder_id (str): 削除するフォルダのID
            
        Returns:
            bool: 削除に成功したかどうか
        """
        try:
            # 親フォルダのパスを取得
            parents = self.get_parents(folder_id)
            print(parents)
            
            if not parents or len(parents) <= 1:
                # トップレベルフォルダの場合（ルート直下）
                result_path = f"result.{folder_id}"
            else:
                # 子フォルダの場合
                # parents[:-1] で親フォルダまでのパスを取得（最後の自分自身を除く）

                result_path = f"result.{'.data.'.join(parents)}"
            
            print(f"🗂️ フォルダ削除パス: {result_path}")
            
            # PyMongoの$unsetオペレータを使用してフィールドを削除
            collection = self._mongo_module.get_collection(self._clustering_results)
            result = collection.update_one(
                {"mongo_result_id": self._mongo_result_id},  # フィルタ条件
                {"$unset": {result_path: ""}}                # 削除操作
            )
            
            success = result.modified_count > 0
            if success:
                print(f"✅ フォルダ {folder_id} を正常に削除しました")
            else:
                print(f"⚠️ フォルダ {folder_id} の削除で変更がありませんでした")
                
            return success
            
        except Exception as e:
            print(f"❌ フォルダ {folder_id} の削除中にエラー: {e}")
            return False

    def commit_changes(self) -> None:
        """
        変更をコミットする（現在は何もしないが、将来的に必要に応じて実装）
        """
        pass

    def rename_node(self, node_id: str, new_name: str = None, is_leaf: bool = None) -> dict:
        """
        指定されたノードの名前やis_leafを変更する
        
        Args:
            node_id (str): 変更対象のノードID
            new_name (str, optional): 新しい名前
            is_leaf (bool, optional): リーフノードかどうか
            
        Returns:
            dict: 操作結果
        """
        try:
            print(f"🏷️ rename_node呼び出し: node_id={node_id}, new_name={new_name}, is_leaf={is_leaf}")
            
            # デバッグ: 現在のドキュメント構造を確認
            collection = self._mongo_module.get_collection(self._clustering_results)
            current_doc = collection.find_one({"mongo_result_id": self._mongo_result_id})
            if current_doc:
                print(f"🔍 current_doc keys: {list(current_doc.keys())}")
                if 'result' in current_doc:
                    print(f"🔍 result keys: {list(current_doc['result'].keys()) if isinstance(current_doc['result'], dict) else 'not dict'}")
                if 'all_nodes' in current_doc:
                    all_nodes = current_doc['all_nodes']
                    if node_id in all_nodes:
                        print(f"🔍 target node in all_nodes: {all_nodes[node_id]}")
                    else:
                        print(f"❌ node_id {node_id} not found in all_nodes")
                        print(f"🔍 all_nodes keys: {list(all_nodes.keys())[:10]}...")  # 最初の10個だけ表示
            
            # 入力検証
            if not node_id or not node_id.strip():
                print(f"❌ 無効なnode_id: {node_id}")
                return {"success": False, "error": "Invalid node_id"}
            
            # nameとis_leafの両方がNoneの場合はエラー
            if new_name is None and is_leaf is None:
                print(f"❌ nameとis_leafの両方が未指定です")
                return {"success": False, "error": "At least one of 'new_name' or 'is_leaf' must be provided"}
            
            # nameが指定されている場合は空文字チェック
            if new_name is not None and not new_name.strip():
                print(f"❌ 無効なnew_name: {new_name}")
                return {"success": False, "error": "Invalid new_name"}
            
            # resultの変更
            parents = self.get_parents(node_id)
            print(f"📍 parents: {parents}")
            
            # 更新用のパスと値を準備
            update_fields = {}
            
            # パス生成の修正
            if not parents or len(parents) == 0:
                print(f"❌ parentsが見つかりません: {parents}")
                return {"success": False, "error": f"Parents not found for node_id: {node_id}"}
            elif len(parents) == 1:
                # トップレベルフォルダの場合（ルート直下）
                base_path = f"result.{node_id}"
            else:
                # サブフォルダの場合（parents[0]はルート、parents[-1]は対象ノード）
                # result.parent1.data.parent2.data...target_node
                parent_path_parts = []
                for i, parent in enumerate(parents[:-1]):  # 最後のnode_id（自分自身）を除く
                    if i == 0:
                        parent_path_parts.append(parent)
                    else:
                        parent_path_parts.extend(["data", parent])
                
                if len(parent_path_parts) > 1:
                    base_path = f"result.{'.'.join(parent_path_parts)}.data.{node_id}"
                else:
                    base_path = f"result.{parent_path_parts[0]}.data.{node_id}"
            
            print(f"🔍 生成されたbase_path: {base_path}")
            
            # nameの更新
            if new_name is not None:
                name_path = f"{base_path}.name"
                update_fields[name_path] = new_name.strip()
                print(f"📝 名前更新パス: {name_path} -> {new_name.strip()}")
            
            # is_leafの更新
            if is_leaf is not None:
                is_leaf_path = f"{base_path}.is_leaf"
                update_fields[is_leaf_path] = is_leaf
                print(f"🍃 is_leaf更新パス: {is_leaf_path} -> {is_leaf}")
            
            # MongoDBで更新実行
            collection = self._mongo_module.get_collection(self._clustering_results)
            result = collection.update_one(
                {"mongo_result_id": self._mongo_result_id},
                {"$set": update_fields}
            )
            
            # all_nodesの更新
            all_nodes_update_fields = {}
            if new_name is not None:
                all_nodes_name_path = f"all_nodes.{node_id}.name"
                all_nodes_update_fields[all_nodes_name_path] = new_name.strip()
                print(f"📝 all_nodes名前更新パス: {all_nodes_name_path} -> {new_name.strip()}")
            
            if is_leaf is not None:
                all_nodes_is_leaf_path = f"all_nodes.{node_id}.is_leaf"
                all_nodes_update_fields[all_nodes_is_leaf_path] = is_leaf
                print(f"🍃 all_nodes is_leaf更新パス: {all_nodes_is_leaf_path} -> {is_leaf}")
            
            # all_nodesも更新
            if all_nodes_update_fields:
                all_nodes_result = collection.update_one(
                    {"mongo_result_id": self._mongo_result_id},
                    {"$set": all_nodes_update_fields}
                )
                print(f"📊 all_nodes更新結果: modified_count={all_nodes_result.modified_count}")
            
            print(f"📊 result更新結果: modified_count={result.modified_count}")
            
            if result.modified_count > 0:
                return {
                    "success": True,
                    "message": "Node updated successfully",
                    "updated_fields": {
                        "name": new_name if new_name is not None else "not updated",
                        "is_leaf": is_leaf if is_leaf is not None else "not updated"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No changes were made to the database"
                }
                
        except Exception as e:
            print(f"❌ rename_node処理中にエラー: {e}")
            return {"success": False, "error": str(e)}

    def _update_name_in_result_recursive(self, node: dict, target_id: str, new_name: str) -> bool:
        """
        result内の指定されたIDのノードの名前を再帰的に検索・更新する
        
        Args:
            node (dict): 現在処理中のノード
            target_id (str): 変更対象のノードID
            new_name (str): 新しい名前
            
        Returns:
            bool: 変更が行われたかどうか
        """
        # 現在のノードが対象の場合
        if node.get('id') == target_id:
            node['name'] = new_name
            print(f"✅ result内でノード名を更新: {target_id} -> {new_name}")
            return True
        
        # 子ノードを再帰的に検索
        changed = False
        if 'children' in node:
            for child in node['children']:
                if self._update_name_in_result_recursive(child, target_id, new_name):
                    changed = True
        
        return changed