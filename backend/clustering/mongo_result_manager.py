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
            node_name = current_node.get('name', '(no name)')
            print(f"🔍 added to path: {current_id} (name: {node_name}), current path: {path}")
            
            # 親ノードのIDを取得
            parent_id = current_node.get('parent_id')
            parent_name = all_nodes.get(parent_id, {}).get('name', '(no name)') if parent_id else None
            print(f"🔍 parent_id: {parent_id} (name: {parent_name})")
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

        source_folder_id = target_node.get('parent_id')
        if not source_folder_id:
            raise ValueError(f"Source folder id for node {node_id} not found")

        # まず result の該当フィールドを unset
        source_parents = self.get_parents(source_folder_id)
        if not source_parents:
            raise ValueError(f"Could not determine parents for source folder {source_folder_id}")

        source_data_path = f"result.{'.data.'.join(source_parents)}.data.{node_id}"
        collection = self._mongo_module.get_collection(self._clustering_results)
        collection.update_one(
            {"mongo_result_id": self._mongo_result_id},
            {"$unset": {source_data_path: ""}}
        )

        # all_nodes からも削除
        self.remove_node_from_all_nodes(node_id)

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
        
        # 2. 移動するフォルダの完全なデータを取得（get_parents を使って result を辿る）
        target_folder_parents = self.get_parents(target_folder_id)

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
                print(f"📊 all_nodes更新結果: matched_count={all_nodes_result.matched_count}, modified_count={all_nodes_result.modified_count}")
            
            print(f"📊 result更新結果: matched_count={result.matched_count}, modified_count={result.modified_count}")
            
            # 更新が成功したかチェック（matched_countがあることを確認）
            # modified_count=0でも、matched_count>0であれば対象ノードは存在する
            if result.matched_count > 0:
                return {
                    "success": True,
                    "message": "Node updated successfully" if result.modified_count > 0 else "Node already has the same value",
                    "updated_fields": {
                        "name": new_name if new_name is not None else "not updated",
                        "is_leaf": is_leaf if is_leaf is not None else "not updated"
                    },
                    "modified": result.modified_count > 0
                }
            else:
                return {
                    "success": False,
                    "error": f"Node with id '{node_id}' not found in database"
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

    def get_node_info(self, node_id: str) -> dict:
        """
        指定されたnode_idのノード情報をall_nodesから取得する（APIエンドポイント用）
        
        Args:
            node_id (str): 取得するノードのID
            
        Returns:
            dict: ノード情報（all_nodesの値のみ）
            成功時: {"success": True, "node_id": str, "data": dict}
            失敗時: {"success": False, "node_id": str, "error": str}
        """
        try:
            print(f"🔍 get_node_info呼び出し: node_id={node_id}")
            
            # 入力検証
            if not node_id or not node_id.strip():
                return {
                    "success": False,
                    "node_id": node_id,
                    "error": "Invalid node_id provided"
                }
            
            # all_nodesから直接ノード情報を取得
            all_nodes = self.get_all_nodes()
            if not all_nodes:
                return {
                    "success": False,
                    "node_id": node_id,
                    "error": "No clustering results found"
                }
            
            # 指定されたnode_idの情報を取得
            node_info = all_nodes.get(node_id.strip())
            
            if not node_info:
                return {
                    "success": False,
                    "node_id": node_id,
                    "error": f"Node with id '{node_id}' not found"
                }
            
            
            return {
                "success": True,
                "node_id": node_id,
                "data": node_info
            }
            
        except Exception as e:
            print(f"❌ get_node_info処理中にエラー: {e}")
            return {
                "success": False,
                "node_id": node_id,
                "error": str(e)
            }
    
    def insert_image_to_leaf_folder(self, clustering_id: str, image_path: str, target_folder_id: str) -> dict:
        """
        指定されたリーフフォルダに画像を追加する
        
        Args:
            clustering_id (str): クラスタリングID
            image_path (str): 画像のパス
            target_folder_id (str): 挿入先のフォルダID（リーフフォルダ）
            
        Returns:
            dict: 挿入結果
            成功時: {"success": True, "folder_id": str, "clustering_id": str}
            失敗時: {"success": False, "error": str}
        """
        try:
            print(f"📥 insert_image_to_leaf_folder呼び出し:")
            print(f"   clustering_id: {clustering_id}")
            print(f"   image_path: {image_path}")
            print(f"   target_folder_id: {target_folder_id}")

            # all_nodesから対象フォルダを取得
            all_nodes = self.get_all_nodes()
            if not all_nodes:
                return {"success": False, "error": "No clustering results found"}

            target_node = all_nodes.get(target_folder_id)
            if not target_node:
                return {"success": False, "error": f"Folder {target_folder_id} not found"}

            if not target_node.get('is_leaf', False):
                return {"success": False, "error": f"Folder {target_folder_id} is not a leaf folder"}

            # get_parents を使って result 内の該当ノードに直接到達する
            parents = self.get_parents(target_folder_id)
            if not parents:
                return {"success": False, "error": f"Parents not found for folder {target_folder_id}"}

            # 1. resultのtarget_folderのdataに clustering_id:image_path を追加
            # _perform_file_moveと同じ方式でパスを構築
            destination_data_path = f"result.{'.data.'.join(parents)}.data.{clustering_id}"
            destination_update = {destination_data_path: image_path}
            
            print(f"   📍 result更新パス: {destination_data_path}")
            
            self._mongo_module.update_document(
                collection_name=self._clustering_results,
                query={"mongo_result_id": self._mongo_result_id},
                update=destination_update,
                upsert=False
            )
            
            # 2. all_nodesにファイルノードを追加
            new_file_node = {
                "type": "file",
                "id": clustering_id,
                "name": image_path,
                "parent_id": target_folder_id,
                "is_leaf": None
            }
            
            all_nodes_file_node_path = f"all_nodes.{clustering_id}"
            all_nodes_update = {all_nodes_file_node_path: new_file_node}
            
            print(f"   📍 all_nodes更新パス: {all_nodes_file_node_path}")
            
            self._mongo_module.update_document(
                collection_name=self._clustering_results,
                query={"mongo_result_id": self._mongo_result_id},
                update=all_nodes_update,
                upsert=False
            )

            print(f"✅ insert_image_to_leaf_folder完了")
            print(f"   📄 all_nodesにファイルノード追加: {clustering_id}")
            print(f"   📁 resultのフォルダ {target_folder_id} に画像追加")
            return {
                "success": True,
                "folder_id": target_folder_id,
                "clustering_id": clustering_id
            }

        except Exception as e:
            print(f"❌ insert_image_to_leaf_folder処理中にエラー: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def get_all_leaf_folders(self) -> List[dict]:
        """
        すべてのリーフフォルダ（is_leaf=True）を取得する
        
        Returns:
            List[dict]: リーフフォルダのリスト [{"id": str, "name": str, "parent_id": str}, ...]
        """
        try:
            all_nodes = self.get_all_nodes()
            if not all_nodes:
                return []
            
            leaf_folders = []
            for node_id, node_data in all_nodes.items():
                if node_data.get('is_leaf', False):
                    leaf_folders.append({
                        "id": node_id,
                        "name": node_data.get('name', ''),
                        "parent_id": node_data.get('parent_id', None)
                    })
            
            print(f"📂 get_all_leaf_folders: {len(leaf_folders)}個のリーフフォルダを取得")
            return leaf_folders
            
        except Exception as e:
            print(f"❌ get_all_leaf_folders処理中にエラー: {e}")
            return []
    
    def get_folder_data_from_result(self, folder_id: str) -> dict:
        """
        result内でフォルダIDを探索して、そのフォルダのdataを取得する
        
        Args:
            folder_id (str): フォルダID
            
        Returns:
            dict: フォルダのdata
            成功時: {"success": True, "data": {...}}  # dataには画像のclustering_id: pathのマッピング
            失敗時: {"success": False, "error": str}
        """
        try:
            result = self.get_result()
            if not result:
                return {"success": False, "error": "No clustering results found"}
            
            # resultを再帰的に探索してフォルダを見つける
            def find_folder_recursive(node: dict, target_id: str) -> dict:
                for current_folder_id, folder_data in node.items():
                    if current_folder_id == target_id:
                        # 見つかった！
                        if folder_data.get('is_leaf', False):
                            # リーフフォルダの場合、dataには画像のマッピングが入っている
                            return {"success": True, "data": folder_data.get('data', {})}
                        else:
                            # 非リーフフォルダの場合
                            return {"success": True, "data": folder_data.get('data', {})}
                    elif not folder_data.get('is_leaf', False) and isinstance(folder_data.get('data'), dict):
                        # 非リーフフォルダの場合、再帰的に探索
                        result = find_folder_recursive(folder_data['data'], target_id)
                        if result['success']:
                            return result
                
                return {"success": False, "error": f"Folder {target_id} not found"}
            
            return find_folder_recursive(result, folder_id)
            
        except Exception as e:
            print(f"❌ get_folder_data_from_result処理中にエラー: {e}")
            return {"success": False, "error": str(e)}

    def get_leaf_folder_image_clustering_ids(self, folder_id: str) -> dict:
        """
        指定したフォルダIDがリーフ（is_leaf=True）の場合、そのフォルダ内に含まれる
        画像のclustering_id一覧を返します。

        Returns:
            dict: 成功時: {"success": True, "data": [<clustering_id>, ...]}
                  失敗時: {"success": False, "error": str}
        """
        try:
            if not folder_id or not folder_id.strip():
                return {"success": False, "error": "Invalid folder_id provided"}

            folder_data_result = self.get_folder_data_from_result(folder_id)

            if not folder_data_result.get('success', False):
                return {"success": False, "error": folder_data_result.get('error', f"Folder {folder_id} not found in result")}

            folder_data = folder_data_result.get('data', {})
            if not isinstance(folder_data, dict):
                print(f"⚠️ folder_data is not a dict for folder {folder_id}: {type(folder_data)}")
                return {"success": False, "error": "Folder data structure is invalid"}

            clustering_ids = list(folder_data.keys())
            print(f"🔍 clustering_ids for folder {folder_id}: count={len(clustering_ids)} sample={clustering_ids[:10]}")
            return {"success": True, "data": clustering_ids}

        except Exception as e:
            print(f"❌ get_leaf_folder_image_clustering_ids処理中にエラー: {e}")
            return {"success": False, "error": str(e)}
    
    def export_classification_data(self) -> dict:
        """
        分類結果データをエクスポート用に取得する
        
        Returns:
            dict: エクスポートデータ
            成功時: {"success": True, "result": dict, "all_nodes": dict}
            失敗時: {"success": False, "error": str}
        """
        try:
            result = self.get_result()
            all_nodes = self.get_all_nodes()
            
            if result is None:
                return {"success": False, "error": "Result data not found"}
            
            if all_nodes is None:
                return {"success": False, "error": "All nodes data not found"}
            
            return {
                "success": True,
                "result": result,
                "all_nodes": all_nodes
            }
            
        except Exception as e:
            print(f"❌ export_classification_data処理中にエラー: {e}")
            return {"success": False, "error": str(e)}

    def get_child_folders(self, folder_id: str, folder_type: Optional[str] = None) -> dict:
        """
        指定されたフォルダIDの直下にあるフォルダ（子フォルダ）の node データ一覧を返すインターフェース。

        Args:
            folder_id (str): 対象フォルダのノードID

            folder_type (Optional[str]): 返却する子フォルダの `type` を指定します。
                例: "folder" や "file"。
                指定されていない場合はフィルタを適用しません（すべての直下子ノードを返します）。

        Returns:
            dict: 成功時は {"success": True, "data": { <child_id>: <node_data>, ... }}
                  失敗時は {"success": False, "error": str}

        注意: 直下の子フォルダのみを返します（孫ノードは含みません）。
        """
        try:
            if not folder_id or not folder_id.strip():
                return {"success": False, "error": "Invalid folder_id provided"}

            all_nodes = self.get_all_nodes()
            if not all_nodes:
                return {"success": False, "error": "No clustering results found"}

            # 指定フォルダが存在するか確認
            if folder_id not in all_nodes:
                return {"success": False, "error": f"Folder with id '{folder_id}' not found"}

            child_folders: Dict[str, Any] = {}
            for node_id, node_data in all_nodes.items():
                # 直下の子要素を収集
                parent_id = node_data.get('parent_id')
                if parent_id != folder_id:
                    continue

                # folder_type が指定されている場合は type フィールドでフィルタ
                if folder_type and folder_type.strip():
                    node_type = node_data.get('type')
                    if node_type != folder_type:
                        continue

                child_folders[node_id] = node_data

            return {"success": True, "data": child_folders}
        except Exception as e:
            print(f"❌ get_child_folders処理中にエラー: {e}")
            return {"success": False, "error": str(e)}
    
    def create_new_leaf_folder(
        self, 
        folder_name: str, 
        parent_id: Optional[str], 
        initial_clustering_id: str, 
        initial_image_path: str
    ) -> dict:
        """
        新しいリーフフォルダをトップレベル（またはparent配下）に作成し、初期画像を挿入する
        
        Args:
            folder_name (str): 新しいフォルダの名前
            parent_id (Optional[str]): 親フォルダID（Noneの場合はトップレベル）
            initial_clustering_id (str): 初期画像のclustering_id
            initial_image_path (str): 初期画像のパス
            
        Returns:
            dict: 成功時: {"success": True, "folder_id": str}
                  失敗時: {"success": False, "error": str}
        """
        try:
            import uuid
            
            # 新しいフォルダIDを生成
            new_folder_id = str(uuid.uuid4())
            
            # all_nodesとresultを取得
            all_nodes = self.get_all_nodes()
            result = self.get_result()
            
            if all_nodes is None or result is None:
                return {"success": False, "error": "No clustering results found"}
            
            # 新しいフォルダノードを作成（all_nodes用）
            new_folder_node = {
                "type": "folder",
                "id": new_folder_id,
                "name": folder_name,
                "parent_id": parent_id,
                "is_leaf": True
            }
            
            # 新しいファイルノード（画像）を作成（all_nodes用）
            new_file_node = {
                "type": "file",
                "id": initial_clustering_id,  # clustering_idをファイルノードのIDとして使用
                "name": initial_image_path,
                "parent_id": new_folder_id,  # 新しいフォルダを親として指定
                "is_leaf": None
            }
            
            # all_nodesに両方追加
            all_nodes[new_folder_id] = new_folder_node
            all_nodes[initial_clustering_id] = new_file_node
            
            # resultに新しいフォルダを追加
            new_folder_data = {
                "is_leaf": True,
                "data": {
                    initial_clustering_id: initial_image_path
                }
            }
            
            if parent_id is None:
                # トップレベルに追加
                result[new_folder_id] = new_folder_data
            else:
                # 親フォルダの配下に追加
                # resultを再帰的に探索して親フォルダを見つける
                def add_to_parent_recursive(node: dict, target_parent_id: str) -> bool:
                    for folder_id, folder_data in node.items():
                        if folder_id == target_parent_id:
                            # 親フォルダが見つかった
                            if not folder_data.get('is_leaf', False):
                                # 非リーフフォルダの場合、dataに追加
                                folder_data['data'][new_folder_id] = new_folder_data
                                return True
                            else:
                                # 親がリーフフォルダの場合はエラー
                                return False
                        elif not folder_data.get('is_leaf', False) and isinstance(folder_data.get('data'), dict):
                            # 非リーフフォルダの場合、再帰的に探索
                            if add_to_parent_recursive(folder_data['data'], target_parent_id):
                                return True
                    return False
                
                if not add_to_parent_recursive(result, parent_id):
                    return {"success": False, "error": f"Parent folder {parent_id} not found or is a leaf folder"}
            
            # MongoDBに更新を保存
            self.update_result(result, all_nodes)
            
            print(f"✅ create_new_leaf_folder: 新しいフォルダ '{folder_name}' (ID: {new_folder_id}) を作成しました")
            print(f"   📁 フォルダノード追加: {new_folder_id}")
            print(f"   📄 ファイルノード追加: {initial_clustering_id} (name: {initial_image_path})")
            return {"success": True, "folder_id": new_folder_id}
            
        except Exception as e:
            print(f"❌ create_new_leaf_folder処理中にエラー: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
