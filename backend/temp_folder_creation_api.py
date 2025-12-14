# このコードをaction.pyの適切な場所に追加してください

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
        from clustering.utils import Utils
        new_folder_id = Utils.generate_uuid()
        
        # フォルダ名をIDとして設定
        folder_name = new_folder_id
        
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
        
        # resultに空のデータを追加
        new_folder_data = {
            "type": "folder",
            "name": folder_name,
            "is_leaf": is_leaf,
            "data": {}  # 空のフォルダとして作成
        }
        
        # 親フォルダの配下に追加
        if parent_folder_id in result_data:
            parent_folder_data = result_data[parent_folder_id]
            if "data" in parent_folder_data:
                parent_folder_data["data"][new_folder_id] = new_folder_data
            else:
                parent_folder_data["data"] = {new_folder_id: new_folder_data}
        else:
            # 親がトップレベルの場合
            result_data[new_folder_id] = new_folder_data
        
        # 更新をMongoDBに保存
        result_manager.update_result(result_data, all_nodes)
        
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
