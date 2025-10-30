import base64
from datetime import datetime
import mimetypes
import os
import sys
import json
import asyncio
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Response, UploadFile, File, Form, status, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

current_dir = os.path.dirname(__file__)  # = subfolder/
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # = your_project/
clustering_path = os.path.join(parent_dir, "clustering")
# from backend.clustering.caption_manager import CaptionManager
from clustering.caption_manager import CaptionManager
from db_utils.commons import create_connect_session, execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewImage
from pathlib import Path
from config import DEFAULT_IMAGE_PATH ,OPENAI_API_KEY
from clustering.caption_manager import CaptionManager
from clustering.chroma_db_manager import ChromaDBManager
from clustering.embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from clustering.embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from clustering.utils import Utils

images_endpoint = APIRouter()

# アップロード状況を管理するディクショナリ
upload_status_cache = {}

class UploadResult:
    def __init__(self, filename: str, success: bool, message: str, data: dict = None, error_type: str = None):
        self.filename = filename
        self.success = success
        self.message = message
        self.data = data or {}
        self.error_type = error_type
        self.timestamp = datetime.now().isoformat()

def validate_image_file(file: UploadFile) -> tuple[bool, str]:
    """
    画像ファイルの妥当性を検証
    """
    # ファイル名チェック
    if not file.filename:
        return False, "ファイル名が空です"
    
    # ファイルサイズチェック（例: 10MB制限）
    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        return False, "ファイルサイズが10MBを超えています"
    
    # 拡張子チェック
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return False, f"サポートされていないファイル形式です: {file_ext}"
    
    return True, "OK"

async def process_single_upload(
    project_id: int, 
    uploaded_user_id: int, 
    file: UploadFile,
    delay: float = 0.0
) -> UploadResult:
    """
    単一画像のアップロード処理（非同期）
    """
    if delay > 0:
        await asyncio.sleep(delay)
    
    start_time = time.time()
    filename = file.filename
    
    try:
        # ファイル妥当性チェック
        is_valid, validation_message = validate_image_file(file)
        if not is_valid:
            return UploadResult(filename, False, validation_message, error_type="ValidationError")

        connect_session = create_connect_session()
        if connect_session is None:
            return UploadResult(filename, False, "データベース接続失敗", error_type="DatabaseConnectionError")

        # プロジェクトのoriginal_images_folder_pathを取得
        query_text = f"""
            SELECT original_images_folder_path FROM projects WHERE id = {project_id};
        """
        result, _ = execute_query(connect_session, query_text)
        
        if not result or result.rowcount == 0:
            return UploadResult(filename, False, "プロジェクトが見つかりません", error_type="ProjectNotFoundError")

        original_images_folder_path = result.mappings().first()["original_images_folder_path"]
        save_dir = Path(DEFAULT_IMAGE_PATH) / original_images_folder_path
        os.makedirs(save_dir, exist_ok=True)

        contents = await file.read()
        filename_without_ext, _ = os.path.splitext(filename)
        png_path = f"{filename_without_ext}.png"
        escaped_png_path = png_path.replace("'", "''")  # SQLインジェクション対策

        # 同名画像が既に存在するか確認（DB + ファイルシステム）
        check_query = f"""
            SELECT id FROM images WHERE name = '{escaped_png_path}' AND project_id = {project_id};
        """
        result, _ = execute_query(connect_session, check_query)
        
        save_path = save_dir / png_path
        file_exists_in_fs = save_path.exists()
        
        if (result and result.rowcount > 0) or file_exists_in_fs:
            error_detail = []
            if result and result.rowcount > 0:
                error_detail.append("データベースに存在")
            if file_exists_in_fs:
                error_detail.append("ファイルシステムに存在")
            
            return UploadResult(
                filename, 
                False, 
                f"同名の画像が既に存在します ({', '.join(error_detail)})", 
                {"conflict_sources": error_detail},
                error_type="ConflictError"
            )

        # 画像をPNGに変換して保存
        png_bytes = Utils.image2png(contents)
        with open(save_path, "wb") as f:
            f.write(png_bytes)

        # 仮のキャプション生成
        is_created, created_caption = Utils.get_exmaple_caption(png_path)

        # ベクトルDBへ登録（3つのデータベースに分けて保存）
        sentence_name_db_manager = ChromaDBManager("sentence_name_embeddings")
        sentence_usage_db_manager = ChromaDBManager("sentence_usage_embeddings")
        sentence_category_db_manager = ChromaDBManager("sentence_category_embeddings")
        image_db_manager = ChromaDBManager("image_embeddings")
        
        # chroma_sentence_idを生成
        sentence_id = Utils.generate_uuid()
        image_id = Utils.generate_uuid()
        
        # 生成されたキャプションを3つの部分に分割
        name_part, usage_part, category_part = ChromaDBManager.split_sentence_document(created_caption)
        
        # 各部分のembeddingを生成
        name_embedding = SentenceEmbeddingsManager.sentence_to_embedding(name_part)
        usage_embedding = SentenceEmbeddingsManager.sentence_to_embedding(usage_part)
        category_embedding = SentenceEmbeddingsManager.sentence_to_embedding(category_part)
        
        # 各データベースに同じsentence_idで保存
        sentence_name_db_manager.collection.add(
            ids=[sentence_id],
            documents=[name_part],
            metadatas=[ChromaDBManager.ChromaMetaData(
                path=png_path,
                document=name_part,
                is_success=is_created,
                sentence_id=sentence_id
            ).to_dict()],
            embeddings=[name_embedding]
        )
        
        sentence_usage_db_manager.collection.add(
            ids=[sentence_id],
            documents=[usage_part],
            metadatas=[ChromaDBManager.ChromaMetaData(
                path=png_path,
                document=usage_part,
                is_success=is_created,
                sentence_id=sentence_id
            ).to_dict()],
            embeddings=[usage_embedding]
        )
        
        sentence_category_db_manager.collection.add(
            ids=[sentence_id],
            documents=[category_part],
            metadatas=[ChromaDBManager.ChromaMetaData(
                path=png_path,
                document=category_part,
                is_success=is_created,
                sentence_id=sentence_id
            ).to_dict()],
            embeddings=[category_embedding]
        )
        
        # 画像データベースに保存
        image_embedding = ImageEmbeddingsManager.image_to_embedding(save_path)
        image_db_manager.collection.add(
            ids=[image_id],
            documents=[created_caption],
            metadatas=[ChromaDBManager.ChromaMetaData(
                path=png_path,
                document=created_caption,
                is_success=is_created,
                sentence_id=image_id
            ).to_dict()],
            embeddings=[image_embedding]
        )
        
        is_created_for_sql = 'TRUE' if is_created else 'FALSE'
        escaped_caption = created_caption.replace("'", "''") if is_created else "NULL"

        clustering_id = Utils.generate_uuid()
        # 新しいスキーマに対応：統一sentence_idを保存
        insert_query = f"""
            INSERT INTO images(name, is_created_caption, caption, project_id, clustering_id, chromadb_sentence_id, chromadb_image_id, uploaded_user_id)
            VALUES ('{escaped_png_path}', {is_created_for_sql}, {'NULL' if not is_created else f"'{escaped_caption}'"}, {project_id}, '{clustering_id}', '{sentence_id}', '{sentence_id}', '{uploaded_user_id}');
        """
        result, _ = execute_query(connect_session, insert_query)

        if result:
            # 挿入された画像のIDを取得
            image_id_query = f"""
                SELECT id FROM images WHERE clustering_id = '{clustering_id}';
            """
            image_id_result, _ = execute_query(connect_session, image_id_query)
            
            if image_id_result:
                image_id = image_id_result.mappings().first()["id"]
                
                # プロジェクトメンバー全員のuser_image_clustering_statesレコードを作成
                members_query = f"""
                    SELECT user_id FROM project_memberships WHERE project_id = {project_id};
                """
                members_result, _ = execute_query(connect_session, members_query)
                
                if members_result:
                    members = members_result.mappings().all()
                    for member in members:
                        user_id = member["user_id"]
                        state_insert_query = f"""
                            INSERT INTO user_image_clustering_states(user_id, image_id, project_id, is_clustered)
                            VALUES ({user_id}, {image_id}, {project_id}, 0);
                        """
                        execute_query(connect_session, state_insert_query)
                    
                    # 初期クラスタリングが完了している全メンバーのcontinuous_clustering_stateを2（実行可能）に更新
                    update_continuous_state_query = f"""
                        UPDATE project_memberships
                        SET continuous_clustering_state = 2
                        WHERE project_id = {project_id}
                        AND init_clustering_state = 2
                        AND continuous_clustering_state != 1;
                    """
                    execute_query(connect_session, update_continuous_state_query)
                    print(f"✅ プロジェクト{project_id}の全メンバーのcontinuous_clustering_stateを更新しました")
            
            processing_time = round(time.time() - start_time, 2)
            return UploadResult(
                filename, 
                True, 
                "アップロード成功", 
                {
                    "clustering_id": clustering_id,
                    "chromadb_sentence_id": sentence_id,  # 統一ID
                    "chromadb_image_id": sentence_id,
                    "processing_time": processing_time
                }
            )
        else:
            return UploadResult(filename, False, "データベース挿入失敗", error_type="DatabaseInsertError")

    except Exception as e:
        return UploadResult(
            filename, 
            False, 
            f"予期しないエラーが発生: {str(e)}", 
            {"error_detail": str(e)},
            error_type=type(e).__name__
        )

# 画像一覧取得（指定されたプロジェクトIDに紐づく画像を返す）
@images_endpoint.get('/images', tags=["images"], description="画像一覧を取得", responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}

})
def read_images(project_id: int = None):
    if project_id is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "project_id is required", "data": None}
        )

    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to connect to database", "data": None}
        )

    query_text = f"""
        SELECT id, name, is_created_caption, caption ,project_id, uploaded_user_id, created_at FROM images WHERE project_id = {project_id};
    """
    result, _ = execute_query(connect_session, query_text)
    
    if result:
        rows = result.mappings().all()
        # datetime を文字列に変換
        formatted_rows = []
        for row in rows:
            row_dict = dict(row)

            if isinstance(row_dict.get("created_at"), datetime):
                row_dict["created_at"] = row_dict["created_at"].isoformat()
            formatted_rows.append(row_dict)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "succeeded to read images", "data": formatted_rows}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "failed to read images", "data": None}
        )

@images_endpoint.get("/images/folder/{folder_id}", tags=["images"], description="フォルダ内の画像一覧を取得")
def list_images_in_folder(folder_id: str):
    # backend/images ディレクトリを基準に画像を探す
    backend_dir = Path(__file__).parent.parent  # backend/
    images_root_dir = backend_dir / "images"


    # backend/images 配下のフォルダ一覧を取得
    if not images_root_dir.exists():
        return JSONResponse(status_code=404, content={"message": "images ディレクトリが存在しません", "data": None})

    # 指定されたフォルダ内の画像ファイルを取得
    images_dir = images_root_dir / folder_id
    if not images_dir.exists() or not images_dir.is_dir():
        return JSONResponse(status_code=404, content={"message": "指定フォルダが存在しません", "data": None})

    image_files = [f.name for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    return JSONResponse(status_code=200, content={"message": "画像一覧を取得しました", "data": image_files})


@images_endpoint.get("/images/{folder_id}/{name}")
def get_image(folder_id: str, name: str):
    backend_dir = Path(__file__).parent.parent  # backend/
    images_root_dir = backend_dir / "images"
    folder_path = images_root_dir / folder_id
    image_path = folder_path / name

    if not images_root_dir.exists():
        return JSONResponse(status_code=404, content={"message": "images ディレクトリが存在しません", "data": None})

    if not folder_path.exists() or not folder_path.is_dir():
        return JSONResponse(status_code=404, content={"message": "フォルダが存在しません", "data": None})

    if not image_path.exists():
        return JSONResponse(status_code=404, content={"message": "ファイルが存在しません", "data": None})

    if not image_path.is_file():
        return JSONResponse(status_code=400, content={"message": "指定されたパスはファイルではありません", "data": None})

    # MIMEタイプを推定
    media_type, _ = mimetypes.guess_type(str(image_path))
    if media_type is None:
        media_type = "application/octet-stream"

    return FileResponse(
        path=str(image_path),
        media_type=media_type,
        filename=name
    )

# 画像アップロード（画像ファイルとメタデータを受け取り保存）
@images_endpoint.post('/images', tags=["images"], description="画像をアップロード", responses={
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    409: {"description": "Conflict - image with the same name exists", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
async def upload_image(
    project_id: int = Form(...),
    uploaded_user_id: int = Form(...),
    file: UploadFile = File(...)
):
    """
    単一画像のアップロード（互換性維持のため）
    """
    result = await process_single_upload(project_id, uploaded_user_id, file)
    
    if result.success:
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": result.message, "data": result.data}
        )
    else:
        status_code = status.HTTP_409_CONFLICT if result.error_type == "ConflictError" else status.HTTP_400_BAD_REQUEST
        if result.error_type in ["DatabaseConnectionError", "DatabaseInsertError"]:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        return JSONResponse(
            status_code=status_code,
            content={"message": result.message, "data": result.data}
        )

# バッチアップロード用エンドポイント
@images_endpoint.post('/images/batch', tags=["images"], description="複数画像の並列アップロード", responses={
    200: {"description": "Batch upload completed", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
async def batch_upload_images(
    project_id: int = Form(...),
    uploaded_user_id: int = Form(...),
    files: List[UploadFile] = File(...),
    max_concurrent: int = Form(default=3),  # 最大同時実行数
    upload_delay: float = Form(default=0.1)  # アップロード間隔（秒）
):
    """
    複数画像の並列アップロード
    - max_concurrent: 最大同時実行数（デフォルト3）
    - upload_delay: 各アップロード間の遅延時間（デフォルト0.1秒）
    """
    if not files:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "ファイルが指定されていません", "data": None}
        )
    
    if len(files) > 50:  # 最大50ファイルまで
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "一度にアップロードできるファイル数は50個までです", "data": None}
        )
    
    # セマフォで同時実行数を制限
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_upload(file: UploadFile, index: int) -> UploadResult:
        async with semaphore:
            delay = index * upload_delay  # インデックスに応じて遅延
            return await process_single_upload(project_id, uploaded_user_id, file, delay)
    
    start_time = time.time()
    
    # 並列でアップロード処理を実行
    tasks = [limited_upload(file, i) for i, file in enumerate(files)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 結果を集計
    success_count = 0
    failure_count = 0
    success_results = []
    failure_results = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failure_count += 1
            failure_results.append({
                "filename": files[i].filename if i < len(files) else "unknown",
                "message": str(result),
                "error_type": type(result).__name__,
                "data": {}
            })
        elif isinstance(result, UploadResult):
            if result.success:
                success_count += 1
                success_results.append({
                    "filename": result.filename,
                    "data": result.data
                })
            else:
                failure_count += 1
                failure_results.append({
                    "filename": result.filename,
                    "message": result.message,
                    "error_type": result.error_type,
                    "data": result.data
                })
        else:
            # 予期しない結果タイプの場合
            failure_count += 1
            failure_results.append({
                "filename": files[i].filename if i < len(files) else "unknown",
                "message": "予期しない結果タイプ",
                "error_type": "UnexpectedResultType",
                "data": {}
            })
    
    total_time = round(time.time() - start_time, 2)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"バッチアップロード完了: 成功 {success_count}件, 失敗 {failure_count}件",
            "data": {
                "total_files": len(files),
                "success_count": success_count,
                "failure_count": failure_count,
                "total_processing_time": total_time,
                "settings": {
                    "max_concurrent": max_concurrent,
                    "upload_delay": upload_delay
                },
                "success_results": success_results,
                "failure_results": failure_results
            }
        }
    )

# アップロード進捗状況取得用エンドポイント（将来的に使用）
@images_endpoint.get('/images/upload-status/{batch_id}', tags=["images"], description="バッチアップロードの進捗状況を取得")
async def get_upload_status(batch_id: str):
    """
    バッチアップロードの進捗状況を取得
    """
    if batch_id in upload_status_cache:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "進捗状況を取得しました", "data": upload_status_cache[batch_id]}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "指定されたバッチIDが見つかりません", "data": None}
        )

# 画像削除
@images_endpoint.delete('/images/{image_id}', tags=["images"], description="画像を削除", responses={
    204: {"description": "No Content"},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def delete_image(image_id: str):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})

    query_text = f"DELETE FROM images WHERE id = '{image_id}';"
    result, _ = execute_query(connect_session, query_text)
    if result:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to delete image", "data": None})