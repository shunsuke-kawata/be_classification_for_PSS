import base64
from datetime import datetime
import mimetypes
import os
import sys
import json
from fastapi import APIRouter, Response, UploadFile, File, Form, status
from fastapi.responses import FileResponse, JSONResponse

current_dir = os.path.dirname(__file__)  # = subfolder/
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # = your_project/
clustering_path = os.path.join(parent_dir, "clustering")
# from backend.clustering.caption_manager import CaptionManager
from clustering.caption_manager import CaptionManager
from db_utils.commons import create_connect_session, execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewImage
from utils.utils import generate_uuid, image2png
from pathlib import Path
from config import DEFAULT_IMAGE_PATH ,OPENAI_API_KEY
from clustering.caption_manager import CaptionManager
from clustering.chroma_db_manager import ChromaDBManager
from clustering.embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager
from clustering.embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from clustering.utils import Utils

images_endpoint = APIRouter()

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
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})

    try:

        # プロジェクトのoriginal_images_folder_pathを取得
        query_text = f"""
            SELECT original_images_folder_path FROM projects WHERE id = {project_id};
        """
        result, _ = execute_query(connect_session, query_text)
        
        # プロジェクトが存在しない場合の処理
        if not result or result.rowcount == 0:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "project not found", "data": None})

        original_images_folder_path = result.mappings().first()["original_images_folder_path"]
        save_dir = Path(DEFAULT_IMAGE_PATH) / original_images_folder_path
        os.makedirs(save_dir, exist_ok=True)

        contents = await file.read()
        filename, _ = os.path.splitext(file.filename)
        png_path = f"{filename}.png"
        escaped_png_path = png_path.replace("'", "''")  # SQLインジェクション対策

        # 同名画像が既に存在するか確認
        check_query = f"""
            SELECT id FROM images WHERE name = '{escaped_png_path}' AND project_id = {project_id};
        """
        result, _ = execute_query(connect_session, check_query)
        if result and result.rowcount > 0:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"message": "image with the same name already exists in this project", "data": None}
            )

        # 画像をPNGに変換して保存
        png_bytes = image2png(contents)
        save_path = save_dir / png_path
        with open(save_path, "wb") as f:
            f.write(png_bytes)

        # 仮のキャプション生成
        is_created, created_caption = Utils.get_exmaple_caption(png_path)

        # ベクトルDBへ登録
        sentence_db_manager = ChromaDBManager("sentence_embeddings")
        image_db_manager = ChromaDBManager("image_embeddings")
        
        # id_exists = sentence_db_manager.()
        
        #生成されたキャプションから文章特徴量をデータベースに保存
        chromadb_sentence_id = sentence_db_manager.add_one(
            document=created_caption,
            metadata=ChromaDBManager.ChromaMetaData(
                path=png_path,
                document=created_caption,
                is_success=is_created
            ),
            embeddings=SentenceEmbeddingsManager.sentence_to_embedding(created_caption)
        )
        
        #保存された画像から画像特徴量をデータベースに保存
        chromadb_image_id = image_db_manager.add_one(
            document=created_caption,
            metadata=ChromaDBManager.ChromaMetaData(
                path=png_path,
                document=created_caption,
                is_success=is_created
            ),
            embeddings=ImageEmbeddingsManager.image_to_embedding(save_path)
        )
        
        is_created_for_sql = 'TRUE' if is_created else 'FALSE'
        escaped_caption = created_caption.replace("'", "''") if is_created else "NULL"

        clustering_id = Utils.generate_uuid()
        insert_query = f"""
            INSERT INTO images(name, is_created_caption, caption, project_id, clustering_id,chromadb_sentence_id, chromadb_image_id, uploaded_user_id)
            VALUES ('{escaped_png_path}', {is_created_for_sql}, {'NULL' if not is_created else f"'{escaped_caption}'"}, {project_id}, '{clustering_id}','{chromadb_sentence_id}','{chromadb_image_id}', '{uploaded_user_id}');
        """
        result, _ = execute_query(connect_session, insert_query)

        if result:
            return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content={"message": "succeeded to upload image", "data": {"chromadb_sentence_id": chromadb_sentence_id,"chromadb_image_id":chromadb_image_id}}
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": "failed to insert into DB", "data": None}
            )

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"upload failed: {str(e)}", "data": None}
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