import os
import sys
import json
from fastapi import APIRouter, Response, UploadFile, File, Form, status
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session, execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewImage
from utils.utils import generate_uuid, image2png
from pathlib import Path
from config import DEFAULT_IMAGE_PATH

images_endpoint = APIRouter()

# 画像一覧取得（指定されたプロジェクトIDに紐づく画像を返す）
@images_endpoint.get('/images', tags=["images"], description="画像一覧を取得", responses={
    200: {"description": "OK", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def read_images(project_id: int = None):
    if project_id is None:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "project_id is required", "data": None})

    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})

    query_text = f"""
        SELECT id, name, project_id, uploaded_user_id,
               DATE_FORMAT(created_at, '%%Y-%%m-%%dT%%H:%%i:%%sZ') as created_at,
               DATE_FORMAT(updated_at, '%%Y-%%m-%%dT%%H:%%i:%%sZ') as updated_at
        FROM images
        WHERE project_id = {project_id};
    """
    result, _ = execute_query(connect_session, query_text)
    if result:
        rows = result.mappings().all()
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "succeeded to read images", "data": [dict(row) for row in rows]})
    else:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to read images", "data": None})


# 画像アップロード（画像ファイルとメタデータを受け取り保存）
@images_endpoint.post('/images', tags=["images"], description="画像をアップロード", responses={
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
async def upload_image(
    name: str = Form(...),
    project_id: int = Form(...),
    uploaded_user_id: int = Form(...),
    file: UploadFile = File(...)
):
    connect_session = create_connect_session()
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to connect to database", "data": None})

    try:
        
        image_id = generate_uuid()  # 画像IDを生成
        
        # プロジェクトのoriginal_images_folder_pathを取得
        query_text = f"""
            SELECT original_images_folder_path FROM projects WHERE id = {project_id};
        """
        result, _ = execute_query(connect_session, query_text)
        if not result or result.rowcount == 0:
            raise Exception("project not found")
        original_images_folder_path = result.mappings().first()["original_images_folder_path"]
        
        save_dir = Path(DEFAULT_IMAGE_PATH) / original_images_folder_path
        # PNG形式で保存
        os.makedirs(save_dir, exist_ok=True)
        
        name_only, _ = os.path.splitext(file.filename)
        contents = await file.read()
        
        png_bytes = image2png(contents)
        
        save_path = save_dir / f"{name_only}.png"
        with open(save_path, "wb") as f:
            f.write(png_bytes)
        

        query_text = f"""
            INSERT INTO images(id, name, project_id, uploaded_user_id)
            VALUES ('{image_id}', '{name}', {project_id}, {uploaded_user_id});
        """
        result, _ = execute_query(connect_session, query_text)
        if result:
            return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message": "succeeded to upload image", "data": {"image_id": image_id}})
        else:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "failed to insert into DB", "data": None})
    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": f"upload failed: {str(e)}", "data": None})


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