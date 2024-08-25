import json
from fastapi import APIRouter, File, Form, UploadFile, status
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewImage
from utils.utils import create_s3_prefix

#分割したエンドポイントの作成
images_endpoint = APIRouter()

#プロジェクトの作成
@images_endpoint.post('/images/{project_id}',tags=["images"],description="新規画像の追加",responses={
    201: {"description": "Created", "model": CustomResponseModel},
    400: {"description": "Bad Request", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
async def create_image(project_id: str,
    name: str = Form(...),
    project_id_form: int = Form(...),
    image_file: UploadFile = File(...),):
    connect_session = create_connect_session()
    #データベース接続確認
    if connect_session is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to connect to database", "data":None})
    
    # #バリデーションの実行
    # if not(validate_data(image, 'image')):
    #     return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})
    
    #バリデーションの実行
    if not(project_id==str(project_id_form)):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})
    
    query_text =f"SELECT root_folder_path,images_folder_path,object_images_folder_path FROM projects WHERE id ={project_id};"    
    result,_ = execute_query(session=connect_session,query_text=query_text)
    if (result is not None):
        rows = result.mappings().all()
        project_info = [dict(row) for row in rows][0]
        root_folder_path = project_info['root_folder_path']
        images_folder_path = project_info['images_folder_path']
        prefix_path = f'{root_folder_path}/{images_folder_path}/{name}'
        image_bytes = await image_file.read()
        is_successed_objects = create_s3_prefix(prefix_str=prefix_path,byte_data=image_bytes)
    
    #SQLの実行
    # query_text =f"INSERT INTO project_memberships(user_id, project_id) VALUES ('{project_membership.user_id}','{project_membership.project_id}');"