import json
from fastapi import APIRouter, File, Form, UploadFile, status
import sys
import os
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from db_utils.commons import create_connect_session,execute_query
from db_utils.validators import validate_data
from db_utils.models import CustomResponseModel, NewImage
from utils.utils import create_s3_prefix,image2png

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
        
    #バリデーションの実行
    if not(project_id==str(project_id_form)):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,content={"message": "failed to validate", "data":None})
    
    query_text =f"SELECT root_folder_path,images_folder_path,object_images_folder_path FROM projects WHERE id ={project_id};"    
    project_result,_ = execute_query(session=connect_session,query_text=query_text)
    if (project_result is not None):
        rows = project_result.mappings().all()
        project_info = [dict(row) for row in rows][0]
        root_folder_path = project_info['root_folder_path']
        images_folder_path = project_info['images_folder_path']
        basename_without_ext = os.path.splitext(os.path.basename(name))[0]
        prefix_path = f'{root_folder_path}/{images_folder_path}/{basename_without_ext}.png'
        image_bytes = await image_file.read()
        image_png_data = image2png(image_bytes)
        is_successed_image = create_s3_prefix(prefix_str=prefix_path,byte_data=image_png_data)
        
        if(is_successed_image):
            query_text =f"INSERT INTO images(name,path,project_id) VALUES('{basename_without_ext}.png','{prefix_path}','{project_id}');"    
            result,new_image_id = execute_query(session=connect_session,query_text=query_text)
            if (result is not None):
                return JSONResponse(status_code=status.HTTP_201_CREATED,content={"message": "succeeded to create project", "data":{'image_id':new_image_id}})
            else:
                return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to create image data", "data":None})    
        else:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to upload image", "data":None})