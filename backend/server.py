from fastapi.responses import JSONResponse
from config import FRONTEND_PORT,BACKEND_PORT,DEFAULT_IMAGE_PATH
from fastapi import FastAPI, status
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from db_utils.models import CustomResponseModel
from routers.users import users_endpoint
from routers.projects import projects_endpoint
from routers.project_memberships import project_memberships_endpoint
from routers.auth import auth_endpoint
from routers.systems import systems_endpoint
from routers.images import images_endpoint
from routers.test import test_endpoint
import json
from routers.systems import HTML_TEMPLATE
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pathlib import Path

#CORSの設定
origins = [
    f"http://localhost:{FRONTEND_PORT}",
    f"http://localhost:{BACKEND_PORT}",
]

openapi_tags_metadata = [
    {
        "name":"users",
        "description":"ユーザ関連処理"
    },
    {
        "name":"projects",
        "description":"プロジェクト関連処理"
    },
    {
        "name":"project_memberships",
        "description":"ユーザ・プロジェクト間の連携処理"
    },
    {
        "name":"auth",
        "description":"ユーザ認証関連処理"
    },
    {
        "name":"systems",
        "description":"開発者用の処理　ユーザは使用しない"
    } 
]

#アップインスタンスの作成
app = FastAPI(title='be_classification_for_PSS',description='classification_for_PSSのバックエンドAPIエンドポイント一覧',openapi_tags=openapi_tags_metadata)

# CORSを回避するために追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  
    allow_methods=["*"],      
    allow_headers=["*"]       
)

#分割したエンドポイントの追加
app.include_router(users_endpoint)
app.include_router(projects_endpoint)
app.include_router(project_memberships_endpoint)
app.include_router(auth_endpoint)
app.include_router(systems_endpoint)
app.include_router(images_endpoint)
app.include_router(test_endpoint)

#バックエンドエンドポイントルート
@app.get("/",tags=["systems"],description="特に使用しない")
def root():
    return {"root": "be-pss"}

#appのインポートが必要になるためルートに記述
@app.get("/system/docs/update",tags=["systems"],description="デプロイするdocsを更新する",responses={
    200: {"description": "OK", "model": CustomResponseModel},
    500: {"description": "Internal Server Error", "model": CustomResponseModel}
})
def update_docs_html():
    try:
        with open(f"index.html", "w") as fd:
            print(HTML_TEMPLATE % json.dumps(app.openapi()), file=fd)
        return JSONResponse(status_code=status.HTTP_200_OK,content={"message": "succeeded to update docs", "data":None})
    except Exception as e:
        print(e)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content={"message": "failed to update docs", "data":None})
        
#アプリの起動
if __name__ == "__main__":
    
    images_path = Path(DEFAULT_IMAGE_PATH)
    os.makedirs(images_path, exist_ok=True)
    uvicorn.run("server:app", host="0.0.0.0", port=int(BACKEND_PORT), reload=True)