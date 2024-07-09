import config
from fastapi import FastAPI, Response, status
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from routers.users import users_endpoint
from routers.projects import projects_endpoint
from routers.project_memberships import project_memberships_endpoint
from routers.auth import auth_endpoint
import json
from routers.systems import HTML_TEMPLATE
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


#CORSの設定
origins = [
    f"http://localhost:{config.FRONTEND_PORT}",
    f"http://localhost:{config.BACKEND_PORT}",
]

#アップインスタンスの作成
app = FastAPI()

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

#バックエンドエンドポイントルート
@app.get("/")
def root():
    return {"root": "be-pss"}

#appのインポートが必要になるためルートに記述
@app.get("/system/docs/update")
def update_docs_html():
    try:
        with open(f"index.html", "w") as fd:
            print(HTML_TEMPLATE % json.dumps(app.openapi()), file=fd)
        return Response(status_code=status.HTTP_200_OK,content=json.dumps({"message":"succeeded to create api docs"}))
    except Exception as e:
        print(e)
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,content=json.dumps({"message":"failed to create api docs"}))
        
        

#アプリの起動
if __name__ == "__main__":
    uvicorn.run("server:app",host="0.0.0.0",port=int(config.BACKEND_PORT),reload=True)