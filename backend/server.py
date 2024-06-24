import config
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from routers.users import users_endpoint
from routers.projects import projects_endpoint

#CORSの設定
origins = [
    "http://localhost",
]

#アップの作成
app = FastAPI()

#ミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#エンドポイントの追加
app.include_router(users_endpoint)
app.include_router(projects_endpoint)

@app.get("/")
def root():
    return {"root": "be-pss"}

if __name__ == "__main__":
    uvicorn.run("server:app",host="0.0.0.0",port=int(config.BACKEND_PORT),reload=True)