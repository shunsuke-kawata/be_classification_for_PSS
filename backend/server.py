import config
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from routers.users import users_endpoint
from routers.projects import projects_endpoint


#CORSの設定
origins = [
    f"http://localhost:{config.FRONTEND_PORT}",
    f"http://localhost:{config.BACKEND_PORT}",
]

#アップの作成
app = FastAPI()

# CORSを回避するために追加（今回の肝）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # 追記により追加
    allow_methods=["*"],      # 追記により追加
    allow_headers=["*"]       # 追記により追加
)

#エンドポイントの追加
app.include_router(users_endpoint)
app.include_router(projects_endpoint)

@app.get("/")
def root():
    return {"root": "be-pss"}

if __name__ == "__main__":
    uvicorn.run("server:app",host="0.0.0.0",port=int(config.BACKEND_PORT),reload=True)